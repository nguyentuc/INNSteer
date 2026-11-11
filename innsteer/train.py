"""INN training for INNSteer (paper Sec. 3 and Appendix "Optimization and Checkpoint Selection").

Objective per mini-batch:
    L = NLL + lambda_dir * L_dir + lambda_logdet * (mean(log|det J|)^2 + Var(log|det J|))
The checkpoint with the best held-out score is kept, and the best of several restarts is selected.
"""
import torch
import torch.nn as nn

from .inn import INN

# Defaults used for the released runs (see README, "Hyperparameters").
NUM_RESTARTS = 3
BATCH_SIZE = 128
WARMUP_EPOCHS = 20

DEFAULT_INN_CFG = dict(num_layers=6, hidden_dim=2048, dropout=0.3)
DEFAULT_TRAIN_CFG = dict(num_epochs=300, lr=1e-3, lambda_logdet=1.0, lambda_dir=1.0, weight_decay=1e-3)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_activations(acts_pos, acts_neg, eval_ratio=0.2, seed=42):
    """
    Deterministic pair-wise 80/20 split. Same permutation for pos and neg
    so contrastive pairs remain aligned across splits.
    """
    N = acts_pos.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm  = torch.randperm(N, generator=rng)
    n_eval = int(N * eval_ratio)
    idx_eval = perm[:n_eval]
    idx_train = perm[n_eval:]
    return (
        acts_pos[idx_train], acts_neg[idx_train],
        acts_pos[idx_eval],  acts_neg[idx_eval],
    )

# ──────────────────────────────────────────────────────────────────────────────
# Z-SCORE NORMALISATION
# ──────────────────────────────────────────────────────────────────────────────

def zscore_normalize_train(acts_pos_train, acts_neg_train):
    """Compute normalization stats from training split."""
    all_train = torch.cat([acts_pos_train, acts_neg_train], dim=0)
    mu = all_train.mean(0)
    sigma = all_train.std(0).clamp(min=1e-8)
    return (acts_pos_train - mu) / sigma, (acts_neg_train - mu) / sigma, mu, sigma

def zscore_normalize_eval(acts_pos_eval, acts_neg_eval, mu, sigma):
    """Apply training-set statistics to the eval split."""
    return (acts_pos_eval - mu) / sigma, (acts_neg_eval - mu) / sigma


# ──────────────────────────────────────────────────────────────────────────────
# INN TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_inn_for_layer(
    acts_pos_train,
    acts_neg_train,
    acts_pos_eval,
    acts_neg_eval,
    layer,
    cfg_inn,
    cfg_train,
    seed=2026,
    verbose=False,
    device=None,
):
    """
    Train one INN with the three-fix objective.
    Objective per mini-batch:
        L = NLL + lambda_dir * dir_loss + lambda_logdet * (mean_ld^2 + var_ld)

    where:
        NLL = -(log p(z_pos) + log p(z_neg)) / (2 * dim)
        dir_loss = -||mu_pos - mu_neg|| / sqrt(dim) 
        var_ld   = variance of per-sample log|det J| 
    Checkpoint: highest eval_score on the held-out split. 
    """
    DEVICE = device or default_device()
    torch.manual_seed(seed + layer)
    torch.cuda.manual_seed(seed + layer)

    num_epochs = cfg_train["num_epochs"]
    lambda_logdet = cfg_train["lambda_logdet"]
    lambda_dir = cfg_train["lambda_dir"]

    dim = acts_pos_train.shape[1]
    N_tr = acts_pos_train.shape[0]
    pos_tr = acts_pos_train.to(DEVICE).float()
    neg_tr = acts_neg_train.to(DEVICE).float()
    pos_ev = acts_pos_eval.to(DEVICE).float()
    neg_ev = acts_neg_eval.to(DEVICE).float()

    inn = INN(dim=dim, **cfg_inn).to(DEVICE)

    # ── Optimizer: AdamW (decoupled weight decay) ─────────────────────────
    opt = torch.optim.AdamW(
        inn.parameters(),
        lr = cfg_train["lr"],
        weight_decay = cfg_train["weight_decay"],
    )

    # ── LR schedule: linear warmup => cosine decay ─────────────────────────
    warmup_epochs = min(WARMUP_EPOCHS, num_epochs // 10)
    cosine_epochs = num_epochs - warmup_epochs
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max = cosine_epochs,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
    )

    batch_size = min(BATCH_SIZE, N_tr)
    n_batches = max(1, N_tr // batch_size)

    best_eval_score = -float("inf")
    best_state = None
    best_epoch = -1

    # Separate RNG for per-epoch shuffling so init seed and shuffle are independent
    shuffle_rng = torch.Generator(device="cpu")
    shuffle_rng.manual_seed(seed + layer + 999)

    for epoch in range(num_epochs):
        inn.train()
        perm = torch.randperm(N_tr, generator=shuffle_rng)
        pos_shuf = pos_tr[perm]
        neg_shuf = neg_tr[perm]
        n_valid_batches = 0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, N_tr)
            pos_b = pos_shuf[start:end]
            neg_b = neg_shuf[start:end]

            opt.zero_grad()
            z_pos, ld_pos = inn(pos_b)   # z: (B, dim)  ld: (B,)
            z_neg, ld_neg = inn(neg_b)

            # ── Gaussian NLL ──────────────────────────────────────
            # log p(z) = -0.5*||z||^2 + log|det J|  (constant dropped)
            log_pz_pos = -0.5 * (z_pos ** 2).sum(dim=1) + ld_pos   # (B,)
            log_pz_neg = -0.5 * (z_neg ** 2).sum(dim=1) + ld_neg   # (B,)
            nll_loss = -(log_pz_pos.mean() + log_pz_neg.mean()) / (2.0 * dim)

            # ── Directional separation loss ───────────────────────
            # Maximize ||mu_pos - mu_neg|| so the mean-difference vector in
            mu_pos = z_pos.mean(0)
            mu_neg = z_neg.mean(0)
            v_phi = mu_pos - mu_neg
            dir_loss = -v_phi.norm() / (dim ** 0.5)

            # ── Logdet mean + variance penalty ─────────────────────
            # mean_ld^2 prevents volume collapse / explosion.
            # var_ld    prevents per-sample log-det spread that causes PPL
            all_ld = torch.cat([ld_pos, ld_neg])          # (2B,)
            mean_ld = all_ld.mean()
            var_ld = all_ld.var() if all_ld.shape[0] > 1 \
                          else torch.zeros(1, device=DEVICE)
            logdet_loss = lambda_logdet * (mean_ld ** 2 + var_ld)

            loss = nll_loss + lambda_dir * dir_loss + logdet_loss

            if torch.isnan(loss):
                if verbose:
                    print(f"WARNING: NaN at epoch {epoch+1} batch {b}, skipping")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(inn.parameters(), max_norm=1.0)
            opt.step()
            n_valid_batches += 1

        sched.step()

        if n_valid_batches == 0:
            continue

        # ── Eval checkpoint on held-out split ─────────────────────────────
        # Criterion: -eval_nll + lambda_dir * eval_dir
        # Higher is better — matches the restart selection criterion so
        # train_best_inn_for_layer() picks consistently.
        inn.eval()
        with torch.no_grad():
            z_pos_e, ld_pos_e = inn(pos_ev)
            z_neg_e, ld_neg_e = inn(neg_ev)

            log_pz_pos_e = -0.5 * (z_pos_e ** 2).sum(dim=1) + ld_pos_e
            log_pz_neg_e = -0.5 * (z_neg_e ** 2).sum(dim=1) + ld_neg_e
            eval_nll = -(log_pz_pos_e.mean() + log_pz_neg_e.mean()) / (2.0 * dim)

            v_phi_e = z_pos_e.mean(0) - z_neg_e.mean(0)
            eval_dir = v_phi_e.norm() / (dim ** 0.5)

            eval_score = -eval_nll.item() + lambda_dir * eval_dir.item()

        if eval_score > best_eval_score:
            best_eval_score = eval_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in inn.state_dict().items()}

        if verbose and (epoch + 1) % 50 == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"Layer {layer} | ep {epoch+1:>3}/{num_epochs} "
                f"| lr={cur_lr:.2e} "
                f"| nll={eval_nll.item():.4f}  dir={eval_dir.item():.4f} "
                f"| eval_score={eval_score:.4f}"
            )

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint found for layer {layer}")
    if verbose:
        print(f"Layer {layer} | best epoch={best_epoch} "
              f"| eval_score={best_eval_score:.4f}")
    inn.load_state_dict(best_state)
    inn.eval()
    return inn, best_eval_score


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-RESTART SELECTION
# ──────────────────────────────────────────────────────────────────────────────
def train_best_inn_for_layer(
    layer_idx,
    layer_data,
    cfg_inn,
    cfg_train,
    seed,
    num_restarts=NUM_RESTARTS,
    verbose=False,
    device=None,
):
    d = layer_data[layer_idx]
    best_score = -float("inf")
    best_inn = None

    print(f"[Restarts {num_restarts}X] layer={layer_idx} seed={seed}")

    for restart in range(num_restarts):
        sub_seed = seed * 100 + restart
        try:
            inn, eval_score = train_inn_for_layer(
                acts_pos_train = d["acts_pos_train_n"],
                acts_neg_train = d["acts_neg_train_n"],
                acts_pos_eval = d["acts_pos_eval_n"],
                acts_neg_eval = d["acts_neg_eval_n"],
                layer = layer_idx,
                cfg_inn = cfg_inn,
                cfg_train = cfg_train,
                seed = sub_seed,
                verbose = verbose,
                device = device,
            )
        except RuntimeError as e:
            print(f"Restart {restart} (sub_seed={sub_seed}) FAILED: {e}")
            continue

        print(f"Restart {restart} | sub_seed={sub_seed} | "
              f"eval_score={eval_score:.4f}")

        if eval_score > best_score:
            best_score = eval_score
            best_inn = inn

    if best_inn is None:
        raise RuntimeError(
            f"All {num_restarts} restarts failed for layer {layer_idx}, seed {seed}"
        )

    print(f"=> selected restart with eval_score={best_score:.4f}")
    return best_inn
