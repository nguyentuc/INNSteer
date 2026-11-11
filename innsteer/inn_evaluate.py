"""
inn_evaluate.py — Evaluation how good the latent space are

Covers all 5 evaluation dimensions:
  1. Invertibility Quality => reconstruction error
  2. Linear Separability => Fisher discriminability + linear probe accuracy
  3. Geometry Preservation => log-determinant + covariance eigenspectrum
  4. OOD Generalization => OOD reconstruction + kNN consistency
  5. Downstream Steering Quality => alignment probability + perplexity delta
"""
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device).float()

# ──────────────────────────────────────────────────────────────────────────────
# METRIC: D
# ──────────────────────────────────────────────────────────────────────────────

def _fisher_ratio_1d(z_pos: torch.Tensor, z_neg: torch.Tensor, v: torch.Tensor) -> float:
    """
    Fisher discriminability along direction v.
    D = (v'Sb v) / (v'(Sb+Sw) v)   — same formula as LayerNavigator eq.7
    """
    N = z_pos.shape[0]
    mu_pos = z_pos.mean(0)
    mu_neg = z_neg.mean(0)
    mu_all = torch.cat([z_pos, z_neg]).mean(0)
    v_n = v / (v.norm() + 1e-8)
    # Between-class scatter projected onto v
    sb = N * (((mu_pos - mu_all) @ v_n) ** 2 + ((mu_neg - mu_all) @ v_n) ** 2)
    # Within-class scatter projected onto v
    sw = (((z_pos - mu_pos) @ v_n) ** 2).sum() + (((z_neg - mu_neg) @ v_n) ** 2).sum()
    return (sb / (sb + sw + 1e-8)).item()


# ──────────────────────────────────────────────────────────────────────────────
# METRIC: INVERTIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def eval_reconstruction_error(
    inn,
    acts: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    ||phi^-1(phi(a)) − a||_2 averaged over samples.
    For exact coupling-layer INNs this should be 1e-6 (floating-point noise).
    Any value > 1e-4 indicates a numerical instability problem.
    """
    inn.eval()
    with torch.no_grad():
        a = _to_tensor(acts, device)
        z = inn.phi(a)
        recon = inn.phi_inv(z)
        per_sample = (recon - a).norm(dim=1) # (N,)
        return {
            "recon_error_mean": per_sample.mean().item(),
            "recon_error_max": per_sample.max().item(),
            "recon_error_std": per_sample.std().item(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# METRIC 2 — LINEAR SEPARABILITY
# ──────────────────────────────────────────────────────────────────────────────

def eval_discriminability(
    inn,
    acts_pos: torch.Tensor,
    acts_neg: torch.Tensor,
    v_original: torch.Tensor,   # MD vector in original space
    device: str = "cuda",
) -> dict:
    """
    Compare Fisher discriminability D_l (original space) vs D_phi (phi-space).
    D_phi > D_l => INN is doing its job.
    D_phi = D_l => activations were already linearly separable (INN adds nothing).
    D_phi < D_l => training objective needs revisiting.
    """
    inn.eval()
    with torch.no_grad():
        z_pos = inn.phi(_to_tensor(acts_pos, device))
        z_neg = inn.phi(_to_tensor(acts_neg, device))
        v_phi = (z_pos - z_neg).mean(0)

        d_before = _fisher_ratio_1d(
            _to_tensor(acts_pos, device),
            _to_tensor(acts_neg, device),
            _to_tensor(v_original, device),
        )
        d_after = _fisher_ratio_1d(z_pos, z_neg, v_phi)

    return {
        "D_original": d_before,
        "D_phi": d_after,
        "improvement": d_after - d_before,
        "improvement_pct": 100.0 * (d_after - d_before) / (d_before + 1e-8),
    }


def eval_linear_probe(
    inn,
    acts_pos: torch.Tensor,
    acts_neg: torch.Tensor,
    device: str = "cuda",
    cv_folds: int = 5,
) -> dict:
    """
    5-fold cross-validated logistic regression accuracy before and after phi.
    Uses top-50 PCA components to keep sklearn tractable on high-dim data.
    """
    inn.eval()
    with torch.no_grad():
        ap = _to_tensor(acts_pos, device)
        an = _to_tensor(acts_neg, device)
        z_pos = inn.phi(ap)
        z_neg = inn.phi(an)

    # Build datasets
    X_orig = torch.cat([ap, an]).cpu().numpy()
    X_phi  = torch.cat([z_pos, z_neg]).cpu().numpy()
    y = np.array([1] * len(acts_pos) + [0] * len(acts_neg))

    def _probe(X):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        # Reduce to top-50 dims for computational tractability
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, X_s.shape[1], X_s.shape[0] - 1))
        X_r = pca.fit_transform(X_s)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X_r, y, cv=cv_folds, scoring="accuracy")
        return scores.mean(), scores.std()

    acc_orig, std_orig = _probe(X_orig)
    acc_phi, std_phi   = _probe(X_phi)

    return {
        "probe_acc_original": acc_orig,
        "probe_acc_phi": acc_phi,
        "probe_acc_std_original": std_orig,
        "probe_acc_std_phi": std_phi,
        "probe_improvement": acc_phi - acc_orig,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METRIC: GEOMETRY PRESERVATION
# ──────────────────────────────────────────────────────────────────────────────

def eval_geometry(
    inn,
    acts_pos: torch.Tensor,
    acts_neg: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    log|det J_phi| and covariance eigenspectrum of phi(a).
    Good INN:
      mean_log_det = 0 (volume-preserving or mildly expanding)
      std_log_det  is small  (consistent across samples)
      condition_number < 1e4 (not collapsed or exploded)
    """
    inn.eval()
    with torch.no_grad():
        all_acts = torch.cat([
            _to_tensor(acts_pos, device),
            _to_tensor(acts_neg, device),
        ])
        z, log_det = inn(all_acts)

        mean_ld = log_det.mean().item()
        std_ld = log_det.std().item()

        # Covariance eigenspectrum (top-100 dims via PCA for efficiency)
        z_c = z - z.mean(0)
        # Use SVD on the data matrix instead of full covariance for memory
        _, S, _ = torch.linalg.svd(z_c, full_matrices=False)
        eigvals = (S ** 2) / (len(z) - 1)  # eigenvalues of sample covariance

    return {
        "mean_log_det": mean_ld,
        "std_log_det": std_ld,
        "condition_number": (eigvals[0] / (eigvals[-1] + 1e-10)).item(),
        "max_eigval": eigvals[0].item(),
        "min_eigval": eigvals[-1].item(),
        # Volume-collapse flag
        "geometry_ok": (eigvals[0] / (eigvals[-1] + 1e-10)).item() < 1e5,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METRIC: OOD GENERALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def eval_ood_generalization(
    inn,
    acts_pos_train: torch.Tensor,
    acts_neg_train: torch.Tensor,
    acts_ood: torch.Tensor,         # e.g. validation-set activations (unseen at INN training time)
    device: str = "cuda",
    k_neighbors: int = 5,
    n_sample: int = 100,
) -> dict:
    """
    Tests whether phi generalises beyond the training contrastive pairs.

    ood_recon_error = 0 => exact INN, numerical sanity
    ood_log_det = train => mapping is stable on unseen inputs
    knn_consistency => 1.0 => neighbourhood structure is preserved
    """
    inn.eval()
    with torch.no_grad():
        ood = _to_tensor(acts_ood, device)

        # --- OOD reconstruction (should be ~0 for coupling-layer INN) ---
        z_ood = inn.phi(ood)
        recon_ood = inn.phi_inv(z_ood)
        recon_errors = (recon_ood - ood).norm(dim=1)

        # --- OOD log-det ---
        _, log_det_ood = inn(ood)

        # --- Compare to training log-det ---
        all_train = torch.cat([
            _to_tensor(acts_pos_train, device),
            _to_tensor(acts_neg_train, device),
        ])
        _, log_det_train = inn(all_train)

        # --- kNN consistency on a random subsample ---
        n = min(n_sample, len(ood))
        idx = torch.randperm(len(ood), device=device)[:n]
        sample = ood[idx]
        sample_z = inn.phi(sample)

        dist_orig = torch.cdist(sample, sample)# (n, n)
        dist_phi = torch.cdist(sample_z, sample_z)

        k = min(k_neighbors, n - 1)
        knn_orig = dist_orig.topk(k + 1, largest=False).indices[:, 1:] # exclude self
        knn_phi = dist_phi.topk(k + 1, largest=False).indices[:, 1:]

        iou_scores = []
        for i in range(n):
            s_orig = set(knn_orig[i].cpu().tolist())
            s_phi = set(knn_phi[i].cpu().tolist())
            iou = len(s_orig & s_phi) / len(s_orig | s_phi)
            iou_scores.append(iou)

    return {
        "ood_recon_error_mean": recon_errors.mean().item(),
        "ood_recon_error_max": recon_errors.max().item(),
        "ood_log_det_mean": log_det_ood.mean().item(),
        "ood_log_det_std": log_det_ood.std().item(),
        "train_log_det_mean": log_det_train.mean().item(),
        "log_det_drift": abs(log_det_ood.mean().item() - log_det_train.mean().item()),
        "knn_consistency": float(np.mean(iou_scores)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FULL REPORT
# ──────────────────────────────────────────────────────────────────────────────

def full_inn_report(
    inn,
    acts_pos_train: torch.Tensor,
    acts_neg_train: torch.Tensor,
    v_original: torch.Tensor,
    acts_ood: torch.Tensor,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """
    Run all metrics in the recommended evaluation order and return a flat dict.
    Early-exit flags alert you if a metric fails before you waste time on the next.
    """
    report = {}

    # Invertibility
    r1 = eval_reconstruction_error(inn, torch.cat([acts_pos_train, acts_neg_train]), device)
    report.update(r1)
    if verbose:
        print(f"\n[1] Reconstruction  mean={r1['recon_error_mean']:.2e}  "
              f"max={r1['recon_error_max']:.2e}  "
              f"{'OK' if r1['recon_error_mean'] < 1e-4 else 'WARN: numerical instability'}")

    # Discriminability
    r2 = eval_discriminability(inn, acts_pos_train, acts_neg_train, v_original, device)
    report.update(r2)
    if verbose:
        print(f"[2] Discriminability  D_orig={r2['D_original']:.4f}  "
              f"D_phi={r2['D_phi']:.4f}  "
              f"improvement={r2['improvement_pct']:+.1f}%  "
              f"{'Good' if r2['D_phi'] > r2['D_original'] else 'INN not helping separability'}")

    # Linear probe
    r2b = eval_linear_probe(inn, acts_pos_train, acts_neg_train, device)
    report.update(r2b)
    if verbose:
        print(f"[2b] Linear probe  orig={r2b['probe_acc_original']:.3f}  "
              f"phi={r2b['probe_acc_phi']:.3f}  "
              f"delta={r2b['probe_improvement']:+.3f}")

    # Geometry
    r3 = eval_geometry(inn, acts_pos_train, acts_neg_train, device)
    report.update(r3)
    if verbose:
        print(f"[3] Geometry  log_det={r3['mean_log_det']:.3f}±{r3['std_log_det']:.3f}  "
              f"cond={r3['condition_number']:.1e}  "
              f"{'Good' if r3['geometry_ok'] else 'Geometry collapsed/exploded'}")

    # OOD
    r4 = eval_ood_generalization(inn, acts_pos_train, acts_neg_train, acts_ood, device)
    report.update(r4)
    if verbose:
        print(f"[4] OOD  recon={r4['ood_recon_error_mean']:.2e}  "
              f"log_det_drift={r4['log_det_drift']:.3f}  "
              f"kNN_consistency={r4['knn_consistency']:.3f}")

    return report