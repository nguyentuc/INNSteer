"""INNSteer on the Anthropic Persona behaviors (paper Table 1 / Figure 1 setting).

Pipeline per behavior:
  1. Final-token activations for 700 contrastive training pairs at every layer.
  2. Pick the steering layer with the D + C layer score (or pass --layer).
  3. Split the pairs 80/20 (seed 42), z-score with training statistics, train the INN
     (best of --restarts restarts, best held-out checkpoint), v_phi = latent mean difference.
  4. Evaluate on the 100 test questions: alignment probability (Prob) and generation PPL,
     for the unsteered model, mean-difference steering (MD / CAA) and INNSteer.

Example (paper setting for one behavior):
  python scripts/run_persona.py --model meta-llama/Meta-Llama-3-8B-Instruct \
      --tasks believes-it-has-phenomenal-consciousness --out results/llama3-8b.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from innsteer.data import PERSONA_TASKS, load_split  # noqa: E402
from innsteer.inn_evaluate import full_inn_report  # noqa: E402
from innsteer.layers import mean_difference, rank_layers  # noqa: E402
from innsteer.llm import SteerableLM  # noqa: E402
from innsteer.metrics import alignment_probability, generate_batches, generation_perplexity  # noqa: E402
from innsteer.steering import add_vector_hooks, innsteer_hooks, latent_steering_vector  # noqa: E402
from innsteer.train import (DEFAULT_INN_CFG, DEFAULT_TRAIN_CFG, split_activations,  # noqa: E402
                            train_best_inn_for_layer, zscore_normalize_eval, zscore_normalize_train)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path (LLaMA-3 / Qwen2.5 chat models)")
    p.add_argument("--tasks", nargs="+", default=list(PERSONA_TASKS), choices=list(PERSONA_TASKS))
    p.add_argument("--layer", type=int, default=None, help="steering layer; default: best D + C layer score")
    p.add_argument("--alpha", type=float, default=1.0, help="steering strength")
    p.add_argument("--seeds", type=int, nargs="+", default=[2026])
    p.add_argument("--restarts", type=int, default=3, help="INN restarts per seed; the best held-out score is kept")
    # INN / training hyperparameters (defaults = released configuration)
    p.add_argument("--num-coupling", type=int, default=DEFAULT_INN_CFG["num_layers"])
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_INN_CFG["hidden_dim"])
    p.add_argument("--dropout", type=float, default=DEFAULT_INN_CFG["dropout"])
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CFG["num_epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CFG["lr"])
    p.add_argument("--lambda-dir", type=float, default=DEFAULT_TRAIN_CFG["lambda_dir"])
    p.add_argument("--lambda-logdet", type=float, default=DEFAULT_TRAIN_CFG["lambda_logdet"])
    p.add_argument("--weight-decay", type=float, default=DEFAULT_TRAIN_CFG["weight_decay"])
    # evaluation
    p.add_argument("--no-ppl", action="store_true", help="skip generation perplexity (much faster)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-train", type=int, default=None, help="use only the first N training pairs (smoke tests)")
    p.add_argument("--max-test", type=int, default=None, help="use only the first N test questions (smoke tests)")
    p.add_argument("--dtype", default=None, choices=["auto", "float32", "bfloat16", "float16"],
                   help="model dtype; default: float32 for LLaMA, checkpoint dtype for Qwen (as in the paper)")
    p.add_argument("--out", default=None, help="write results as JSON here")
    return p.parse_args()


def main():
    args = parse_args()
    lm = SteerableLM(args.model, dtype=args.dtype)
    dev = lm.device
    print(f"model={args.model} device={dev} layers={lm.num_layers}")
    cfg_inn = dict(num_layers=args.num_coupling, hidden_dim=args.hidden_dim, dropout=args.dropout)
    cfg_train = dict(num_epochs=args.epochs, lr=args.lr, lambda_logdet=args.lambda_logdet,
                     lambda_dir=args.lambda_dir, weight_decay=args.weight_decay)
    results = {"model": args.model, "alpha": args.alpha, "inn_cfg": cfg_inn, "train_cfg": cfg_train, "tasks": {}}

    def prob_and_ppl(rows, make_hooks=None):
        import contextlib
        hooks = make_hooks or contextlib.nullcontext
        with hooks():
            prob = alignment_probability(lm, rows, args.batch_size)
        ppl = None
        if not args.no_ppl:
            with hooks():  # generate under steering ...
                gens = generate_batches(lm, rows, args.batch_size, args.max_new_tokens)
            ppl = generation_perplexity(lm, gens)  # ... score with the unsteered model
        return prob, ppl

    for task in args.tasks:
        t0 = time.time()
        print(f"\n=== {task} ({PERSONA_TASKS[task]}) ===")
        train_rows = load_split(task, "train", args.max_train)
        test_rows = load_split(task, "test", args.max_test)
        res = {"n_train_pairs": len(train_rows), "n_test": len(test_rows)}

        res["base_prob"], res["base_ppl"] = prob_and_ppl(test_rows)
        print(f"Base      prob={res['base_prob']:.4f} ppl={res['base_ppl']}")

        acts = lm.contrastive_activations(train_rows, list(range(lm.num_layers)))
        ranking = rank_layers(acts)
        layer = args.layer if args.layer is not None else ranking[0][0]
        res["layer"] = layer
        res["layer_scores"] = {l: s for l, s in ranking}
        print(f"Steering layer {layer} (D+C ranking top-3: {[l for l, _ in ranking[:3]]})")

        pos, neg = acts[layer]
        v_md = mean_difference(pos, neg)
        res["md_prob"], res["md_ppl"] = prob_and_ppl(
            test_rows, lambda: add_vector_hooks(lm, {layer: args.alpha * v_md}))
        print(f"MD (CAA)  prob={res['md_prob']:.4f} ppl={res['md_ppl']}")

        p_tr, n_tr, p_ev, n_ev = split_activations(pos, neg, eval_ratio=0.2)
        p_tr_n, n_tr_n, mu, sigma = zscore_normalize_train(p_tr, n_tr)
        p_ev_n, n_ev_n = zscore_normalize_eval(p_ev, n_ev, mu, sigma)
        layer_data = {layer: dict(acts_pos_train_n=p_tr_n, acts_neg_train_n=n_tr_n,
                                  acts_pos_eval_n=p_ev_n, acts_neg_eval_n=n_ev_n)}
        res["seeds"] = {}
        for seed in args.seeds:
            inn = train_best_inn_for_layer(layer, layer_data, cfg_inn, cfg_train, seed,
                                           num_restarts=args.restarts, device=dev)
            v_phi = latent_steering_vector(inn, p_tr_n, n_tr_n, dev)
            report = full_inn_report(inn=inn, acts_pos_train=p_ev_n, acts_neg_train=n_ev_n,
                                     v_original=v_md, acts_ood=torch.cat([p_ev_n, n_ev_n]).to(dev),
                                     device=dev, verbose=False)
            steer = {layer: dict(inn=inn, v_phi=v_phi, mu=mu, sigma=sigma, alpha=args.alpha)}
            prob, ppl = prob_and_ppl(test_rows, lambda: innsteer_hooks(lm, steer))
            res["seeds"][seed] = {"prob": prob, "ppl": ppl,
                                  "D_original": report.get("D_original"), "D_phi": report.get("D_phi"),
                                  "recon_error_mean": report.get("recon_error_mean")}
            print(f"INNSteer  seed={seed} prob={prob:.4f} ppl={ppl}  "
                  f"D: {report.get('D_original', float('nan')):.3f} -> {report.get('D_phi', float('nan')):.3f}")
        res["minutes"] = round((time.time() - t0) / 60, 2)
        results["tasks"][task] = res

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
