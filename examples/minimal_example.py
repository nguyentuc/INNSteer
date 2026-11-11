"""Minimal INNSteer example (CPU, ~1 minute, no LLM download).

Uses 2-D synthetic "activations" shaped like the paper's two-ring setup: the source
behavior lies on an outer ring and the target behavior on an inner ring, so the right
edit points inward and differs from input to input. The script

  1. trains an INN with the INNSteer objective on contrastive pairs,
  2. computes the latent steering vector v_phi (latent mean difference),
  3. checks that alpha = 0 reproduces the input exactly (invertibility),
  4. applies h' = phi^-1(phi(h) + alpha * v_phi) and compares with one global vector.

Run:  python examples/minimal_example.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from innsteer import (innsteer_transform, latent_steering_vector, split_activations,  # noqa: E402
                      train_best_inn_for_layer, zscore_normalize_eval, zscore_normalize_train)

torch.manual_seed(0)


def two_rings(n=512, r_out=2.0, r_in=1.0, gap=0.35, noise=0.04):
    theta = torch.linspace(gap, 2 * math.pi - gap, n) + 0.012 * torch.randn(n)
    ring = torch.stack([theta.cos(), theta.sin()], 1)
    target = r_in * ring + noise * torch.randn(n, 2)   # behavior-matching activations
    source = r_out * ring + noise * torch.randn(n, 2)  # non-matching activations
    return target, source


def main():
    pos, neg = two_rings()
    p_tr, n_tr, p_ev, n_ev = split_activations(pos, neg, eval_ratio=0.2)
    p_tr_n, n_tr_n, mu, sigma = zscore_normalize_train(p_tr, n_tr)
    p_ev_n, n_ev_n = zscore_normalize_eval(p_ev, n_ev, mu, sigma)
    layer_data = {0: dict(acts_pos_train_n=p_tr_n, acts_neg_train_n=n_tr_n,
                          acts_pos_eval_n=p_ev_n, acts_neg_eval_n=n_ev_n)}

    inn = train_best_inn_for_layer(
        layer_idx=0, layer_data=layer_data,
        cfg_inn=dict(num_layers=6, hidden_dim=64, dropout=0.0),
        cfg_train=dict(num_epochs=150, lr=1e-3, lambda_logdet=0.1, lambda_dir=1.0, weight_decay=1e-3),
        seed=0, num_restarts=1, device="cpu")
    v_phi = latent_steering_vector(inn, p_tr_n, n_tr_n, "cpu")

    with torch.no_grad():
        identity = innsteer_transform(inn, v_phi, mu, sigma, alpha=0.0)
        recon = (identity(n_ev) - n_ev).abs().max().item()
        steered = innsteer_transform(inn, v_phi, mu, sigma, alpha=1.0)(n_ev)

    linear = n_ev + (p_tr - n_tr).mean(0)  # one global mean-difference vector
    radius = lambda x: x.norm(dim=1)  # noqa: E731
    print(f"max |phi^-1(phi(h)) - h| at alpha=0: {recon:.2e}")
    print(f"mean radius  source: {radius(n_ev).mean():.3f}   target: {radius(p_ev).mean():.3f}")
    print(f"mean radius  INNSteer: {radius(steered).mean():.3f}   linear (global vector): {radius(linear).mean():.3f}")
    print(f"mean distance to paired target  INNSteer: {(steered - p_ev).norm(dim=1).mean():.3f}   "
          f"linear: {(linear - p_ev).norm(dim=1).mean():.3f}")
    assert recon < 1e-4, "alpha = 0 must return the input (exact invertibility)"


if __name__ == "__main__":
    main()
