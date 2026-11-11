"""Fast CPU checks: invertibility, the steering transform, layer scores, and INN training."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from innsteer import INN, innsteer_transform, train_inn_for_layer  # noqa: E402
from innsteer.data import PERSONA_TASKS, load_split  # noqa: E402
from innsteer.layers import layer_scores  # noqa: E402


def test_inn_exact_inverse():
    torch.manual_seed(0)
    inn = INN(dim=33, num_layers=4, hidden_dim=32).eval()
    for p in inn.parameters():  # move away from the identity initialisation
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(64, 33)
    with torch.no_grad():
        z, log_det = inn(x)
        assert (inn.phi_inv(z) - x).abs().max() < 1e-4
        assert log_det.shape == (64,)


def test_alpha_zero_is_identity():
    torch.manual_seed(0)
    inn = INN(dim=16, num_layers=3, hidden_dim=16).eval()
    h = torch.randn(2, 5, 16)
    f = innsteer_transform(inn, torch.randn(16), torch.zeros(16), torch.ones(16), alpha=0.0)
    with torch.no_grad():
        assert torch.allclose(f(h), h, atol=1e-5)


def test_layer_scores_match_dense_formula():
    torch.manual_seed(0)
    pos, neg = torch.randn(50, 8) + 0.5, torch.randn(50, 8)
    s = layer_scores(pos, neg)
    v = (pos - neg).mean(0)
    v = v / v.norm()
    x = torch.cat([pos, neg])
    x = (x - x.mean(0)) / x.std(0)
    m, sb, sw = x.mean(0), torch.zeros(8, 8), torch.zeros(8, 8)
    for g in (x[:50], x[50:]):
        d = g - g.mean(0)
        sw += d.T @ d
        sb += len(g) * torch.outer(g.mean(0) - m, g.mean(0) - m)
    d_dense = (v @ sb @ v) / (v @ (sb + sw) @ v)
    assert abs(s["d_score"] - d_dense.item()) < 1e-5
    assert 0 <= s["d_score"] <= 1 and -1 <= s["c_score"] <= 1


def test_training_runs_and_separates():
    torch.manual_seed(0)
    pos, neg = torch.randn(80, 6) + 1.0, torch.randn(80, 6)
    inn, score = train_inn_for_layer(pos[:64], neg[:64], pos[64:], neg[64:], layer=0,
                                     cfg_inn=dict(num_layers=2, hidden_dim=16, dropout=0.0),
                                     cfg_train=dict(num_epochs=20, lr=1e-3, lambda_logdet=1.0,
                                                    lambda_dir=1.0, weight_decay=1e-3),
                                     seed=0, device="cpu")
    assert torch.isfinite(torch.tensor(score))


def test_persona_data_present():
    for task in PERSONA_TASKS:
        assert len(load_split(task, "train")) == 700
        assert len(load_split(task, "val")) == 200
        assert len(load_split(task, "test")) == 100
