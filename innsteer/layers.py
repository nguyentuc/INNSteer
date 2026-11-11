"""Steering-layer selection used before training the INN.

Following the paper, the steering layer is chosen with the LayerNavigator criterion
(Sun et al., NeurIPS 2025): S = D + C on z-scored activations, where for the unit
mean-difference direction v
  D = v'S_b v / v'(S_b + S_w) v   (Fisher discriminability along v)
  C = mean_i cos(h+_i - h-_i, v)  (directional consistency of the contrastive pairs)
This is an independent implementation of those published formulas.
"""
import torch


def mean_difference(acts_pos, acts_neg):
    return (acts_pos - acts_neg).mean(0)


def layer_scores(acts_pos, acts_neg, v=None):
    """Return dict(d_score, c_score, s_score) for one layer."""
    v = mean_difference(acts_pos, acts_neg) if v is None else v
    v = v.float() / v.norm()
    x = torch.cat([acts_pos, acts_neg]).float()
    x = (x - x.mean(0)) / x.std(0)  # joint z-score
    pos, neg = x[: len(acts_pos)], x[len(acts_pos):]

    diff = pos - neg
    c = ((diff / diff.norm(dim=1, keepdim=True)) @ v).mean().item()

    m = x.mean(0)
    sb = sum(len(g) * (((g.mean(0) - m) @ v) ** 2) for g in (pos, neg))
    sw = sum((((g - g.mean(0)) @ v) ** 2).sum() for g in (pos, neg))
    d = (sb / (sb + sw)).item()
    return {"d_score": d, "c_score": c, "s_score": d + c}


def rank_layers(acts):
    """acts: {layer: (acts_pos, acts_neg)} -> list of (layer, scores) sorted by s_score, best first."""
    scored = [(l, layer_scores(p, n)) for l, (p, n) in acts.items()]
    return sorted(scored, key=lambda t: t[1]["s_score"], reverse=True)
