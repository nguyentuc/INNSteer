"""Inference-time steering hooks.

INNSteer (paper Eq. for the forward hook at layer l):
    h' = sigma * phi^-1( phi((h - mu) / sigma) + alpha * v_phi ) + mu
applied to every position of the layer output. Mean-difference (CAA-style) steering adds
alpha * v to the last position only, as in the paper's MD baseline.
"""
import contextlib

import torch


def latent_steering_vector(inn, acts_pos_n, acts_neg_n, device):
    """v_phi = mean_i [ phi(h+_i) - phi(h-_i) ] on z-scored training activations."""
    inn.eval()
    with torch.no_grad():
        return (inn.phi(acts_pos_n.to(device)) - inn.phi(acts_neg_n.to(device))).mean(0)


def innsteer_transform(inn, v_phi, mu, sigma, alpha):
    """Return f(h) implementing the INNSteer edit for a (..., d) tensor."""
    def f(h):
        shape, dtype = h.shape, h.dtype
        x = (h.reshape(-1, shape[-1]).float() - mu) / sigma
        x = inn.phi_inv(inn.phi(x) + alpha * v_phi.unsqueeze(0)) * sigma + mu
        return x.reshape(shape).to(dtype)
    return f


@contextlib.contextmanager
def innsteer_hooks(lm, steer):
    """steer: {layer: dict(inn=..., v_phi=..., mu=..., sigma=...)}; alpha is read from each dict."""
    handles = []
    try:
        for layer, s in steer.items():
            inn = s["inn"].to(lm.device).eval()
            f = innsteer_transform(inn, s["v_phi"].to(lm.device).float(), s["mu"].to(lm.device),
                                   s["sigma"].to(lm.device), s.get("alpha", 1.0))
            handles.append(lm.layers[layer].register_forward_hook(_replace_output(f)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def add_vector_hooks(lm, vectors):
    """vectors: {layer: alpha * v}; adds the vector at the last position (MD / CAA baseline)."""
    def make(v):
        def f(h):
            h = h.clone()
            h[:, -1, :] += v.to(h.dtype)
            return h
        return f
    handles = []
    try:
        for layer, v in vectors.items():
            handles.append(lm.layers[layer].register_forward_hook(_replace_output(make(v.to(lm.device)))))
        yield
    finally:
        for h in handles:
            h.remove()


def _replace_output(f):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (f(output[0]),) + tuple(output[1:])
        return f(output)
    return hook
