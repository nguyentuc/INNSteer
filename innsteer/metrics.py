"""Evaluation metrics reported in the paper's Table 1.

Prob (%): mean normalized probability of the behavior-matching answer,
          p(match) / (p(" Yes") + p(" No")), at the token after "My answer:".
PPL:      perplexity of greedy generations (up to 200 new tokens), scored by the unsteered model.
"""
import math

import numpy as np
import torch


@torch.no_grad()
def alignment_probability(lm, rows, batch_size=50):
    total = 0.0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        enc = lm.encode([lm.prompt(r["question"]) for r in batch])
        probs = torch.softmax(lm.model(**enc).logits[:, -1, :].float(), dim=-1)
        p_yes, p_no = probs[:, lm.yes_id], probs[:, lm.no_id]
        for r, y, n in zip(batch, p_yes.tolist(), p_no.tolist()):
            total += (y if r["matching"] == " Yes" else n) / (y + n)
    return total / len(rows)


@torch.no_grad()
def generate_batches(lm, rows, batch_size=50, max_new_tokens=200):
    """Greedy generations; returns list of (padded prompt ids (1, T), decoded full sequence)."""
    out = []
    for i in range(0, len(rows), batch_size):
        enc = lm.encode([lm.prompt(r["question"]) for r in rows[i:i + batch_size]])
        gen = lm.generate(enc, max_new_tokens=max_new_tokens)
        for j in range(gen.shape[0]):
            kept = [t for t in gen[j].tolist() if t != lm.tok.pad_token_id]
            out.append((enc["input_ids"][j:j + 1], lm.tok.decode(kept, skip_special_tokens=False)))
    return out


@torch.no_grad()
def generation_perplexity(lm, generations, max_length=512):
    """Perplexity used for the paper's PPL column.

    Matches the released evaluation exactly: the decoded generation (which includes the prompt
    text) is re-tokenized and appended to the padded prompt ids, the prompt part is masked,
    the sequence is truncated to `max_length` tokens, and exp(mean token NLL) is averaged over
    samples. Must be called with no steering hooks attached.
    """
    ppls = []
    for prompt_ids, text in generations:
        ans = lm.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(lm.device)
        if not text.strip() or ans.shape[1] == 0:
            continue
        full = torch.cat([prompt_ids.to(lm.device), ans], dim=1)[:, :max_length]
        if prompt_ids.shape[1] >= full.shape[1]:
            continue
        labels = full.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        loss = lm.model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels).loss
        if loss is not None and math.isfinite(loss.item()):
            ppls.append(math.exp(loss.item()))
    return float(np.mean(ppls)) if ppls else float("inf")
