"""Loading a chat LLM, building the paper's prompts, and reading last-token activations."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ANSWER_PREFIX, QUESTION_SUFFIX

# Prompt formats used in the paper's experiments (single user turn, no system prompt).
TEMPLATES = {
    "llama3": ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n{}", "<|eot_id|>"),
    "qwen2": ("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}", "<|im_end|>"),
}


def model_family(name):
    n = name.lower()
    if "llama-3" in n or "llama3" in n:
        return "llama3"
    if "qwen2" in n:
        return "qwen2"
    raise ValueError(f"Unsupported model {name!r}: add its prompt template to innsteer/llm.py::TEMPLATES")


class SteerableLM:
    """Thin wrapper around a Hugging Face causal LM.

    dtype: "auto" keeps the checkpoint dtype. In the paper's runs LLaMA models were loaded in
    float32 and Qwen models with "auto"; `default_dtype` reproduces that choice.
    """

    def __init__(self, name, device=None, dtype=None):
        self.name = name
        self.family = model_family(name)
        self.template, end_token = TEMPLATES[self.family]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or ("float32" if self.family == "llama3" else "auto")
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.terminators = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids(end_token)]
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self.yes_id = self._single_token(" Yes")
        self.no_id = self._single_token(" No")

    def _single_token(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token for {self.name}")
        return ids[0]

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def num_layers(self):
        return len(self.layers)

    # ---------------------------------------------------------------- prompts
    def prompt(self, question, answer=""):
        return self.template.format(question + QUESTION_SUFFIX, ANSWER_PREFIX + answer)

    def encode(self, texts):
        return self.tok(texts, padding="longest", return_tensors="pt").to(self.device)

    # ------------------------------------------------------------ activations
    @torch.no_grad()
    def contrastive_activations(self, rows, layers):
        """Final-token activations for each (behavior-matching, non-matching) answer pair.

        Returns {layer: (acts_pos, acts_neg)}, each of shape (N, d) in float32 on CPU.
        """
        store = {l: ([], []) for l in layers}
        cache = {}
        hooks = [self.layers[l].register_forward_hook(_capture(cache, l)) for l in layers]
        try:
            for r in rows:
                other = " No" if r["matching"] == " Yes" else " Yes"
                self.model(**self.encode([self.prompt(r["question"], r["matching"]),
                                          self.prompt(r["question"], other)]))
                for l in layers:
                    store[l][0].append(cache[l][0])
                    store[l][1].append(cache[l][1])
        finally:
            for h in hooks:
                h.remove()
        return {l: (torch.stack(p), torch.stack(n)) for l, (p, n) in store.items()}

    @torch.no_grad()
    def generate(self, enc, max_new_tokens=200):
        return self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self.tok.eos_token_id, eos_token_id=self.terminators)


def _capture(cache, layer):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        cache[layer] = h[:, -1, :].detach().float().cpu()
    return hook
