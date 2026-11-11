#!/usr/bin/env bash
# Main AI-alignment experiments (paper Table 1 / Figure 1): 5 LLMs x 6 behaviors.
# Each call evaluates Base, mean-difference steering (MD / CAA) and INNSteer on one model.
# Requires a GPU; LLaMA checkpoints are gated on Hugging Face (accept the license and `huggingface-cli login`).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Meta-Llama-3-8B-Instruct
  Qwen/Qwen2.5-3B-Instruct
  Qwen/Qwen2.5-7B-Instruct
  Qwen/Qwen2.5-32B-Instruct
)

for model in "${MODELS[@]}"; do
  name=$(basename "$model")
  python scripts/run_persona.py --model "$model" --out "results/${name}.json" "$@"
done
