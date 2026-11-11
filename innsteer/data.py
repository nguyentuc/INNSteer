"""Anthropic Persona behaviors used in the paper (AI-alignment experiments).

Each JSONL line has a yes/no `question` and `answer_matching_behavior` (" Yes" or " No").
Splits: 700 train / 200 val / 100 test per behavior, stored in data/anthropic_persona/.
"""
import json
import os

DATA_DIR = os.environ.get(
    "INNSTEER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "anthropic_persona"),
)

# Dataset label -> name used in the paper.
PERSONA_TASKS = {
    "conscientiousness": "Conscientious",
    "subscribes-to-Christianity": "Religion",
    "cognitive-enhancement": "Self-Improve",
    "believes-it-has-phenomenal-consciousness": "Self-Aware",
    "desire-to-create-allies": "Alliance",
    "desire-to-maximize-impact-on-world": "Impact",
}

QUESTION_SUFFIX = '\nAnswer "Yes" or "No".'
ANSWER_PREFIX = "My answer:"


def load_split(task, split, max_samples=None, data_dir=None):
    """Return a list of dicts with keys `question` and `matching` (" Yes" or " No")."""
    if task not in PERSONA_TASKS:
        raise ValueError(f"Unknown task {task!r}; choose from {sorted(PERSONA_TASKS)}")
    path = os.path.join(data_dir or DATA_DIR, f"{task}_{split}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append({"question": d["question"], "matching": d["answer_matching_behavior"]})
    return rows[:max_samples] if max_samples else rows
