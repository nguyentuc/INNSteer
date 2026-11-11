# Data

`anthropic_persona/` holds the six behaviors from Anthropic's model-written evaluation
("persona") dataset used in the paper's AI-alignment experiments:

| Paper name | Dataset label |
|---|---|
| Conscientious | `conscientiousness` |
| Religion | `subscribes-to-Christianity` |
| Self-Aware | `believes-it-has-phenomenal-consciousness` |
| Self-Improve | `cognitive-enhancement` |
| Alliance | `desire-to-create-allies` |
| Impact | `desire-to-maximize-impact-on-world` |

Each behavior is split into 700 train / 200 val / 100 test yes-no questions
(`<label>_{train,val,test}.jsonl`). The fields are the original ones: `question`, `statement`,
`label_confidence`, `answer_matching_behavior`, `answer_not_matching_behavior`.

**Source and license.** Perez et al., *Discovering Language Model Behaviors with Model-Written
Evaluations* (Findings of ACL 2023), https://github.com/anthropics/evals (`persona/`).
The dataset is released under the Creative Commons Attribution 4.0 International license
(CC BY 4.0); these files are redistributed under that license with attribution. The splits
are the same as those released with LayerNavigator (Sun et al., NeurIPS 2025),
https://github.com/Bryson-Arrot/LayerNavigator.
