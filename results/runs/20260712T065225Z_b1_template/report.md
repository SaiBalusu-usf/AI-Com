# Run report: b1_template

## Run info

| field | value |
| --- | --- |
| name | b1_template |
| system | template |
| config_path | configs/baseline_template.yaml |
| overrides | [] |
| data | data/processed/dataset.jsonl |
| split | test |
| n_rows | 689 |
| seed | 13 |
| timestamp | 20260712T065225Z |
| git_commit | 6dd63cc |

## faithfulness

| metric | value |
| --- | --- |
| n | 689 |
| hallucination_rate | 0.0000 |
| omission_rate | 0.0000 |
| faithfulness_score | 1.0000 |
| per_slot.attribution.claims | 129 |
| per_slot.attribution.false_claims | 0 |
| per_slot.attribution.precision | 1.0000 |
| per_slot.boundary.claims | 148 |
| per_slot.boundary.false_claims | 0 |
| per_slot.boundary.precision | 1.0000 |
| per_slot.milestone.claims | 10 |
| per_slot.milestone.false_claims | 0 |
| per_slot.milestone.precision | 1.0000 |
| per_slot.runs.claims | 421 |
| per_slot.runs.false_claims | 0 |
| per_slot.runs.precision | 1.0000 |
| per_slot.score.claims | 148 |
| per_slot.score.false_claims | 0 |
| per_slot.score.precision | 1.0000 |
| per_slot.wicket.claims | 49 |
| per_slot.wicket.false_claims | 0 |
| per_slot.wicket.precision | 1.0000 |
| recall.wicket_mentioned | 1.0000 |
| recall.boundary_mentioned | 1.0000 |
| ner | unavailable (spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.) |
| examples.violations | [] |
| examples.omissions | [] |
| reference_baseline.hallucination_rate | 0.0000 |
| reference_baseline.omission_rate | 0.1168 |

## excitement

| metric | value |
| --- | --- |
| reference_validation.spearman_rho | 0.7071 |
| reference_validation.n | 689 |
| reference_validation.mode | heuristic |
| reference_validation.fallback_reason | excitement model half needs transformers, which is not installed: No module named 'transformers' |
| reference_validation.per_phase.powerplay | 0.6010 |
| reference_validation.per_phase.middle | 0.6424 |
| reference_validation.per_phase.death | 0.8092 |
| reference_validation.passed | True |
| calibration.spearman_rho | 0.5487 |
| calibration.n | 689 |
| calibration.mode | heuristic |
| calibration.fallback_reason | excitement model half needs transformers, which is not installed: No module named 'transformers' |
| calibration.per_phase.powerplay | 0.4659 |
| calibration.per_phase.middle | 0.5042 |
| calibration.per_phase.death | 0.6948 |
| calibration.passed | True |

## surface

| metric | value |
| --- | --- |
| bleu | 5.7963 |
| rouge_l | 0.2992 |
| bert_score | unavailable (bert-score/torch not installed or not loadable: No module named 'bert_score') |

## diversity

| metric | value |
| --- | --- |
| distinct_1 | 0.2224 |
| distinct_2 | 0.4195 |
| self_bleu | 0.1055 |
| repetition_4gram_rate | 0.0337 |
| n_generations | 689 |

## latency

| metric | value |
| --- | --- |
| n_timed | 689 |
| p50_ms | 0.0558 |
| p95_ms | 0.0925 |
| mean_ms | 0.0601 |
| tokens_per_sec | 110588.4530 |
| hardware.platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| hardware.cpu_count | 4 |
| hardware.torch_device | not_installed |

## Metrics not computed

- `faithfulness.ner`: spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
- `surface.bert_score`: bert-score/torch not installed or not loadable: No module named 'bert_score'
