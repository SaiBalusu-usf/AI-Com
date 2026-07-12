# Run report: b1_template_mini

## Run info

| field | value |
| --- | --- |
| name | b1_template_mini |
| system | template |
| config_path | configs/baseline_template.yaml |
| overrides | [] |
| data | data/fixtures/mini.jsonl |
| split | all |
| n_rows | 200 |
| seed | 13 |
| timestamp | 20260712T062340Z |
| git_commit | c5b9bc7 |

## faithfulness

| metric | value |
| --- | --- |
| n | 200 |
| hallucination_rate | 0.0000 |
| omission_rate | 0.0000 |
| faithfulness_score | 1.0000 |
| per_slot.attribution.claims | 45 |
| per_slot.attribution.false_claims | 0 |
| per_slot.attribution.precision | 1.0000 |
| per_slot.boundary.claims | 47 |
| per_slot.boundary.false_claims | 0 |
| per_slot.boundary.precision | 1.0000 |
| per_slot.milestone.claims | 3 |
| per_slot.milestone.false_claims | 0 |
| per_slot.milestone.precision | 1.0000 |
| per_slot.runs.claims | 103 |
| per_slot.runs.false_claims | 0 |
| per_slot.runs.precision | 1.0000 |
| per_slot.score.claims | 32 |
| per_slot.score.false_claims | 0 |
| per_slot.score.precision | 1.0000 |
| per_slot.wicket.claims | 22 |
| per_slot.wicket.false_claims | 0 |
| per_slot.wicket.precision | 1.0000 |
| recall.wicket_mentioned | 1.0000 |
| recall.boundary_mentioned | 1.0000 |
| ner | unavailable (spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.) |
| examples.violations | [] |
| examples.omissions | [] |
| reference_baseline.hallucination_rate | 0.0000 |
| reference_baseline.omission_rate | 0.1594 |

## excitement

| metric | value |
| --- | --- |
| reference_validation.spearman_rho | 0.7841 |
| reference_validation.n | 200 |
| reference_validation.mode | heuristic |
| reference_validation.fallback_reason | excitement model half needs transformers, which is not installed: No module named 'transformers' |
| reference_validation.per_phase.powerplay | 0.6506 |
| reference_validation.per_phase.middle | 0.7696 |
| reference_validation.per_phase.death | 0.8641 |
| reference_validation.passed | True |
| calibration.spearman_rho | 0.6501 |
| calibration.n | 200 |
| calibration.mode | heuristic |
| calibration.fallback_reason | excitement model half needs transformers, which is not installed: No module named 'transformers' |
| calibration.per_phase.powerplay | 0.4826 |
| calibration.per_phase.middle | 0.6989 |
| calibration.per_phase.death | 0.7130 |
| calibration.passed | True |

## surface

| metric | value |
| --- | --- |
| bleu | 6.2990 |
| rouge_l | 0.3140 |
| bert_score | unavailable (bert-score/torch not installed or not loadable: No module named 'bert_score') |

## diversity

| metric | value |
| --- | --- |
| distinct_1 | 0.5742 |
| distinct_2 | 0.7870 |
| self_bleu | 0.0874 |
| repetition_4gram_rate | 0.0231 |
| n_generations | 200 |

## latency

| metric | value |
| --- | --- |
| n_timed | 200 |
| p50_ms | 0.0544 |
| p95_ms | 0.0860 |
| mean_ms | 0.0586 |
| tokens_per_sec | 115463.2024 |
| hardware.platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| hardware.cpu_count | 4 |
| hardware.torch_device | not_installed |

## Metrics not computed

- `faithfulness.ner`: spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
- `surface.bert_score`: bert-score/torch not installed or not loadable: No module named 'bert_score'
