# Run report: smoke_tiny_t5_random_smoke-eval

## Run info

| field | value |
| --- | --- |
| name | smoke_tiny_t5_random_smoke-eval |
| system | finetune_seq2seq |
| config_path | configs/smoke/tiny_t5.yaml |
| overrides | ['model.adapter_dir=results/runs/20260712T171338Z_smoke_tiny_t5_random_smoke/adapter'] |
| data | data/fixtures/mini.jsonl |
| split | all |
| n_rows | 200 |
| seed | 13 |
| timestamp | 20260712T171533Z |
| git_commit | c6351ec |

## faithfulness

| metric | value |
| --- | --- |
| n | 200 |
| hallucination_rate | 0.0000 |
| omission_rate | 1.0000 |
| faithfulness_score | 1.0000 |
| recall.wicket_mentioned | 0.0000 |
| recall.boundary_mentioned | 0.0000 |
| ner | unavailable (spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.) |
| examples.violations | [] |
| examples.omissions | [{'text': 'wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate wate', 'omissions': ['four unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['wicket unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['four unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['wicket unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['wicket unmentioned']}, {'text': '! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !', 'omissions': ['four unmentioned']}, {'text': '109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109', 'omissions': ['six unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['six unmentioned']}, {'text': 'Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings Jennings', 'omissions': ['six unmentioned']}, {'text': 'gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap gap', 'omissions': ['four unmentioned']}] |
| reference_baseline.hallucination_rate | 0.0000 |
| reference_baseline.omission_rate | 0.2025 |

## excitement

| metric | value |
| --- | --- |
| reference_validation.spearman_rho | 0.8064 |
| reference_validation.n | 200 |
| reference_validation.mode | heuristic |
| reference_validation.fallback_reason | could not load excitement model 'j-hartmann/emotion-english-distilroberta-base': We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'. |
| reference_validation.per_phase.powerplay | 0.6733 |
| reference_validation.per_phase.middle | 0.8035 |
| reference_validation.per_phase.death | 0.8561 |
| reference_validation.passed | True |
| calibration.spearman_rho | 0.0472 |
| calibration.n | 200 |
| calibration.mode | heuristic |
| calibration.fallback_reason | could not load excitement model 'j-hartmann/emotion-english-distilroberta-base': We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'. |
| calibration.per_phase.powerplay | - |
| calibration.per_phase.middle | -0.0291 |
| calibration.per_phase.death | 0.1547 |
| calibration.passed | True |

## surface

| metric | value |
| --- | --- |
| bleu | 0.0087 |
| rouge_l | 0.0009 |
| bert_score | unavailable (bert-score could not run (roberta-large weights or torch backend missing): We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.) |

## diversity

| metric | value |
| --- | --- |
| distinct_1 | 0.0063 |
| distinct_2 | 0.0065 |
| self_bleu | 0.4300 |
| repetition_4gram_rate | 0.5691 |
| n_generations | 200 |

## latency

| metric | value |
| --- | --- |
| n_timed | 200 |
| p50_ms | 80.6435 |
| p95_ms | 90.8266 |
| mean_ms | 81.7119 |
| tokens_per_sec | 391.6200 |
| hardware.platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| hardware.cpu_count | 4 |
| hardware.torch_device | cpu |

## Metrics not computed

- `faithfulness.ner`: spacy model 'en_core_web_sm' not downloadable/installed: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
- `surface.bert_score`: bert-score could not run (roberta-large weights or torch backend missing): We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
