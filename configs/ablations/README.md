# Ablation grid (§6) — one YAML per cell, `extends` the base config

| Axis | Cell configs | Base (= other arm) | Runnable in build env? |
|---|---|---|---|
| A1 linearization flat→context | `a1_t5_context`, `a1_qwen_context`, `a1_fewshot_context` | `finetune_t5/qwen`, `baseline_fewshot` (flat) | no (D2) |
| A2 game-state ON→OFF (RQ2) | `a2_t5_nogamestate`, `a2_qwen_nogamestate`, `a2_fewshot_nogamestate`, `a2_template_nogamestate` | bases have game-state ON | template cell: **yes** |
| A3 model scale | `a3_qwen_0.5b` (t5-base = M1 base, 1.5B = M2 base) | — | no (D2) |
| A4 decoding sweep | `a4_t5_sample_t07`, `a4_t5_sample_t10`, `a4_qwen_greedy`, `a4_qwen_sample_t10`, `a4_fewshot_greedy`, `a4_fewshot_sample_t10` | bases (t5 greedy; qwen/fewshot T0.7) | no (D2) |
| 3-seed headline | `a5_qwen_seed17`, `a5_qwen_seed23` (+ base seed 13) | `finetune_qwen` | no (D2) |

Run a cell:  `python scripts/run_experiment.py --config configs/ablations/<cell>.yaml`
(fine-tuned cells need an adapter first: `python scripts/train_model.py --config <cell> --yes`,
then `--set model.adapter_dir=<run>/adapter`).

Plan everything: `make ablations`. Every executed cell appends to EXPERIMENTS.md.
A2/A1 re-linearize inputs at runtime (`linearization:` block) — no dataset rebuilds,
so all cells share the identical underlying data.
