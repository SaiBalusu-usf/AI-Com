# Model cards

## B1 — rule-based template generator (`models/template.py`)

- **Type:** deterministic slot-filling over the structured ball record; ~5
  surface variants per outcome class, per-ball seeded variant choice.
- **Game-state conditioning:** `tension × salience > excitement_gate` selects
  an excited register (config `generator.use_game_state`).
- **Guarantees:** zero hallucination by construction (verified by the fact
  checker over the full test split); latency ≈ 0.06 ms/ball on 4-CPU container.
- **Intended use:** faithfulness ceiling / diversity floor baseline.

## B2 — few-shot prompted LLM (`models/fewshot.py`)

- **Backends:** `hf_local` (default: Qwen/Qwen2.5-1.5B-Instruct; 7B config
  commented for CUDA machines), `anthropic`/`openai` behind `--confirm-spend`.
- **Prompting:** system prompt + 8 seeded, outcome-stratified in-context
  examples from the train split; template lives in
  `configs/baseline_fewshot.yaml`, not code.
- **Status:** NOT RUN — blocked in the build environment (no ML stack /
  huggingface egress; PLAN.md D2). Prompt assembly and example selection are
  unit-tested.

## M1 — flan-t5-base + LoRA (`models/finetune.py`, `configs/finetune_t5.yaml`)

- **Base:** google/flan-t5-base (~250M). LoRA r=16, α=32, dropout 0.05 on
  q/v; lr 2e-4, ≤3 epochs, early stopping on val loss; max_src 512 /
  max_tgt 96. Input: `describe this ball: <linearized_input>`.
- **Status:** NOT TRAINED — pending D2/D3. Smoke gate:
  `make train-t5-smoke` (50 steps on the committed fixture).

## M2 — Qwen2.5-1.5B-Instruct + QLoRA via TRL (`configs/finetune_qwen.yaml`)

- **Base:** Qwen/Qwen2.5-1.5B-Instruct; 4-bit NF4 only when CUDA+bitsandbytes
  are present (`resolve_device_plan` documents any degradation); LoRA r=16,
  α=32 on q/k/v/o projections; lr 2e-4; fixed system prompt "You are a live
  cricket commentator. Describe this ball."
- **CPU substitute (A3 cell):** Qwen2.5-0.5B-Instruct, full precision.
- **Status:** NOT TRAINED — pending D2/D3. Smoke gate: `make train-qwen-smoke`.

## Adapter artifacts

Trained adapters land in `results/runs/<stamp>_<name>_train/adapter/`
(gitignored; weights are not committed). Evaluation re-attaches them with
`python scripts/run_experiment.py --config <same config> --set
model.adapter_dir=<run>/adapter`, so every model flows through the identical
harness.
