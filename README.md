# Faithful, Game-State-Aware Cricket Commentary Generation

A data-to-text study for CAP 6640 (graduate NLP). Given a structured
ball-by-ball cricket event record, generate one-to-three sentences of English
commentary — and measure, with a slot-level fact checker, how faithful each
generation approach stays to the structured input.

> Status: under construction phase by phase; see PLAN.md for the phase
> checklist and EXPERIMENTS.md for the run log. This README is finalized in
> Phase 7 with the results table.

## Quick start

```bash
make venv && . .venv/bin/activate   # optional but recommended
make setup        # core stack (data + eval + template baseline), CPU-only OK
make test         # unit tests
make mini         # golden path: fixture -> baseline -> full eval -> report, <5 min
```

The neural systems (few-shot LLM, LoRA fine-tunes, BERTScore) additionally
need `make setup-full` (and `make setup-cuda` on CUDA machines) plus network
access to huggingface.co. On CPU-only machines install torch from the CPU
wheel index first — see the note at the top of `requirements-ml.txt`.

## Research questions

- **RQ1 Faithfulness:** template vs. few-shot vs. fine-tuned generation,
  measured by slot-level fact-checking against the structured input.
- **RQ2 Game-state conditioning:** do match-tension features improve
  excitement calibration?
- **RQ3 Efficiency:** can a fine-tuned ≤1.5B model match a large prompted
  model on faithfulness at <1 s/ball?

## Repository map

See §3 of the build prompt: `src/cricket_commentary/{data,models,eval,utils}`,
config-driven experiments in `configs/`, run artifacts in `results/runs/`,
one-command entry points in the `Makefile`.
