# Golden-path entry points. `make mini` is the grader's one-command check.
# POSIX: an existing .venv is used automatically, else system python3.
# Windows (PowerShell/cmd + GNU Make): activate the venv first, then `python`
# resolves inside it. Core targets (setup/test/fixture/mini/tables/figures/
# repro/train-*) are Windows-safe; `ablations` and `clean` use a POSIX shell —
# run those from Git Bash or WSL.
ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
endif
PIP    := $(PYTHON) -m pip

.PHONY: venv setup setup-full setup-cuda test data download fixture mini \
        baseline train-t5 train-t5-smoke train-qwen train-qwen-smoke \
        tiny-models train-t5-smoke-tiny train-qwen-smoke-tiny \
        annotate annotate-score examples ablations eval tables figures repro clean

venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup-full: setup
	$(PIP) install -r requirements-ml.txt

setup-cuda: setup-full
	$(PIP) install -r requirements-cuda.txt

test:
	$(PYTHON) -m pytest -q

# Build the processed dataset from the source selected in configs/data.yaml.
# With source.kind=cricsheet this needs `make download` (network) first.
data:
	$(PYTHON) scripts/build_dataset.py --config configs/data.yaml

download:
	$(PYTHON) scripts/download_data.py --config configs/data.yaml

# Regenerate the committed 200-row golden-path fixture (seeded; deterministic).
fixture:
	$(PYTHON) scripts/build_dataset.py --config configs/data.yaml --fixture-only

# Golden path: fixture -> template baseline -> full eval harness -> report.
# Must finish in under 5 minutes on a laptop CPU.
mini:
	$(PYTHON) scripts/run_experiment.py --config configs/baseline_template.yaml \
	    --data data/fixtures/mini.jsonl --tag mini

baseline:
	$(PYTHON) scripts/run_experiment.py --config configs/baseline_template.yaml
	$(PYTHON) scripts/run_experiment.py --config configs/baseline_fewshot.yaml

# Training prints its plan; full runs require --yes (§9.5). Smoke = 50 steps
# on the fixture (the phase-4 gate; needs make setup-full).
train-t5:
	$(PYTHON) scripts/train_model.py --config configs/finetune_t5.yaml --plan-only

train-t5-smoke:
	$(PYTHON) scripts/train_model.py --config configs/finetune_t5.yaml --smoke

train-qwen:
	$(PYTHON) scripts/train_model.py --config configs/finetune_qwen.yaml --plan-only

train-qwen-smoke:
	$(PYTHON) scripts/train_model.py --config configs/finetune_qwen.yaml --smoke

# Phase-4 machinery smoke WITHOUT huggingface access: from-scratch tiny
# random models (outputs are noise; runs are named smoke_* on purpose).
tiny-models:
	$(PYTHON) scripts/make_tiny_model.py --out data/processed/tiny_random

train-t5-smoke-tiny: tiny-models
	$(PYTHON) scripts/train_model.py --config configs/smoke/tiny_t5.yaml --smoke

train-qwen-smoke-tiny: tiny-models
	$(PYTHON) scripts/train_model.py --config configs/smoke/tiny_causal.yaml --smoke

# Print the plan for every ablation cell (run individual cells explicitly).
ablations:
	@for f in configs/ablations/*.yaml; do \
		$(PYTHON) scripts/run_experiment.py --config $$f --plan-only || true; \
	done

# Re-evaluate an existing run directory: make eval RUN=results/runs/<dir>
eval:
	$(PYTHON) scripts/run_experiment.py --eval-only --run $(RUN)

# Checker-accuracy protocol: draw a sheet, hand-label it, score agreement.
#   make annotate RUN=results/runs/<dir> OUT=results/annotation/m2.csv
#   make annotate-score FILE=results/annotation/m2.csv
annotate:
	$(PYTHON) scripts/audit_factcheck.py --sample 100 --run $(RUN) --out $(OUT)

annotate-score:
	$(PYTHON) scripts/audit_factcheck.py --score $(FILE)

tables:
	$(PYTHON) scripts/make_tables.py --runs results/runs --out results/tables

figures:
	$(PYTHON) scripts/make_figures.py --runs results/runs --out results/figures

# Fresh-clone reproducibility check at fixture scale (Definition of Done #1).
repro: test fixture mini tables figures
	@echo "repro OK: fixture pipeline, tables and figures rebuilt from scratch"

clean:
	rm -rf data/processed/* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

# Qualitative side-by-side table for the paper (inputs, outputs, checker verdicts).
examples:
	$(PYTHON) scripts/make_examples.py --runs results/runs --out results/tables
