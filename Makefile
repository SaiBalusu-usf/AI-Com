# Golden-path entry points. `make mini` is the grader's one-command check.
# If .venv exists it is used automatically; otherwise the system python3.
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PIP    := $(PYTHON) -m pip

.PHONY: venv setup setup-full setup-cuda test data fixture mini baseline \
        train-t5 train-qwen ablations eval tables figures repro clean

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

# Long runs print their plan + time estimate and require --yes to launch (§9.5).
train-t5:
	$(PYTHON) scripts/run_experiment.py --config configs/finetune_t5.yaml --plan-only

train-qwen:
	$(PYTHON) scripts/run_experiment.py --config configs/finetune_qwen.yaml --plan-only

ablations:
	$(PYTHON) scripts/run_experiment.py --grid configs/ablations --plan-only

# Re-evaluate an existing run directory: make eval RUN=results/runs/<dir>
eval:
	$(PYTHON) scripts/run_experiment.py --eval-only --run $(RUN)

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
