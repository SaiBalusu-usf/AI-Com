# ML dependency migration

The neural stack now uses Transformers 5, Torch 2.14, TRL 1.16 and PEFT 0.21.
`make setup` installs the audited core lock; `make setup-full` installs the
complete, hash-pinned lock. Regenerate both with `uv pip compile --universal
--generate-hashes` against their input requirements and review the dependency
diff plus a fresh vulnerability audit before committing them.

Transformers 5 chat templates return a mapping. Both causal adapter inference
and few-shot inference now pass its input IDs and attention mask to generation.
Regression tests train real tiny T5/Llama models from scratch, save and reload
LoRA adapters, and execute generation. They require no model downloads or keys.
Random-model outputs validate machinery, not commentary quality.

Local validation on Python 3.12: 177 existing tests passed (7 conditional tests
skipped), plus 3 new real-model compatibility tests. Both 50-step smoke training
paths completed. T5 and causal adapters were evaluated on all 200 fixture rows.
The resolved dependency audit reported no known vulnerabilities on 2026-10-26.
GPU/4-bit behavior and pretrained-model quality still require their documented
hardware and approved model artifacts; they were not claimed as validated here.

CI retains the core reproducibility workflow and adds Windows CPU tests using
the PyPI wheel and the full lock. No pretrained weights are fetched by that job.
