"""B2 — few-shot prompted LLM (§6).

FRONTIER_API=none for this project, so the default backend is a local HF
instruct model (``hf_local``); ``anthropic``/``openai`` backends exist behind
an explicit spend confirmation (§9 rule 5 — never called silently). The
prompt template lives in configs/baseline_fewshot.yaml, not in code.

The 8 in-context examples are drawn once per run from the train split,
stratified across outcome classes (wicket, six, four, dot, runs, extras) with
a seeded RNG — deterministic given (seed, dataset).

Environment note (PLAN.md): huggingface.co is egress-blocked in the build
container and the ML stack is not installed, so instantiating hf_local there
raises GeneratorUnavailable with the documented reason. The class still
builds its prompt without a model, which is what the unit tests cover.
"""

from __future__ import annotations

import os

from ..utils.seeding import rng_for


class GeneratorUnavailable(Exception):
    """The generation backend cannot run in this environment (missing
    dependency, missing weights, or unapproved spend)."""


def _outcome_class(record: dict) -> str:
    if record["wicket"]:
        return "wicket"
    if record["runs_batter"] == 6:
        return "six"
    if record["runs_batter"] == 4:
        return "four"
    if record["extras_type"] != "none":
        return "extra"
    if record["runs_total"] == 0:
        return "dot"
    return "runs"


def select_examples(train_rows: list[dict], n_shots: int, seed: int) -> list[dict]:
    """Stratified, seeded draw covering outcome classes before topping up."""
    if not train_rows:
        raise ValueError("few-shot example selection needs train rows")
    rng = rng_for(seed, "fewshot-examples")
    buckets: dict[str, list[int]] = {}
    for i, row in enumerate(train_rows):
        buckets.setdefault(_outcome_class(row["record"]), []).append(i)
    chosen: list[int] = []
    for cls in sorted(buckets):
        if len(chosen) < n_shots:
            chosen.append(int(rng.choice(buckets[cls])))
    remaining = [i for i in range(len(train_rows)) if i not in set(chosen)]
    while len(chosen) < n_shots and remaining:
        pick = int(rng.choice(remaining))
        chosen.append(pick)
        remaining.remove(pick)
    return [train_rows[i] for i in chosen[:n_shots]]


class FewShotGenerator:
    def __init__(self, cfg: dict, seed: int, train_rows: list[dict],
                 decoding: dict | None = None, allow_paid_api: bool = False):
        self.cfg = cfg
        self.seed = seed
        self.decoding = decoding or {"strategy": "greedy"}
        self.examples = select_examples(train_rows, int(cfg["n_shots"]), seed)
        self.example_block = "\n".join(
            f"Input: {row['linearized_input']}\nCommentary: {row['target_commentary']}\n"
            for row in self.examples
        )
        backend = cfg["backend"]
        if backend == "hf_local":
            self._pipe = self._load_hf(cfg)
        elif backend in ("anthropic", "openai"):
            if not allow_paid_api:
                raise GeneratorUnavailable(
                    f"backend={backend} is a paid API; re-run with "
                    "--confirm-spend after human approval (§9 rule 5)"
                )
            self._pipe = self._load_api(backend, cfg)
        else:
            raise ValueError(f"unknown fewshot backend: {backend}")

    # ------------------------------------------------------------- backends

    def _load_hf(self, cfg: dict):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise GeneratorUnavailable(
                f"hf_local backend needs the ML stack (make setup-full): {err}"
            ) from err
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
            model_kwargs = {}
            if cfg.get("load_in_4bit"):
                model_kwargs["load_in_4bit"] = True
            model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)
        except OSError as err:
            raise GeneratorUnavailable(
                f"could not load {cfg['model_name']!r} (huggingface unreachable "
                f"or weights absent): {err}"
            ) from err
        return ("hf", tokenizer, model)

    def _load_api(self, backend: str, cfg: dict):
        key_var = "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(key_var):
            raise GeneratorUnavailable(f"{backend} backend needs {key_var} set")
        try:
            client = (
                __import__("anthropic").Anthropic()
                if backend == "anthropic"
                else __import__("openai").OpenAI()
            )
        except ImportError as err:
            raise GeneratorUnavailable(f"{backend} sdk not installed: {err}") from err
        return (backend, client, None)

    # ------------------------------------------------------------ inference

    def build_prompt(self, row: dict) -> str:
        return self.cfg["prompt_template"].format(
            examples=self.example_block, input=row["linearized_input"]
        )

    def generate(self, row: dict) -> str:
        kind = self._pipe[0]
        prompt = self.build_prompt(row)
        if kind == "hf":
            return self._generate_hf(prompt)
        return self._generate_api(kind, prompt)

    def _generate_hf(self, prompt: str) -> str:
        import torch

        _, tokenizer, model = self._pipe
        messages = [
            {"role": "system", "content": self.cfg["system_prompt"]},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        sample = self.decoding.get("strategy") == "sample"
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=int(self.cfg.get("max_new_tokens", 60)),
                do_sample=sample,
                temperature=float(self.decoding.get("temperature", 1.0)) if sample else None,
                top_p=float(self.decoding.get("top_p", 1.0)) if sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0]

    def _generate_api(self, backend: str, prompt: str) -> str:
        _, client, _ = self._pipe
        if backend == "anthropic":
            msg = client.messages.create(
                model=self.cfg.get("api_model", "claude-haiku-4-5-20251001"),
                max_tokens=int(self.cfg.get("max_new_tokens", 60)),
                system=self.cfg["system_prompt"],
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip().split("\n")[0]
        rsp = client.chat.completions.create(
            model=self.cfg.get("api_model", "gpt-4o-mini"),
            max_tokens=int(self.cfg.get("max_new_tokens", 60)),
            messages=[
                {"role": "system", "content": self.cfg["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
        )
        return rsp.choices[0].message.content.strip().split("\n")[0]


def build(cfg: dict, seed: int, train_rows: list[dict] | None = None,
          decoding: dict | None = None, allow_paid_api: bool = False) -> FewShotGenerator:
    return FewShotGenerator(cfg, seed, train_rows or [], decoding, allow_paid_api)
