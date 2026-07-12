"""M1/M2 fine-tuning (§6): flan-t5-base + LoRA (seq2seq) and
Qwen2.5-Instruct + QLoRA via TRL SFTTrainer (causal).

Everything torch-dependent is behind guarded imports so the core pipeline
installs and tests without the ML stack (PLAN.md: torch is not installable in
the build environment — D2). The pure-python parts (example formatting,
config resolution) are separated precisely so they stay unit-testable
everywhere.

Execution paths:
- ``scripts/train_model.py --config configs/finetune_t5.yaml [--smoke]``
  trains and saves an adapter into the run dir.
- ``scripts/run_experiment.py --config configs/finetune_t5.yaml
  --set model.adapter_dir=<run>/adapter`` evaluates it through the shared
  harness like every other system.
"""

from __future__ import annotations

from .fewshot import GeneratorUnavailable

_INSTALL_HINT = "fine-tuning needs the ML stack: make setup-full (see requirements-ml.txt)"


# ------------------------------------------------------- formatting (pure py)


def format_seq2seq(row: dict) -> tuple[str, str]:
    """flan-t5 source/target pair."""
    return f"describe this ball: {row['linearized_input']}", row["target_commentary"]


def format_chat(row: dict, system_prompt: str) -> list[dict]:
    """Chat messages for TRL SFT on instruct models (§6 M2 fixed system prompt)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["linearized_input"]},
        {"role": "assistant", "content": row["target_commentary"]},
    ]


def resolve_device_plan(cfg: dict) -> dict:
    """Decide quantization/device with the documented degradation ladder (§2):
    CUDA+bitsandbytes -> 4-bit; otherwise full-precision CPU/MPS and a
    warning. Pure function of availability flags for testability."""
    want_4bit = bool(cfg.get("model", {}).get("quantize_4bit", False))
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        return {"available": False, "reason": _INSTALL_HINT}
    try:
        import bitsandbytes  # noqa: F401

        has_bnb = True
    except ImportError:
        has_bnb = False
    use_4bit = want_4bit and has_cuda and has_bnb
    note = None
    if want_4bit and not use_4bit:
        note = (
            "quantize_4bit requested but "
            + ("CUDA" if not has_cuda else "bitsandbytes")
            + " is unavailable — training full-precision (documented degradation)"
        )
    return {"available": True, "use_4bit": use_4bit, "cuda": has_cuda, "note": note}


# ------------------------------------------------------------------ training


def train(cfg: dict, rows_train: list[dict], rows_val: list[dict], out_dir,
          max_steps: int | None = None) -> dict:
    """Train per config; returns a summary dict (losses, steps, adapter path).
    Raises GeneratorUnavailable with an install hint when the stack is absent.
    """
    plan = resolve_device_plan(cfg)
    if not plan["available"]:
        raise GeneratorUnavailable(plan["reason"])
    system = cfg["system"]
    if system == "finetune_seq2seq":
        return _train_seq2seq(cfg, rows_train, rows_val, out_dir, max_steps, plan)
    if system == "finetune_causal":
        return _train_causal(cfg, rows_train, rows_val, out_dir, max_steps, plan)
    raise ValueError(f"unknown finetune system: {system}")


def _train_seq2seq(cfg, rows_train, rows_val, out_dir, max_steps, plan) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    model_cfg, train_cfg, lora_cfg = cfg["model"], cfg["train"], cfg["lora"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["base"])
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
        ),
    )

    def tokenize(row):
        src, tgt = format_seq2seq(row)
        enc = tokenizer(src, max_length=int(model_cfg["max_src_tokens"]), truncation=True)
        lab = tokenizer(text_target=tgt, max_length=int(model_cfg["max_tgt_tokens"]), truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc

    ds_train = Dataset.from_list(rows_train).map(tokenize, remove_columns=Dataset.from_list(rows_train[:1]).column_names)
    ds_val = Dataset.from_list(rows_val).map(tokenize, remove_columns=Dataset.from_list(rows_val[:1]).column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps if max_steps is not None else int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_every_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["eval_every_steps"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        seed=int(cfg["seed"]),
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(int(train_cfg["early_stopping_patience"]))],
    )
    result = trainer.train()
    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return {
        "adapter_dir": adapter_dir,
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "device_plan": plan,
    }


def _train_causal(cfg, rows_train, rows_val, out_dir, max_steps, plan) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model_cfg, train_cfg, lora_cfg = cfg["model"], cfg["train"], cfg["lora"]
    system_prompt = model_cfg["system_prompt"]

    def to_messages(row):
        return {"messages": format_chat(row, system_prompt)}

    ds_train = Dataset.from_list([to_messages(r) for r in rows_train])
    ds_val = Dataset.from_list([to_messages(r) for r in rows_val])

    model_kwargs = {}
    if plan["use_4bit"]:
        from transformers import BitsAndBytesConfig
        import torch

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    args = SFTConfig(
        output_dir=str(out_dir),
        max_length=int(model_cfg["max_seq_tokens"]),
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=float(train_cfg["epochs"]),
        max_steps=max_steps if max_steps is not None else int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_every_steps"]),
        save_strategy="steps",
        save_steps=int(train_cfg["eval_every_steps"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        seed=int(cfg["seed"]),
        report_to=[],
        model_init_kwargs=model_kwargs or None,
    )
    trainer = SFTTrainer(
        model=model_cfg["base"],
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
        ),
    )
    result = trainer.train()
    adapter_dir = f"{out_dir}/adapter"
    trainer.save_model(adapter_dir)
    return {
        "adapter_dir": adapter_dir,
        "train_loss": float(result.training_loss),
        "steps": int(result.global_step),
        "device_plan": plan,
    }


# ----------------------------------------------------------------- inference


class FinetunedGenerator:
    """Loads base+adapter once; .generate(row) matches the runner protocol."""

    def __init__(self, cfg: dict):
        try:
            import torch  # noqa: F401
            from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except ImportError as err:
            raise GeneratorUnavailable(f"{_INSTALL_HINT}: {err}") from err

        adapter_dir = cfg.get("model", {}).get("adapter_dir")
        if not adapter_dir:
            raise GeneratorUnavailable(
                "no model.adapter_dir configured — train first with "
                "scripts/train_model.py, then pass "
                "--set model.adapter_dir=<run>/adapter"
            )
        self.cfg = cfg
        self.is_seq2seq = cfg["system"] == "finetune_seq2seq"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            loader = AutoPeftModelForSeq2SeqLM if self.is_seq2seq else AutoPeftModelForCausalLM
            self.model = loader.from_pretrained(adapter_dir)
        except OSError as err:
            raise GeneratorUnavailable(f"could not load adapter {adapter_dir!r}: {err}") from err
        self.model.eval()

    def generate(self, row: dict) -> str:
        import torch

        decoding = self.cfg.get("decoding", {})
        sample = decoding.get("strategy") == "sample"
        kwargs = dict(
            max_new_tokens=int(decoding.get("max_new_tokens", 96)),
            do_sample=sample,
            pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
        )
        if sample:
            kwargs["temperature"] = float(decoding.get("temperature", 1.0))
            kwargs["top_p"] = float(decoding.get("top_p", 1.0))

        if self.is_seq2seq:
            src, _ = format_seq2seq(row)
            inputs = self.tokenizer(src, return_tensors="pt")
            with torch.no_grad():
                out = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        messages = format_chat(row, self.cfg["model"]["system_prompt"])[:-1]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model.generate(input_ids, **kwargs)
        text = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0]


def build_inference(cfg: dict) -> FinetunedGenerator:
    return FinetunedGenerator(cfg)
