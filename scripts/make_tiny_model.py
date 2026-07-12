#!/usr/bin/env python3
"""Build FROM-SCRATCH tiny models + a fixture-trained tokenizer, locally.

Purpose (PLAN.md D2/D3): huggingface.co is egress-blocked in the build
environment, so no pretrained weights exist here — but the Phase-4 gate
("smoke-train 50 steps on the fixture") is about proving the TRAINING
MACHINERY: dataset formatting, LoRA wrapping, Trainer loop, adapter
save/load, and harness evaluation. A randomly-initialised tiny T5 / tiny
Llama with a BPE tokenizer trained on the fixture exercises every one of
those paths with zero network access. Outputs of these models are noise;
runs built on them are named smoke_* and are machinery validation, never
results.

Usage:
    python scripts/make_tiny_model.py --out data/processed/tiny_random
    # then: make train-t5-smoke-tiny / train-qwen-smoke-tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def build_tokenizer(texts: list[str], vocab_size: int = 2000):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"],
    )
    tok.train_from_iterator(texts, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        # seq2seq/causal models take no token_type_ids; the fast-tokenizer
        # default of emitting them breaks model.generate input validation
        model_input_names=["input_ids", "attention_mask"],
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def build_t5(tokenizer, out_dir: Path) -> None:
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=64, d_ff=256, d_kv=16, num_heads=4,
        num_layers=2, num_decoder_layers=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = T5ForConditionalGeneration(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"tiny random T5 ({model.num_parameters():,} params) -> {out_dir}")


def build_causal(tokenizer, out_dir: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        # room for 8-shot prompts (the B2 smoke path feeds ~600-token prompts)
        max_position_embeddings=2048,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"tiny random Llama ({model.num_parameters():,} params) -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="data/fixtures/mini.jsonl")
    parser.add_argument("--out", default="data/processed/tiny_random")
    parser.add_argument("--which", choices=["both", "t5", "causal"], default="both")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in open(args.fixture)]
    texts = [r["linearized_input"] for r in rows] + [r["target_commentary"] for r in rows]
    tokenizer = build_tokenizer(texts)

    out = Path(args.out)
    if args.which in ("both", "t5"):
        build_t5(tokenizer, out / "t5")
    if args.which in ("both", "causal"):
        build_causal(tokenizer, out / "causal")


if __name__ == "__main__":
    main()
