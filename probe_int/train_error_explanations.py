import argparse
import hashlib
import json
import os
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint


def make_batch(examples, pad_token_id):
    import torch

    width = max(len(example["input_ids"]) for example in examples)
    input_ids, labels, attention_masks = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        labels.append(example["labels"] + [-100] * padding)
        attention_masks.append([1] * len(example["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full-parameter SFT for first-error detection and explanation"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.batch_size,
           args.gradient_accumulation_steps, args.max_length) <= 0:
        parser.error("Training settings must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("gpu-memory-fraction must be in (0, 1]")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Full-parameter training requires torchrun with at least two GPUs")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty training output: {output}")

    import torch
    from datasets import Dataset
    from functools import partial
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for row in read_jsonl(args.data):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            completion_ids.append(tokenizer.eos_token_id)
        input_ids = prompt_ids + completion_ids
        if len(input_ids) > args.max_length:
            raise ValueError(
                f"{row['id']} has {len(input_ids)} tokens, above --max-length={args.max_length}"
            )
        if not completion_ids:
            raise ValueError(f"Empty target for {row['id']}")
        rows.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        })
    if not rows:
        raise ValueError("No training rows found")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    fsdp_layers = getattr(model, "_no_split_modules", None)
    if not fsdp_layers:
        raise ValueError("Model does not define transformer layers for FSDP auto-wrap")

    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": fsdp_layers,
            "use_orig_params": True,
        },
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        data_collator=partial(make_batch, pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output)
        metadata = {
            "arguments": vars(args),
            "input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "tokenizer_fingerprint": fingerprint,
            "examples": len(rows),
            "objective": "causal language modeling loss on assistant responses only",
            "train_metrics": result.metrics,
            "status": "complete",
        }
        output.joinpath("experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
