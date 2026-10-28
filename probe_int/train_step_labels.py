import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint
from .prompts import JUDGE_PROBE


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


def probe_prompt(tokenizer, row):
    content = JUDGE_PROBE.format(
        question=row["question"],
        gold_answer=row["reference_solution"],
        prefix=row["reasoning_prefix"],
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    think_open = tokenizer.encode("<think>\n", add_special_tokens=False)
    if think_open and prompt[-len(think_open):] == think_open:
        prompt = prompt[:-len(think_open)]
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a Yes/No probe on teacher-labeled reasoning prefixes"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.batch_size,
           args.gradient_accumulation_steps, args.max_length, args.lora_r) <= 0:
        parser.error("Training settings must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("gpu-memory-fraction must be in (0, 1]")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty training output: {output}")

    import torch
    from datasets import Dataset
    from functools import partial
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    fingerprint = tokenizer_fingerprint(tokenizer)
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("This probe requires Yes and No to each be a single token")

    rows = []
    label_counts = Counter()
    for row in read_jsonl(args.data):
        label = row.get("label", "").strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"Invalid label in {row.get('id')}: {label!r}")
        prompt = probe_prompt(tokenizer, row)
        target = yes_ids[0] if label == "yes" else no_ids[0]
        input_ids = prompt + [target]
        if len(input_ids) > args.max_length:
            raise ValueError(
                f"{row['id']} has {len(input_ids)} tokens, above --max-length={args.max_length}"
            )
        rows.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt) + [target],
        })
        label_counts[label] += 1
    if not rows or label_counts["yes"] != label_counts["no"]:
        raise ValueError(f"Expected a nonempty balanced Yes/No dataset, got {dict(label_counts)}")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    ))
    model.enable_input_require_grads()
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
    tokenizer.save_pretrained(output)
    metadata = {
        "arguments": vars(args),
        "input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "tokenizer_fingerprint": fingerprint,
        "examples": len(rows),
        "label_counts": dict(label_counts),
        "train_metrics": result.metrics,
        "status": "complete",
    }
    output.joinpath("experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
