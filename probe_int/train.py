import argparse
import json
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
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks)}


def main():
    parser = argparse.ArgumentParser(description="SFT on exported token-level intervention targets")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.batch_size, args.gradient_accumulation_steps, args.max_length) <= 0 or args.lora_r < 0:
        parser.error("Training settings must be positive; lora-r may be zero for full SFT")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty training output: {output}")
    manifest = Path(args.data).with_suffix(Path(args.data).suffix + ".manifest.json")
    if manifest.exists() and json.loads(manifest.read_text())["status"] != "complete":
        raise ValueError("Refusing SFT data from an incomplete export")

    import torch
    from datasets import Dataset
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    fingerprint = tokenizer_fingerprint(tokenizer)
    rows = []
    for row in read_jsonl(args.data):
        if row["split"] != "train" or row["tokenizer_fingerprint"] != fingerprint:
            raise ValueError("SFT requires training-only rows and an identical tokenizer/template")
        if len(row["input_ids"]) != len(row["labels"]) or not any(label != -100 for label in row["labels"][1:]):
            raise ValueError(f"Invalid labels in {row['id']}")
        if len(row["input_ids"]) > args.max_length:
            raise ValueError(f"{row['id']} exceeds max-length; refusing silent target truncation")
        rows.append({"input_ids": row["input_ids"], "labels": row["labels"]})
    if not rows:
        raise ValueError("No successful training examples; inspect collection/localization/repair summaries")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa", trust_remote_code=False,
    )
    model.config.use_cache = False
    if args.lora_r:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear",
            task_type="CAUSAL_LM",
        ))
        model.enable_input_require_grads()
    training_args = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=torch.cuda.is_available(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="epoch", report_to="none", seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=Dataset.from_list(rows),
        data_collator=partial(make_batch, pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        output.joinpath("experiment.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
