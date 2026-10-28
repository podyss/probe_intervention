import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .core import read_jsonl, tokenizer_fingerprint
from .prompts import JUDGE_PROBE


def make_batch(examples, pad_token_id):
    import torch

    width = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_masks, targets = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_masks.append([1] * len(example["input_ids"]) + [0] * padding)
        targets.append(example["target"])
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "target": torch.tensor(targets),
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
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.learning_rate, args.batch_size,
           args.gradient_accumulation_steps, args.max_length, args.lora_r) <= 0:
        parser.error("Training settings must be positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("gpu-memory-fraction must be in (0, 1]")
    if args.full_finetune and int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("--full-finetune requires torchrun with at least two GPU processes")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing nonempty training output: {output}")

    import torch
    import torch.nn.functional as F
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
    yes_ids = tokenizer.encode(" yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError("This probe requires Yes and No to each be a single token")

    rows = []
    label_counts = Counter()
    for row in read_jsonl(args.data):
        label = row.get("label", "").strip().lower()
        if label not in {"yes", "no"}:
            raise ValueError(f"Invalid label in {row.get('id')}: {label!r}")
        prompt = probe_prompt(tokenizer, row)
        if len(prompt) > args.max_length:
            raise ValueError(
                f"{row['id']} has {len(prompt)} tokens, above --max-length={args.max_length}"
            )
        rows.append({
            "input_ids": prompt,
            "target": 0 if label == "yes" else 1,
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
    if args.full_finetune:
        fsdp_layers = getattr(model, "_no_split_modules", None)
        if not fsdp_layers:
            raise ValueError("Model does not define transformer layers for FSDP auto-wrap")
    else:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ))
        model.enable_input_require_grads()

    class YesNoTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("target")
            attention_mask = inputs["attention_mask"]
            outputs = model(**inputs)
            last_positions = attention_mask.sum(dim=1) - 1
            rows_idx = torch.arange(len(last_positions), device=last_positions.device)
            candidate_logits = outputs.logits[rows_idx, last_positions][:, [yes_ids[0], no_ids[0]]]
            loss = F.cross_entropy(candidate_logits.float(), targets)
            return (loss, outputs) if return_outputs else loss

    training_args_kwargs = {}
    if args.full_finetune:
        training_args_kwargs.update(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": fsdp_layers,
                "use_orig_params": True,
            },
        )
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
        **training_args_kwargs,
    )
    trainer = YesNoTrainer(
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
            "label_counts": dict(label_counts),
            "objective": "cross_entropy over the exact lowercase yes/no tokens scored at inference",
            "train_metrics": result.metrics,
            "status": "complete",
        }
        output.joinpath("experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
