#!/usr/bin/env python3
"""Prepare IG-selected and random rationale-token subsets for answer-slot training."""

import argparse
import json
import random
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean_rationale(text):
    text = text.strip().strip('"')
    match = re.search(
        r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)",
        text,
        re.IGNORECASE,
    )
    return text[:match.start()].rstrip(" .,:;\n") if match else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/gsm-mc/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.top_k < 1 or args.ig_steps < 1 or args.batch_size < 1:
        parser.error("top-k, ig-steps, and batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {args.output_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Integrated Gradients")
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}, trust_remote_code=False,
    ).eval()
    embedding = model.get_input_embeddings()
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidate_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {ids}")
        candidate_ids.append(ids[0])

    rows = read_rows(args.input)
    if args.max_examples and len(rows) > args.max_examples:
        chosen = random.Random(args.seed).sample(range(len(rows)), args.max_examples)
        rows = [rows[index] for index in chosen]

    prepared = []
    for row in rows:
        options = "\n".join(
            f"({labels[index]}) {option}" for index, option in enumerate(row["options"])
        )
        user_content = f"Question: {row['question']}\nOptions:\n{options}"
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False,
        )
        rationale = clean_rationale(row.get("rationale", ""))
        continuation = f"Reasoning: {rationale}\nAnswer: (" if rationale else "Answer: ("
        tokenized = tokenizer(continuation, add_special_tokens=False, return_offsets_mapping=True)
        continuation_ids = tokenized["input_ids"]
        start = len("Reasoning: ") if rationale else -1
        end = start + len(rationale) if rationale else -1
        rationale_positions = [
            len(prompt_ids) + index
            for index, (left, right) in enumerate(tokenized["offset_mapping"])
            if rationale and right > start and left < end
        ]
        input_ids = prompt_ids + continuation_ids
        if len(input_ids) > args.max_length:
            raise ValueError(f"Teacher context exceeds max length for {row['id']}")
        if not rationale_positions:
            raise ValueError(f"No rationale tokens found for {row['id']}")
        prepared.append({
            "row": row,
            "input_ids": input_ids,
            "rationale_positions": rationale_positions,
            "target_token_id": candidate_ids[row["gold_index"]],
        })

    attribution_scores = {}
    model_device = embedding.weight.device
    step_alphas = [(index + 1) / args.ig_steps for index in range(args.ig_steps)]
    for batch_start in range(0, len(prepared), args.batch_size):
        batch = prepared[batch_start:batch_start + args.batch_size]
        width = max(len(item["input_ids"]) for item in batch)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=model_device)
        attention_mask = torch.zeros_like(input_ids)
        rationale_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, item in enumerate(batch):
            offset = width - len(item["input_ids"])
            input_ids[index, offset:] = torch.tensor(item["input_ids"], device=model_device)
            attention_mask[index, offset:] = 1
            rationale_mask[index, [offset + pos for pos in item["rationale_positions"]]] = True
        with torch.no_grad():
            original = embedding(input_ids).detach()
        baseline = original.masked_fill(rationale_mask.unsqueeze(-1), 0)
        delta = original - baseline
        grad_total = torch.zeros_like(original, dtype=torch.float32)
        for alpha in step_alphas:
            interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
            outputs = model(
                inputs_embeds=interpolated,
                attention_mask=attention_mask,
                logits_to_keep=1,
            )
            target_ids = torch.tensor(
                [item["target_token_id"] for item in batch], device=model_device,
            )
            target_logits = outputs.logits[:, -1, :].gather(1, target_ids[:, None]).sum()
            gradients = torch.autograd.grad(target_logits, interpolated)[0]
            grad_total += gradients.float()
            del outputs, interpolated, gradients, target_logits
        token_attribution = (delta.float() * (grad_total / args.ig_steps)).sum(dim=-1).abs()
        for index, item in enumerate(batch):
            scores = token_attribution[index]
            valid_positions = [
                width - len(item["input_ids"]) + pos for pos in item["rationale_positions"]
            ]
            k = min(args.top_k, len(valid_positions))
            chosen_positions = torch.topk(scores[valid_positions], k=k).indices.tolist()
            selected = sorted(valid_positions[position] for position in chosen_positions)
            selected_ids = [int(input_ids[index, position].item()) for position in selected]
            attribution_scores[item["row"]["id"]] = selected_ids
        del input_ids, attention_mask, rationale_mask, original, baseline, delta, grad_total
        del token_attribution
        if (batch_start + len(batch)) % 100 < args.batch_size:
            print(f"Attributed {batch_start + len(batch)}/{len(prepared)}", flush=True)

    ig_rows, random_rows = [], []
    for item in prepared:
        row = dict(item["row"])
        full_rationale = clean_rationale(row.get("rationale", ""))
        all_ids = tokenizer.encode(full_rationale, add_special_tokens=False)
        selected_ids = attribution_scores[row["id"]]
        selected_text = tokenizer.decode(selected_ids, skip_special_tokens=True).strip()
        row["teacher_rationale"] = full_rationale
        row["rationale"] = selected_text
        row["rationale_selection"] = "integrated_gradients_top_k_abs"
        row["rationale_selected_token_count"] = len(selected_ids)
        ig_rows.append(row)

        random_row = dict(item["row"])
        random_row["teacher_rationale"] = full_rationale
        rng = random.Random(f"{args.seed}:{random_row['id']}:random-rationale")
        random_ids = sorted(rng.sample(range(len(all_ids)), min(args.top_k, len(all_ids))))
        random_row["rationale"] = tokenizer.decode(
            [all_ids[index] for index in random_ids], skip_special_tokens=True,
        ).strip()
        random_row["rationale_selection"] = "random_k_tokens"
        random_row["rationale_selected_token_count"] = len(random_ids)
        random_rows.append(random_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, output_rows in (("ig-top15.jsonl", ig_rows), ("random-15.jsonl", random_rows)):
        output = args.output_dir / name
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
            encoding="utf-8",
        )
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": args.model,
        "input": str(args.input),
        "examples": len(prepared),
        "seed": args.seed,
        "top_k": args.top_k,
        "ig_steps": args.ig_steps,
        "attribution": "absolute signed Integrated Gradients of the gold-answer next-token logit with zero rationale-embedding baseline",
        "selection_order": "selected tokens retain their original trajectory order",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prepared)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
