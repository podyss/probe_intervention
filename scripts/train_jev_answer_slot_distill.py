#!/usr/bin/env python3
"""Distill full-trace answer-slot logits and hidden states into answer-only inputs."""

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def remove_answer_disclosure(rationale):
    rationale = rationale.strip().strip('"')
    match = re.search(
        r"\b(?:answer\s*:|the\s+answer\s+is|answer\s+is|option\s+[a-e]\b)",
        rationale,
        re.IGNORECASE,
    )
    if match:
        rationale = rationale[:match.start()].rstrip(" .,:;\n")
    return rationale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=Path("data/jev-mathqa/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-answer-slot-distill"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Stop and save after this epoch while keeping the full --epochs LR schedule")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", choices=("linear", "constant"), default="linear",
                        help="constant keeps the learning rate fixed, with no warmup")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--logit-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-kl-field", type=str, default=None,
                        help="Boolean row field whose examples maximize the teacher KL instead of minimizing it")
    parser.add_argument("--rationale-loss-weight", type=float, default=0.0,
                        help="Add causal LM loss for rationale tokens placed after the gold answer")
    parser.add_argument("--logit-kl-scope", choices=("full-vocab", "valid-options"), default="full-vocab",
                        help="Distill over the full vocabulary or only the example's valid answer labels")
    parser.add_argument("--teacher-privileged-answer", action="store_true",
                        help="Expose the gold option label in the frozen teacher context only")
    parser.add_argument("--teacher-privileged-answer-position", choices=("before-rationale", "after-rationale"),
                        default="after-rationale",
                        help="Place the privileged gold option before or after the teacher rationale")
    parser.add_argument("--hidden-loss-weight", type=float, default=0.1)
    parser.add_argument("--hidden-layer-selection", choices=("quartiles", "eighths", "all", "last"), default="quartiles",
                        help="Align four or eight evenly spaced transformer layers, every layer, or only the last layer; exclude embeddings")
    parser.add_argument("--hidden-layers", type=str, default=None,
                        help="Comma-separated 1-indexed transformer layer numbers; overrides --hidden-layer-selection")
    parser.add_argument("--logit-distillation-mode", choices=("standard", "standard-jsd", "reasoning-gain", "reasoning-gain-weighted", "purified-pmi", "trajectory-pmi", "trajectory-consensus", "trajectory-answer"), default="standard",
                        help="use raw teacher KL/JSD, reasoning-gain weighting, or purified PMI targets with JSD")
    parser.add_argument("--trajectory-checkpoints", type=int, default=4,
                        help="number of evenly spaced reference-rationale prefixes for trajectory distillation")
    parser.add_argument("--trajectory-layer-progress", action="store_true",
                        help="for trajectory-answer hidden loss, map shallower student layers to earlier teacher rationale prefixes")
    parser.add_argument("--trajectory-layer-interpolate", action="store_true",
                        help="for trajectory-answer hidden loss, linearly interpolate adjacent rationale-prefix targets by student layer depth")
    parser.add_argument("--trajectory-hidden-confidence-gate", action="store_true",
                        help="weight trajectory hidden JSD by each teacher prefix's gold-option advantage over chance")
    parser.add_argument("--trajectory-contrast", choices=("question-gain", "reasoning-gain"),
                        default="question-gain",
                        help="subtract rationale-only probes or question-only probes when decomposing trajectory option logits")
    parser.add_argument("--pmi-beta", type=float, default=1.0)
    parser.add_argument("--pmi-clip", type=float, default=10.0)
    parser.add_argument("--pmi-aux-loss-weight", type=float, default=0.0,
                        help="Add purified-PMI JSD while retaining the selected logit distillation loss")
    parser.add_argument("--trajectory-kl-gate-weight", type=float, default=0.0,
                        help="Reweight standard per-example teacher KL using trajectory reliability")
    parser.add_argument("--hidden-loss-type", choices=("cosine", "relational-kl", "answer-kl"), default="cosine")
    parser.add_argument("--relation-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        parser.error("--stop-after-epoch must be between 0 and --epochs")
    if (args.answer_loss_weight < 0 or args.logit_loss_weight < 0 or args.hidden_loss_weight < 0
            or args.rationale_loss_weight < 0 or args.pmi_aux_loss_weight < 0
            or args.trajectory_kl_gate_weight < 0):
        parser.error("Distillation weights must be nonnegative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.trajectory_checkpoints < 2:
        parser.error("--trajectory-checkpoints must be at least 2")
    if args.trajectory_layer_progress and args.trajectory_layer_interpolate:
        parser.error("Choose only one of --trajectory-layer-progress and --trajectory-layer-interpolate")
    if args.trajectory_hidden_confidence_gate and args.logit_distillation_mode != "trajectory-answer":
        parser.error("--trajectory-hidden-confidence-gate requires --logit-distillation-mode trajectory-answer")
    if not math.isfinite(args.relation_temperature) or args.relation_temperature <= 0:
        parser.error("--relation-temperature must be finite and positive")
    if not math.isfinite(args.pmi_beta) or args.pmi_beta <= 0:
        parser.error("--pmi-beta must be finite and positive")
    if not math.isfinite(args.pmi_clip) or args.pmi_clip <= 0:
        parser.error("--pmi-clip must be finite and positive")
    if args.logit_distillation_mode == "purified-pmi" and args.logit_loss_weight <= 0:
        parser.error("purified-pmi requires a positive --logit-loss-weight")
    if args.negative_kl_field and (
            args.logit_loss_weight <= 0 or args.logit_distillation_mode != "standard"):
        parser.error("--negative-kl-field requires standard logits KL with a positive --logit-loss-weight")
    if args.pmi_aux_loss_weight > 0 and args.logit_loss_weight <= 0:
        parser.error("PMI auxiliary loss requires a positive --logit-loss-weight")
    if args.trajectory_kl_gate_weight > 0 and args.logit_distillation_mode != "trajectory-consensus":
        parser.error("--trajectory-kl-gate-weight requires --logit-distillation-mode trajectory-consensus")
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        parser.error("Full-parameter training requires torchrun with multiple GPUs")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {args.output}")

    import functools

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from jev_distillation_losses import (
        answer_projected_hidden_kl, gain_weighted_logit_kl, gated_logit_kl,
        reasoning_gain_mask, relational_hidden_kl,
    )
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidate_ids = []
    for label in label_chars:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer label {label!r} is not one token: {ids}")
        candidate_ids.append(ids[0])

    class AnswerSlotDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
            self.truncated = 0

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            options = "\n".join(
                f"({label_chars[i]}) {option}" for i, option in enumerate(row["options"])
            )
            user_content = f"Question: {row['question']}\nOptions:\n{options}"
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            rationale_ids = tokenizer.encode(
                remove_answer_disclosure(row.get("rationale", "")), add_special_tokens=False,
            )

            answer_prefix_ids = tokenizer.encode("Answer: (", add_special_tokens=False)
            gold_token_id = candidate_ids[row["gold_index"]]
            student_answer_position = len(prompt_ids) + len(answer_prefix_ids) - 1
            if args.rationale_loss_weight > 0:
                rationale_text = remove_answer_disclosure(row.get("rationale", ""))
                rationale_suffix_ids = tokenizer.encode(
                    f"\nReasoning: {rationale_text}" if rationale_text else "", add_special_tokens=False,
                )
                student_ids = prompt_ids + answer_prefix_ids + [gold_token_id] + rationale_suffix_ids
                student_lm_labels = [-100] * (len(prompt_ids) + len(answer_prefix_ids) + 1) + rationale_suffix_ids
            else:
                rationale_suffix_ids = []
                student_ids = prompt_ids + answer_prefix_ids
                student_lm_labels = [-100] * len(student_ids)
            student_was_truncated = len(student_ids) > args.max_length
            while len(student_ids) > args.max_length and rationale_suffix_ids:
                overflow = len(student_ids) - args.max_length
                del rationale_suffix_ids[max(0, len(rationale_suffix_ids) - max(overflow, 1)):]
                student_ids = (
                    prompt_ids + answer_prefix_ids + [gold_token_id] + rationale_suffix_ids
                    if args.rationale_loss_weight > 0 else prompt_ids + answer_prefix_ids
                )
                student_lm_labels = (
                    [-100] * (len(prompt_ids) + len(answer_prefix_ids) + 1) + rationale_suffix_ids
                    if args.rationale_loss_weight > 0 else [-100] * len(student_ids)
                )
            if len(student_ids) > args.max_length:
                raise ValueError(f"Question and options exceed max length for {row['id']}")

            def make_teacher_prefix(reason_ids):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                privileged = (
                    f"Known correct option: ({label_chars[row['gold_index']]})\n"
                    if args.teacher_privileged_answer else ""
                )
                reasoning = f"Reasoning: {reason}\n" if reason else ""
                continuation = (
                    privileged + reasoning
                    if args.teacher_privileged_answer_position == "before-rationale" else
                    reasoning + privileged
                )
                continuation += "Answer: ("
                return prompt_ids + tokenizer.encode(continuation, add_special_tokens=False)

            def make_reference_prefix(reason_ids, include_options=False):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                option_context = f"Options:\n{options}\n" if include_options else ""
                reference_content = f"{option_context}Reference solution:\n{reason}"
                reference_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": reference_content}], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False,
                )
                return reference_ids + answer_prefix_ids

            def make_matched_trajectory_prefix(reason_ids, include_question):
                reason = tokenizer.decode(reason_ids, skip_special_tokens=True).strip()
                question_field = f"Question: {row['question']}\n" if include_question else ""
                content = (
                    f"{question_field}Options:\n{options}\n"
                    f"Reference solution so far:\n{reason}\n"
                    "Choose the correct option based on the available information."
                )
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False,
                )
                return ids + answer_prefix_ids

            teacher_rationale_ids = tokenizer.encode(
                remove_answer_disclosure(row.get("teacher_rationale", row.get("rationale", ""))),
                add_special_tokens=False,
            )
            trajectory_teacher_ids = None
            trajectory_answer_ids = None
            trajectory_reference_ids = None
            matched_teacher_trajectory_ids = None
            matched_reference_trajectory_ids = None
            if args.logit_distillation_mode == "trajectory-consensus":
                consensus_prefixes = [
                    teacher_rationale_ids[:round(len(teacher_rationale_ids) * index / (args.trajectory_checkpoints - 1))]
                    for index in range(args.trajectory_checkpoints)
                ]
                matched_teacher_trajectory_ids = [
                    make_matched_trajectory_prefix(prefix, include_question=True)
                    for prefix in consensus_prefixes
                ]
                matched_reference_trajectory_ids = [
                    make_matched_trajectory_prefix(
                        prefix if args.trajectory_contrast == "question-gain" else [],
                        include_question=args.trajectory_contrast == "reasoning-gain",
                    )
                    for prefix in consensus_prefixes
                ]
                longest = max(
                    max(map(len, matched_teacher_trajectory_ids)),
                    max(map(len, matched_reference_trajectory_ids)),
                )
                if longest > args.max_length:
                    budget = max(0, len(teacher_rationale_ids) - (longest - args.max_length))
                    teacher_rationale_ids = teacher_rationale_ids[:budget]
                    consensus_prefixes = [
                        teacher_rationale_ids[:round(len(teacher_rationale_ids) * index / (args.trajectory_checkpoints - 1))]
                        for index in range(args.trajectory_checkpoints)
                    ]
                    matched_teacher_trajectory_ids = [
                        make_matched_trajectory_prefix(prefix, include_question=True)
                        for prefix in consensus_prefixes
                    ]
                    matched_reference_trajectory_ids = [
                        make_matched_trajectory_prefix(
                            prefix if args.trajectory_contrast == "question-gain" else [],
                            include_question=args.trajectory_contrast == "reasoning-gain",
                        )
                        for prefix in consensus_prefixes
                    ]
                if max(max(map(len, matched_teacher_trajectory_ids)), max(map(len, matched_reference_trajectory_ids))) > args.max_length:
                    raise ValueError(f"Matched trajectory prefixes exceed max length for {row['id']}")
            if args.logit_distillation_mode == "trajectory-pmi":
                trajectory_prefixes = [
                    teacher_rationale_ids[:round(len(teacher_rationale_ids) * index / (args.trajectory_checkpoints - 1))]
                    for index in range(args.trajectory_checkpoints)
                ]
                trajectory_teacher_ids = [make_teacher_prefix(prefix) for prefix in trajectory_prefixes]
                trajectory_reference_ids = [
                    make_reference_prefix(prefix, include_options=True) for prefix in trajectory_prefixes
                ]
                max_pair_length = max(
                    max(map(len, trajectory_teacher_ids)), max(map(len, trajectory_reference_ids)),
                )
                if max_pair_length > args.max_length:
                    budget = max(0, len(teacher_rationale_ids) - (max_pair_length - args.max_length))
                    teacher_rationale_ids = teacher_rationale_ids[:budget]
                    trajectory_prefixes = [
                        teacher_rationale_ids[:round(len(teacher_rationale_ids) * index / (args.trajectory_checkpoints - 1))]
                        for index in range(args.trajectory_checkpoints)
                    ]
                    trajectory_teacher_ids = [make_teacher_prefix(prefix) for prefix in trajectory_prefixes]
                    trajectory_reference_ids = [
                        make_reference_prefix(prefix, include_options=True) for prefix in trajectory_prefixes
                    ]
                if max(max(map(len, trajectory_teacher_ids)), max(map(len, trajectory_reference_ids))) > args.max_length:
                    raise ValueError(f"Trajectory prefixes exceed max length for {row['id']}")
            teacher_ids = make_teacher_prefix(teacher_rationale_ids)
            was_truncated = len(teacher_ids) > args.max_length
            while len(teacher_ids) > args.max_length and teacher_rationale_ids:
                overflow = len(teacher_ids) - args.max_length
                del teacher_rationale_ids[max(0, len(teacher_rationale_ids) - max(overflow, 1)):]
                teacher_ids = make_teacher_prefix(teacher_rationale_ids)
            if args.logit_distillation_mode == "trajectory-answer":
                trajectory_prefixes = [
                    teacher_rationale_ids[:round(len(teacher_rationale_ids) * index / (args.trajectory_checkpoints - 1))]
                    for index in range(args.trajectory_checkpoints)
                ]
                trajectory_answer_ids = [make_teacher_prefix(prefix) for prefix in trajectory_prefixes]
            reference_ids = (
                make_reference_prefix(teacher_rationale_ids)
                if args.logit_distillation_mode == "purified-pmi" or args.pmi_aux_loss_weight > 0 else None
            )
            while reference_ids is not None and len(reference_ids) > args.max_length and teacher_rationale_ids:
                overflow = len(reference_ids) - args.max_length
                del teacher_rationale_ids[max(0, len(teacher_rationale_ids) - max(overflow, 1)):]
                reference_ids = make_reference_prefix(teacher_rationale_ids)
            if len(student_ids) > args.max_length or len(teacher_ids) > args.max_length:
                raise ValueError(f"Question and options exceed max length for {row['id']}")
            if reference_ids is not None and len(reference_ids) > args.max_length:
                raise ValueError(f"Reference solution exceeds max length for {row['id']}")
            self.truncated += int(was_truncated or student_was_truncated)
            return {
                "student_input_ids": student_ids,
                "student_lm_labels": student_lm_labels,
                "student_answer_position_from_end": len(student_ids) - student_answer_position,
                "teacher_input_ids": teacher_ids,
                "reference_input_ids": reference_ids,
                "teacher_trajectory_input_ids": trajectory_teacher_ids,
                "trajectory_answer_input_ids": trajectory_answer_ids,
                "reference_trajectory_input_ids": trajectory_reference_ids,
                "matched_teacher_trajectory_input_ids": matched_teacher_trajectory_ids,
                "matched_reference_trajectory_input_ids": matched_reference_trajectory_ids,
                "target": row["gold_index"],
                "option_count": len(row["options"]),
                "negative_kl": bool(row.get(args.negative_kl_field, False)) if args.negative_kl_field else False,
            }

    def collate(examples):
        batch = {}
        names = ["student_input_ids"]
        if args.logit_distillation_mode in ("trajectory-pmi", "trajectory-consensus", "trajectory-answer"):
            trajectory_names = (
                ("matched_teacher_trajectory_input_ids", "matched_reference_trajectory_input_ids")
                if args.logit_distillation_mode == "trajectory-consensus" else
                ("trajectory_answer_input_ids",)
                if args.logit_distillation_mode == "trajectory-answer" else
                ("teacher_trajectory_input_ids", "reference_trajectory_input_ids")
            )
            for name in trajectory_names:
                sequences = [sequence for item in examples for sequence in item[name]]
                width = max(map(len, sequences))
                ids, masks = [], []
                for sequence in sequences:
                    padding = width - len(sequence)
                    ids.append([tokenizer.pad_token_id] * padding + sequence)
                    masks.append([0] * padding + [1] * len(sequence))
                prefix = name.removesuffix("_input_ids")
                batch[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
                batch[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
            if args.logit_distillation_mode in ("trajectory-consensus", "trajectory-answer"):
                names.append("teacher_input_ids")
        else:
            names.append("teacher_input_ids")
        if (args.logit_distillation_mode == "purified-pmi"
                or (args.pmi_aux_loss_weight > 0 and args.logit_distillation_mode not in ("trajectory-consensus", "trajectory-answer"))):
            names.append("reference_input_ids")
        for name in names:
            width = max(len(item[name]) for item in examples)
            ids, masks = [], []
            for item in examples:
                padding = width - len(item[name])
                ids.append([tokenizer.pad_token_id] * padding + item[name])
                masks.append([0] * padding + [1] * len(item[name]))
            prefix = name.removesuffix("_input_ids")
            batch[f"{prefix}_input_ids"] = torch.tensor(ids, dtype=torch.long)
            batch[f"{prefix}_attention_mask"] = torch.tensor(masks, dtype=torch.long)
        width = batch["student_input_ids"].shape[1]
        labels = []
        for item in examples:
            padding = width - len(item["student_lm_labels"])
            labels.append([-100] * padding + item["student_lm_labels"])
        batch["student_lm_labels"] = torch.tensor(labels, dtype=torch.long)
        batch["student_answer_position_from_end"] = torch.tensor(
            [item["student_answer_position_from_end"] for item in examples], dtype=torch.long,
        )
        batch["target"] = torch.tensor([item["target"] for item in examples], dtype=torch.long)
        batch["option_count"] = torch.tensor([item["option_count"] for item in examples], dtype=torch.long)
        batch["negative_kl"] = torch.tensor([item["negative_kl"] for item in examples], dtype=torch.bool)
        return batch

    rows = read_rows(args.train)
    if args.max_train_examples and len(rows) > args.max_train_examples:
        selection = random.Random(args.seed).sample(range(len(rows)), args.max_train_examples)
        rows = [rows[index] for index in selection]
    dataset = AnswerSlotDataset(rows)
    encoded = [dataset[index] for index in range(len(dataset))]
    truncated_examples = dataset.truncated

    def wrap_model(model):
        layer_classes = {
            module.__class__ for module in model.modules()
            if "DecoderLayer" in module.__class__.__name__
        }
        if not layer_classes:
            raise ValueError("Could not identify transformer decoder layers for FSDP auto-wrap")
        return FSDP(
            model,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_classes,
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

    use_teacher = args.logit_loss_weight > 0 or args.hidden_loss_weight > 0
    use_hidden = args.hidden_loss_weight > 0
    teacher = None
    if use_teacher:
        teacher_base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
        )
        teacher_base.config.use_cache = False
        teacher_base.eval()
        if ((use_hidden and args.hidden_loss_type == "answer-kl")
                or args.logit_distillation_mode in ("trajectory-consensus", "trajectory-answer")):
            # Copy the small readout before FSDP shards its source parameters.
            answer_projection = teacher_base.get_output_embeddings().weight[candidate_ids].detach().float().to(local_rank).clone()
            answer_norm_weight = teacher_base.model.norm.weight.detach().float().to(local_rank).clone()
            answer_norm_epsilon = teacher_base.config.rms_norm_eps
        teacher = wrap_model(teacher_base)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    student_base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False,
    )
    student_base.config.use_cache = False
    student_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_layer_count = student_base.config.num_hidden_layers
    if args.hidden_layers:
        try:
            hidden_layers = [int(layer.strip()) for layer in args.hidden_layers.split(",")]
        except ValueError as error:
            raise ValueError("--hidden-layers must be comma-separated integers") from error
        if (not hidden_layers or len(set(hidden_layers)) != len(hidden_layers)
                or any(layer < 1 or layer > hidden_layer_count for layer in hidden_layers)):
            raise ValueError(f"--hidden-layers must contain unique layer numbers in [1, {hidden_layer_count}]")
        hidden_layers = sorted(hidden_layers)
        hidden_layer_selection = "custom"
    else:
        selected_layer_count = {"eighths": 8, "last": 1}.get(args.hidden_layer_selection, 4)
        hidden_layers = (
            list(range(1, hidden_layer_count + 1)) if args.hidden_layer_selection == "all"
            else sorted({
                max(1, round(hidden_layer_count * index / selected_layer_count))
                for index in range(1, selected_layer_count + 1)
            })
        )
        hidden_layer_selection = args.hidden_layer_selection
    student = wrap_model(student_base)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler_size = (len(encoded) // world_size) * world_size
    batches_per_rank = math.ceil((sampler_size // world_size) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_rank / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = 0 if args.lr_scheduler == "constant" else max(1, round(total_steps * 0.03))
    update_step = 0
    completed_epochs = 0
    optimizer.zero_grad(set_to_none=True)
    temperature = args.temperature
    gain_statistics = torch.zeros(3, device=local_rank, dtype=torch.float64)

    class EncodedDataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    for epoch in range(args.epochs):
        sampler = DistributedSampler(
            EncodedDataset(), num_replicas=world_size, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            EncodedDataset(), batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
        )
        epoch_sums = torch.zeros(6, device=local_rank)
        for batch_index, batch in enumerate(loader):
            targets = batch.pop("target").to(local_rank)
            option_counts = batch.pop("option_count").to(local_rank)
            negative_kl_mask = batch.pop("negative_kl").to(local_rank)
            student_lm_labels = batch.pop("student_lm_labels").to(local_rank)
            answer_position_from_end = batch.pop("student_answer_position_from_end").to(local_rank)
            batch = {key: value.to(local_rank) for key, value in batch.items()}
            trajectory_target_log_probs = None
            if args.logit_distillation_mode in ("trajectory-pmi", "trajectory-consensus", "trajectory-answer"):
                checkpoint_count = args.trajectory_checkpoints
                if args.logit_distillation_mode in ("trajectory-consensus", "trajectory-answer"):
                    teacher_batch = {
                        "input_ids": batch.pop("teacher_input_ids"),
                        "attention_mask": batch.pop("teacher_attention_mask"),
                    }
                    if args.logit_distillation_mode == "trajectory-consensus":
                        teacher_trajectory_batch = {
                            "input_ids": batch.pop("matched_teacher_trajectory_input_ids"),
                            "attention_mask": batch.pop("matched_teacher_trajectory_attention_mask"),
                        }
                        reference_trajectory_batch = {
                            "input_ids": batch.pop("matched_reference_trajectory_input_ids"),
                            "attention_mask": batch.pop("matched_reference_trajectory_attention_mask"),
                        }
                    else:
                        teacher_trajectory_batch = {
                            "input_ids": batch.pop("trajectory_answer_input_ids"),
                            "attention_mask": batch.pop("trajectory_answer_attention_mask"),
                        }
                else:
                    teacher_trajectory_batch = {
                        "input_ids": batch.pop("teacher_trajectory_input_ids"),
                        "attention_mask": batch.pop("teacher_trajectory_attention_mask"),
                    }
                    reference_trajectory_batch = {
                        "input_ids": batch.pop("reference_trajectory_input_ids"),
                        "attention_mask": batch.pop("reference_trajectory_attention_mask"),
                    }
            else:
                teacher_batch = {
                    "input_ids": batch.pop("teacher_input_ids"),
                    "attention_mask": batch.pop("teacher_attention_mask"),
                }
            reference_batch = None
            if (args.logit_distillation_mode == "purified-pmi"
                    or (args.pmi_aux_loss_weight > 0
                        and args.logit_distillation_mode not in ("trajectory-consensus", "trajectory-answer"))):
                reference_batch = {
                    "input_ids": batch.pop("reference_input_ids"),
                    "attention_mask": batch.pop("reference_attention_mask"),
                }
            student_batch = {
                "input_ids": batch.pop("student_input_ids"),
                "attention_mask": batch.pop("student_attention_mask"),
            }
            accumulation_remainder = len(loader) % args.gradient_accumulation_steps
            group_size = (
                accumulation_remainder
                if accumulation_remainder and batch_index >= len(loader) - accumulation_remainder
                else args.gradient_accumulation_steps
            )
            sync_now = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            teacher_logits = None
            pmi_target_log_probs = None
            teacher_hidden = []
            trajectory_layer_targets = None
            trajectory_reliability = None
            gain_mask = None
            answer_positions = student_batch["input_ids"].shape[1] - answer_position_from_end
            if args.rationale_loss_weight > 0:
                rationale_positions = torch.where(student_lm_labels[:, 1:] != -100)[1]
                logit_positions = torch.unique(
                    torch.cat([answer_positions, rationale_positions]), sorted=True,
                )
                answer_logit_indices = torch.searchsorted(logit_positions, answer_positions)
            else:
                logit_positions = answer_positions
                answer_logit_indices = torch.arange(targets.shape[0], device=local_rank)
            if use_teacher:
                with torch.no_grad():
                    if args.logit_distillation_mode == "trajectory-consensus":
                        baseline_outputs = teacher(**teacher_batch, logits_to_keep=1)
                        teacher_logits = baseline_outputs.logits[:, -1, :].float()
                        del baseline_outputs
                        trajectory_teacher_outputs = teacher(
                            **teacher_trajectory_batch, logits_to_keep=1, output_hidden_states=True,
                        )
                        trajectory_reference_outputs = teacher(
                            **reference_trajectory_batch, logits_to_keep=1, output_hidden_states=True,
                        )
                        base_outputs = teacher(
                            **student_batch, logits_to_keep=1, output_hidden_states=True,
                        )
                        batch_size = targets.shape[0]
                        valid_options = (
                            torch.arange(len(label_chars), device=local_rank)[None, :]
                            < option_counts[:, None]
                        )

                        def projected_option_logits(hidden, layer):
                            if layer != hidden_layer_count:
                                hidden = hidden * torch.rsqrt(
                                    hidden.square().mean(-1, keepdim=True) + answer_norm_epsilon,
                                )
                                hidden = hidden * answer_norm_weight
                            return F.linear(hidden.float(), answer_projection)

                        all_targets = []
                        all_residuals = []
                        for layer in hidden_layers:
                            teacher_hidden_layer = trajectory_teacher_outputs.hidden_states[layer][:, -1, :].float().reshape(
                                batch_size, checkpoint_count, -1,
                            )
                            reference_hidden_layer = trajectory_reference_outputs.hidden_states[layer][:, -1, :].float().reshape(
                                batch_size, checkpoint_count, -1,
                            )
                            base_hidden_layer = base_outputs.hidden_states[layer][:, -1, :].float()
                            teacher_option_logits = projected_option_logits(
                                teacher_hidden_layer.reshape(-1, teacher_hidden_layer.shape[-1]), layer,
                            ).reshape(batch_size, checkpoint_count, -1).masked_fill(
                                ~valid_options[:, None, :], -1e9,
                            )
                            reference_option_logits = projected_option_logits(
                                reference_hidden_layer.reshape(-1, reference_hidden_layer.shape[-1]), layer,
                            ).reshape(batch_size, checkpoint_count, -1).masked_fill(
                                ~valid_options[:, None, :], -1e9,
                            )
                            base_option_logits = projected_option_logits(base_hidden_layer, layer).masked_fill(
                                ~valid_options, -1e9,
                            )
                            teacher_log_probs = F.log_softmax(teacher_option_logits / temperature, dim=-1)
                            reference_log_probs = F.log_softmax(reference_option_logits / temperature, dim=-1)
                            base_log_probs = F.log_softmax(base_option_logits / temperature, dim=-1)
                            residual = teacher_log_probs - reference_log_probs
                            residual_mean = (
                                residual * valid_options[:, None, :]
                            ).sum(dim=-1, keepdim=True) / option_counts[:, None, None]
                            residual = (residual - residual_mean).masked_fill(~valid_options[:, None, :], 0.0)
                            residual = args.pmi_clip * torch.tanh(residual / args.pmi_clip)
                            corrected_log_probs = F.log_softmax(
                                base_log_probs[:, None, :] + residual / args.pmi_beta,
                                dim=-1,
                            )
                            # Marginalize checkpoint targets in probability space, as required by expected KL.
                            all_targets.append(corrected_log_probs.exp().mean(dim=1))
                            all_residuals.append(residual)
                        trajectory_layer_targets = torch.stack(all_targets, dim=1)
                        residual_grid = torch.stack(all_residuals, dim=2)
                        mean_residual = residual_grid.mean(dim=(1, 2))
                        per_view_agreement = F.cosine_similarity(
                            residual_grid, mean_residual[:, None, None, :], dim=-1,
                        )
                        agreement = per_view_agreement.mean(dim=(1, 2)).clamp_min(0.0)
                        gold_gain = mean_residual.gather(1, targets[:, None]).squeeze(1)
                        non_gold_mask = valid_options.clone()
                        non_gold_mask.scatter_(1, targets[:, None], False)
                        distractor_mean = (
                            mean_residual.masked_fill(~non_gold_mask, 0.0).sum(dim=-1)
                            / non_gold_mask.sum(dim=-1).clamp_min(1)
                        )
                        positive_gain = (gold_gain - distractor_mean).clamp_min(0.0)
                        positive_gain = positive_gain / (1.0 + positive_gain)
                        trajectory_reliability = agreement * positive_gain
                        if args.pmi_aux_loss_weight > 0:
                            pmi_target_log_probs = trajectory_layer_targets.mean(dim=1).clamp_min(1e-12).log()
                        teacher_hidden = []
                        del trajectory_teacher_outputs, trajectory_reference_outputs, base_outputs
                        del all_targets, all_residuals, residual_grid, per_view_agreement
                    elif args.logit_distillation_mode == "trajectory-answer":
                        trajectory_outputs = teacher(
                            **teacher_trajectory_batch, logits_to_keep=1, output_hidden_states=True,
                        )
                        batch_size = targets.shape[0]
                        valid_options = (
                            torch.arange(len(label_chars), device=local_rank)[None, :]
                            < option_counts[:, None]
                        )
                        trajectory_logits = trajectory_outputs.logits[:, -1, :].float().reshape(
                            batch_size, checkpoint_count, -1,
                        )
                        teacher_logits = trajectory_logits[:, -1, :]
                        layer_targets = []
                        for layer in hidden_layers:
                            trajectory_hidden = trajectory_outputs.hidden_states[layer][:, -1, :].float()
                            if layer != hidden_layer_count:
                                trajectory_hidden = trajectory_hidden * torch.rsqrt(
                                    trajectory_hidden.square().mean(-1, keepdim=True) + answer_norm_epsilon,
                                )
                                trajectory_hidden = trajectory_hidden * answer_norm_weight
                            option_logits = F.linear(trajectory_hidden, answer_projection).masked_fill(
                                ~valid_options.repeat_interleave(checkpoint_count, dim=0), -1e9,
                            )
                            option_log_probs = F.log_softmax(
                                option_logits / temperature, dim=-1,
                            ).reshape(batch_size, checkpoint_count, -1)
                            if args.trajectory_layer_interpolate:
                                progress = (
                                    (layer - 1) / max(hidden_layer_count - 1, 1)
                                    * (checkpoint_count - 1)
                                )
                                lower_checkpoint = math.floor(progress)
                                upper_checkpoint = min(lower_checkpoint + 1, checkpoint_count - 1)
                                upper_weight = progress - lower_checkpoint
                                target_probs = (
                                    option_log_probs[:, lower_checkpoint, :].exp() * (1.0 - upper_weight)
                                    + option_log_probs[:, upper_checkpoint, :].exp() * upper_weight
                                )
                                layer_targets.append(target_probs)
                            elif args.trajectory_layer_progress:
                                checkpoint_index = min(
                                    checkpoint_count - 1,
                                    len(layer_targets) * checkpoint_count // len(hidden_layers),
                                )
                                layer_targets.append(option_log_probs[:, checkpoint_index, :].exp())
                            else:
                                layer_targets.append(option_log_probs.exp().mean(dim=1))
                        trajectory_layer_targets = torch.stack(layer_targets, dim=1)
                        del trajectory_outputs, trajectory_logits, layer_targets
                    elif args.logit_distillation_mode == "trajectory-pmi":
                        teacher_outputs = teacher(
                            **teacher_trajectory_batch, logits_to_keep=1,
                        )
                        reference_outputs = teacher(**reference_trajectory_batch, logits_to_keep=1)
                        batch_size = targets.shape[0]
                        all_teacher_logits = teacher_outputs.logits[:, -1, :].float().reshape(
                            batch_size, checkpoint_count, -1,
                        )
                        all_reference_logits = reference_outputs.logits[:, -1, :].float().reshape(
                            batch_size, checkpoint_count, -1,
                        )
                        valid_options = (
                            torch.arange(len(label_chars), device=local_rank)[None, :]
                            < option_counts[:, None]
                        )
                        teacher_option_logits = all_teacher_logits[:, :, candidate_ids].masked_fill(
                            ~valid_options[:, None, :], torch.finfo(torch.float32).min,
                        )
                        reference_option_logits = all_reference_logits[:, :, candidate_ids].masked_fill(
                            ~valid_options[:, None, :], torch.finfo(torch.float32).min,
                        )
                        teacher_option_log_probs = F.log_softmax(
                            teacher_option_logits / temperature, dim=-1,
                        )
                        reference_option_log_probs = F.log_softmax(
                            reference_option_logits / temperature, dim=-1,
                        )
                        base_outputs = teacher(**student_batch, logits_to_keep=1)
                        base_logits = base_outputs.logits[:, -1, :].float()[:, candidate_ids].masked_fill(
                            ~valid_options, torch.finfo(torch.float32).min,
                        )
                        base_log_probs = F.log_softmax(base_logits / temperature, dim=-1)
                        trajectory_gain = teacher_option_log_probs - reference_option_log_probs
                        trajectory_gain = trajectory_gain - trajectory_gain.mean(dim=-1, keepdim=True)
                        trajectory_gain = args.pmi_clip * torch.tanh(trajectory_gain / args.pmi_clip)
                        trajectory_target_log_probs = F.log_softmax(
                            base_log_probs + trajectory_gain.mean(dim=1) / args.pmi_beta,
                            dim=-1,
                        )
                        teacher_logits = all_teacher_logits[:, -1, :]
                        if use_hidden:
                            final_indices = torch.arange(
                                checkpoint_count - 1,
                                batch_size * checkpoint_count,
                                checkpoint_count,
                                device=local_rank,
                            )
                            final_teacher_batch = {
                                key: value.index_select(0, final_indices)
                                for key, value in teacher_trajectory_batch.items()
                            }
                            final_teacher_outputs = teacher(
                                **final_teacher_batch, logits_to_keep=1, output_hidden_states=True,
                            )
                            teacher_hidden = [
                                final_teacher_outputs.hidden_states[layer][:, -1, :].float()
                                for layer in hidden_layers
                            ]
                            del final_teacher_outputs, final_teacher_batch
                        del teacher_outputs, reference_outputs, base_outputs
                        del all_teacher_logits, all_reference_logits, teacher_option_logits
                        del reference_option_logits, teacher_option_log_probs, reference_option_log_probs
                        del base_logits, base_log_probs, trajectory_gain
                    else:
                        teacher_outputs = teacher(
                            **teacher_batch, logits_to_keep=1, output_hidden_states=use_hidden,
                        )
                        if args.logit_loss_weight > 0:
                            teacher_logits = teacher_outputs.logits[:, -1, :].float()
                    if (args.logit_distillation_mode == "purified-pmi"
                            or (args.pmi_aux_loss_weight > 0
                                and args.logit_distillation_mode != "trajectory-consensus")):
                        reference_outputs = teacher(**reference_batch, logits_to_keep=1)
                        reference_logits = reference_outputs.logits[:, -1, :].float()
                        del reference_outputs
                        base_outputs = teacher(**student_batch, logits_to_keep=1)
                        base_logits = base_outputs.logits[:, -1, :].float()
                        del base_outputs
                        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
                        reference_log_probs = F.log_softmax(reference_logits / temperature, dim=-1)
                        base_log_probs = F.log_softmax(base_logits / temperature, dim=-1)
                        transferable = teacher_log_probs - reference_log_probs
                        transferable = transferable - transferable.mean(dim=-1, keepdim=True)
                        transferable = args.pmi_clip * torch.tanh(transferable / args.pmi_clip)
                        pmi_target_log_probs = F.log_softmax(
                            base_log_probs + transferable / args.pmi_beta, dim=-1,
                        )
                        if use_hidden:
                            teacher_hidden = [
                                teacher_outputs.hidden_states[layer][:, -1, :].float()
                                for layer in hidden_layers
                            ]
                        del teacher_outputs
                        del reference_logits, base_logits, teacher_log_probs, reference_log_probs
                    elif args.logit_distillation_mode not in ("trajectory-pmi", "trajectory-consensus", "trajectory-answer"):
                        if use_hidden:
                            teacher_hidden = [
                                teacher_outputs.hidden_states[layer][:, -1, :].float()
                                for layer in hidden_layers
                            ]
                        del teacher_outputs
                    if args.logit_loss_weight > 0 and args.logit_distillation_mode in (
                        "reasoning-gain", "reasoning-gain-weighted",
                    ):
                        baseline_outputs = teacher(**student_batch, logits_to_keep=1)
                        baseline_logits = baseline_outputs.logits[:, -1, :].float()
                        valid_teacher_options = (
                            torch.arange(len(label_chars), device=local_rank)[None, :]
                            < option_counts[:, None]
                        )
                        gain_mask, reasoning_gain = reasoning_gain_mask(
                            teacher_logits[:, candidate_ids].masked_fill(~valid_teacher_options, -1e9),
                            baseline_logits[:, candidate_ids].masked_fill(~valid_teacher_options, -1e9),
                            targets,
                        )
                        gain_statistics += torch.stack([
                            gain_mask.sum().double(), reasoning_gain.sum().double(),
                            reasoning_gain.new_tensor(reasoning_gain.numel()).double(),
                        ])
                        del baseline_outputs, baseline_logits

            with student.no_sync() if not sync_now else torch.enable_grad():
                student_outputs = student(
                    **student_batch, logits_to_keep=logit_positions,
                    output_hidden_states=use_hidden,
                )
                batch_indices = torch.arange(targets.shape[0], device=local_rank)
                student_logits = student_outputs.logits[batch_indices, answer_logit_indices, :].float()
                option_logits = student_logits[:, candidate_ids]
                valid_options = (
                    torch.arange(len(label_chars), device=local_rank)[None, :]
                    < option_counts[:, None]
                )
                option_logits = option_logits.masked_fill(
                    ~valid_options, torch.finfo(option_logits.dtype).min,
                )
                answer_loss = F.cross_entropy(option_logits, targets)
                rationale_loss = answer_loss.new_zeros(())
                if args.rationale_loss_weight > 0:
                    lm_logits = student_outputs.logits
                    lm_targets = student_lm_labels[:, logit_positions + 1]
                    if (lm_targets != -100).any():
                        rationale_loss = F.cross_entropy(
                            lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1),
                            ignore_index=-100,
                        )
                logit_loss = answer_loss.new_zeros(())
                pmi_loss = answer_loss.new_zeros(())
                if args.logit_loss_weight > 0:
                    if args.logit_distillation_mode == "trajectory-pmi":
                        kl_student_logits = student_logits[:, candidate_ids].masked_fill(
                            ~valid_options, torch.finfo(student_logits.dtype).min,
                        )
                        student_log_probs = F.log_softmax(kl_student_logits / temperature, dim=-1)
                        student_probs = student_log_probs.exp()
                        target_probs = trajectory_target_log_probs.exp()
                        log_mixture = torch.logaddexp(
                            student_log_probs, trajectory_target_log_probs,
                        ) - math.log(2.0)
                        logit_loss = 0.5 * (
                            (student_probs * (student_log_probs - log_mixture)).sum(dim=-1)
                            + (target_probs * (trajectory_target_log_probs - log_mixture)).sum(dim=-1)
                        ).mean() * (temperature ** 2)
                    elif args.logit_distillation_mode in ("purified-pmi", "standard-jsd"):
                        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                        student_probs = student_log_probs.exp()
                        target_log_probs = (
                            pmi_target_log_probs if args.logit_distillation_mode == "purified-pmi"
                            else F.log_softmax(teacher_logits / temperature, dim=-1)
                        )
                        target_probs = target_log_probs.exp()
                        log_mixture = torch.logaddexp(
                            student_log_probs, target_log_probs,
                        ) - math.log(2.0)
                        logit_loss = 0.5 * (
                            (student_probs * (student_log_probs - log_mixture)).sum(dim=-1)
                            + (target_probs * (target_log_probs - log_mixture)).sum(dim=-1)
                        ).mean() * (temperature ** 2)
                    else:
                        if args.logit_kl_scope == "valid-options":
                            kl_student_logits = student_logits[:, candidate_ids].masked_fill(
                                ~valid_options, torch.finfo(student_logits.dtype).min,
                            )
                            kl_teacher_logits = teacher_logits[:, candidate_ids].masked_fill(
                                ~valid_options, torch.finfo(teacher_logits.dtype).min,
                            )
                        else:
                            kl_student_logits = student_logits
                            kl_teacher_logits = teacher_logits
                        if args.trajectory_kl_gate_weight > 0:
                            teacher_log_probs = F.log_softmax(kl_teacher_logits / temperature, dim=-1)
                            student_log_probs = F.log_softmax(kl_student_logits / temperature, dim=-1)
                            per_example_kl = (
                                teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
                            ).sum(dim=-1) * (temperature ** 2)
                            reliability = trajectory_reliability.clamp(0.0, 1.0)
                            kl_weights = 1.0 + args.trajectory_kl_gate_weight * (
                                reliability - reliability.mean()
                            )
                            kl_weights = kl_weights / kl_weights.mean().clamp_min(1e-6)
                            logit_loss = (per_example_kl * kl_weights).mean()
                        elif args.logit_distillation_mode == "reasoning-gain-weighted":
                            logit_loss = gain_weighted_logit_kl(
                                kl_student_logits, kl_teacher_logits, temperature, reasoning_gain,
                            )
                        elif gain_mask is not None:
                            logit_loss = gated_logit_kl(
                                kl_student_logits, kl_teacher_logits, temperature, gain_mask,
                            )
                        elif args.negative_kl_field:
                            teacher_log_probs = F.log_softmax(kl_teacher_logits / temperature, dim=-1)
                            student_log_probs = F.log_softmax(kl_student_logits / temperature, dim=-1)
                            per_example_kl = (
                                teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
                            ).sum(dim=-1) * (temperature ** 2)
                            signs = torch.where(negative_kl_mask, -1.0, 1.0).to(per_example_kl.dtype)
                            logit_loss = (per_example_kl * signs).mean()
                        else:
                            logit_loss = F.kl_div(
                                F.log_softmax(kl_student_logits / temperature, dim=-1),
                                F.softmax(kl_teacher_logits / temperature, dim=-1),
                                reduction="batchmean",
                            ) * (temperature ** 2)
                if args.pmi_aux_loss_weight > 0:
                    if args.logit_distillation_mode == "trajectory-consensus":
                        gated_student_logits = student_logits[:, candidate_ids].masked_fill(
                            ~valid_options, torch.finfo(student_logits.dtype).min,
                        )
                        gated_student_log_probs = F.log_softmax(
                            gated_student_logits / temperature, dim=-1,
                        )
                        gated_student_probs = gated_student_log_probs.exp()
                        gated_target_probs = pmi_target_log_probs.exp()
                        gated_target_log_probs = gated_target_probs.clamp_min(1e-12).log()
                        gated_mixture_log_probs = torch.logaddexp(
                            gated_student_log_probs, gated_target_log_probs,
                        ) - math.log(2.0)
                        per_example_pmi_jsd = 0.5 * (
                            (gated_student_probs * (gated_student_log_probs - gated_mixture_log_probs)).sum(-1)
                            + (gated_target_probs * (gated_target_log_probs - gated_mixture_log_probs)).sum(-1)
                        ) * (temperature ** 2)
                        pmi_loss = (per_example_pmi_jsd * trajectory_reliability).mean()
                    else:
                        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                        student_probs = student_log_probs.exp()
                        target_probs = pmi_target_log_probs.exp()
                        log_mixture = torch.logaddexp(
                            student_log_probs, pmi_target_log_probs,
                        ) - math.log(2.0)
                        pmi_loss = 0.5 * (
                            (student_probs * (student_log_probs - log_mixture)).sum(dim=-1)
                            + (target_probs * (pmi_target_log_probs - log_mixture)).sum(dim=-1)
                        ).mean() * (temperature ** 2)
                hidden_losses = []
                for layer_index, layer in enumerate(hidden_layers if use_hidden else []):
                    if args.logit_distillation_mode in ("trajectory-consensus", "trajectory-answer"):
                        student_hidden = student_outputs.hidden_states[layer][
                            batch_indices, answer_positions,
                        ].float()
                        if layer != hidden_layer_count:
                            student_hidden = student_hidden * torch.rsqrt(
                                student_hidden.square().mean(-1, keepdim=True) + answer_norm_epsilon,
                            )
                            student_hidden = student_hidden * answer_norm_weight
                        student_option_logits = F.linear(student_hidden, answer_projection).masked_fill(
                            ~valid_options, -1e9,
                        )
                        student_log_probs = F.log_softmax(student_option_logits / temperature, dim=-1)
                        target_probs = trajectory_layer_targets[:, layer_index, :]
                        target_log_probs = target_probs.clamp_min(1e-12).log()
                        mixture_log_probs = torch.logaddexp(student_log_probs, target_log_probs) - math.log(2.0)
                        per_example_jsd = 0.5 * (
                            (student_log_probs.exp() * (student_log_probs - mixture_log_probs)).sum(-1)
                            + (target_probs * (target_log_probs - mixture_log_probs)).sum(-1)
                        ) * (temperature ** 2)
                        hidden_losses.append(per_example_jsd)
                    else:
                        target_hidden = teacher_hidden[layer_index]
                        student_hidden = student_outputs.hidden_states[layer][:, -1, :].float()
                        if args.hidden_loss_type == "relational-kl":
                            hidden_losses.append(relational_hidden_kl(
                                student_hidden, target_hidden, args.relation_temperature,
                            ))
                        elif args.hidden_loss_type == "answer-kl":
                            hidden_losses.append(answer_projected_hidden_kl(
                                student_hidden, target_hidden, answer_projection,
                                answer_norm_weight, answer_norm_epsilon, valid_options,
                                temperature=temperature, already_normalized=layer == hidden_layer_count,
                            ))
                        else:
                            hidden_losses.append(
                                1.0 - F.cosine_similarity(student_hidden, target_hidden, dim=-1).mean()
                            )
                if args.logit_distillation_mode == "trajectory-consensus" and hidden_losses:
                    per_layer_loss = torch.stack(hidden_losses, dim=1)
                    hidden_loss = (
                        per_layer_loss * trajectory_reliability[:, None]
                    ).sum() / (trajectory_reliability.sum().clamp_min(1e-6) * len(hidden_layers))
                elif (args.logit_distillation_mode == "trajectory-answer"
                      and args.trajectory_hidden_confidence_gate and hidden_losses):
                    per_layer_loss = torch.stack(hidden_losses, dim=1)
                    gold_probs = trajectory_layer_targets.gather(
                        2, targets[:, None, None].expand(-1, len(hidden_layers), 1),
                    ).squeeze(-1)
                    chance = option_counts[:, None].to(gold_probs.dtype).reciprocal()
                    reliability = ((gold_probs - chance) / (1.0 - chance).clamp_min(1e-6)).clamp(0.0, 1.0)
                    hidden_loss = (per_layer_loss * reliability).sum() / reliability.sum().clamp_min(1e-6)
                else:
                    hidden_loss = (
                        torch.stack(hidden_losses).mean() if hidden_losses else answer_loss.new_zeros(())
                    )
                loss = (
                    args.answer_loss_weight * answer_loss + args.logit_loss_weight * logit_loss
                    + args.hidden_loss_weight * hidden_loss
                    + args.pmi_aux_loss_weight * pmi_loss
                    + args.rationale_loss_weight * rationale_loss
                )
                scaled_loss = loss / group_size
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
                scaled_loss.backward()
            epoch_sums += torch.stack([
                answer_loss.detach(), logit_loss.detach(), hidden_loss.detach(),
                pmi_loss.detach(), rationale_loss.detach(), loss.detach(),
            ])
            del student_outputs, teacher_logits, teacher_hidden, trajectory_target_log_probs

            if sync_now:
                update_step += 1
                grad_norm = student.clip_grad_norm_(1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at update {update_step}")
                lr_scale = (
                    1.0 if args.lr_scheduler == "constant" else
                    update_step / warmup_steps if update_step <= warmup_steps
                    else max(0.0, (total_steps - update_step) / max(1, total_steps - warmup_steps))
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * lr_scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if rank == 0 and update_step % 10 == 0:
                    print(
                        f"epoch={epoch + 1} step={update_step}/{total_steps} "
                        f"ce={answer_loss.item():.5f} kl={logit_loss.item():.5f} "
                        f"hidden={hidden_loss.item():.5f} rationale={rationale_loss.item():.5f} "
                        f"total={loss.item():.5f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8g}",
                        flush=True,
                    )

        dist.all_reduce(epoch_sums, op=dist.ReduceOp.SUM)
        mean_losses = (epoch_sums / (len(loader) * world_size)).tolist()
        completed_epochs = epoch + 1
        if rank == 0:
            print(
                f"epoch={epoch + 1} mean_ce={mean_losses[0]:.6f} "
                f"mean_kl={mean_losses[1]:.6f} mean_hidden={mean_losses[2]:.6f} "
                f"mean_pmi={mean_losses[3]:.6f} mean_rationale={mean_losses[4]:.6f} "
                f"mean_total={mean_losses[5]:.6f}", flush=True,
            )
        if args.stop_after_epoch and completed_epochs >= args.stop_after_epoch:
            break

    dist.all_reduce(gain_statistics)
    args.output.mkdir(parents=True, exist_ok=True)
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, state_config):
        state_dict = student.state_dict()
    if rank == 0:
        student.module.save_pretrained(
            args.output, state_dict=state_dict, safe_serialization=True, max_shard_size="5GB",
        )
        tokenizer.save_pretrained(args.output)
        metadata = {
            "model": args.model,
            "teacher": args.model if use_teacher else None,
            "train": str(args.train),
            "epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "optimizer_steps": update_step,
            "lr_schedule_steps": total_steps,
            "warmup_steps": warmup_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "lr_scheduler": args.lr_scheduler,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "world_size": world_size,
            "global_batch_size": world_size * args.batch_size * args.gradient_accumulation_steps,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "examples": len(rows),
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "max_train_examples": args.max_train_examples,
            "max_length": args.max_length,
            "student_parameter_dtype": "float32",
            "teacher_parameter_dtype": "bfloat16" if use_teacher else None,
            "logit_distillation": (
                f"standard full-vocabulary teacher KL reweighted per example by trajectory reliability (strength {args.trajectory_kl_gate_weight})"
                if args.trajectory_kl_gate_weight > 0 else
                (
                    f"final full-vocabulary teacher KL; hidden answer JSD targets linearly interpolate adjacent teacher distributions across {args.trajectory_checkpoints} rationale-prefix checkpoints by layer depth"
                    if args.trajectory_layer_interpolate else
                    f"final full-vocabulary teacher KL; hidden answer JSD targets map layer depth to one of {args.trajectory_checkpoints} rationale-prefix checkpoints"
                    if args.trajectory_layer_progress else
                    f"final full-vocabulary teacher KL; hidden answer JSD targets are the probability mean over {args.trajectory_checkpoints} evenly spaced full-rationale prefixes"
                )
                if args.logit_distillation_mode == "trajectory-answer" else
                f"final full-vocabulary teacher KL plus reliability-weighted layerwise answer JSD from probability-marginalized PMI across {args.trajectory_checkpoints} matched-format rationale prefixes"
                if args.logit_distillation_mode == "trajectory-consensus" else
                f"trajectory-aggregated PMI JSD over valid answer options at {args.trajectory_checkpoints} rationale-prefix checkpoints"
                if args.logit_distillation_mode == "trajectory-pmi" else
                "PMI-purified JSD over the full vocabulary at the next-answer-token position"
                if args.logit_distillation_mode == "purified-pmi" else
                "JSD against the raw frozen teacher distribution over the full vocabulary at the next-answer-token position"
                if args.logit_distillation_mode == "standard-jsd" else
                "temperature-scaled KL over valid answer-label logits at the next-answer-token position"
                if args.logit_kl_scope == "valid-options" else
                "temperature-scaled KL over the full vocabulary at the next-answer-token position"
            ),
            "logit_kl_scope": args.logit_kl_scope,
            "temperature": temperature,
            "logit_loss_weight": args.logit_loss_weight,
            "answer_loss_weight": args.answer_loss_weight,
            "negative_kl_field": args.negative_kl_field,
            "negative_kl_examples": sum(bool(row.get(args.negative_kl_field, False)) for row in rows)
            if args.negative_kl_field else 0,
            "rationale_loss_weight": args.rationale_loss_weight,
            "logit_distillation_mode": args.logit_distillation_mode,
            "pmi_beta": args.pmi_beta if args.logit_distillation_mode in ("purified-pmi", "trajectory-consensus") or args.pmi_aux_loss_weight > 0 else None,
            "pmi_soft_clip": args.pmi_clip if args.logit_distillation_mode in ("purified-pmi", "trajectory-consensus") or args.pmi_aux_loss_weight > 0 else None,
            "pmi_aux_loss_weight": args.pmi_aux_loss_weight,
            "trajectory_kl_gate_weight": args.trajectory_kl_gate_weight,
            "reasoning_gain_statistics": {
                "selected_examples": int(gain_statistics[0].item()),
                "observed_examples": int(gain_statistics[2].item()),
                "mean_gold_probability_gain": (gain_statistics[1] / gain_statistics[2].clamp_min(1)).item(),
            } if args.logit_distillation_mode in ("reasoning-gain", "reasoning-gain-weighted") else None,
            "answer_loss": ("cross-entropy over valid answer-letter logits"
                            if args.answer_loss_weight > 0 else "disabled"),
            "student_input": (
                "question and options followed by the gold answer and training rationale"
                if args.rationale_loss_weight > 0 else
                "question and options only, followed by Answer: ("
            ),
            "rationale_objective": (
                "causal LM loss on rationale tokens after the gold answer; answer token excluded"
                if args.rationale_loss_weight > 0 else None
            ),
            "teacher_input": (
                "question, options, the gold option label as privileged context before the full training rationale, followed by Answer: ("
                if args.teacher_privileged_answer and args.teacher_privileged_answer_position == "before-rationale" else
                "question, options, full training rationale, and the gold option label as privileged context, followed by Answer: ("
                if args.teacher_privileged_answer else
                "question, options, and full training rationale, followed by Answer: ("
            ),
            "reference_probe_input": (
                "question/options with rationale omitted at each checkpoint"
                if args.logit_distillation_mode == "trajectory-consensus" and args.trajectory_contrast == "reasoning-gain" else
                "same user-template and options as teacher, with question omitted at each rationale-prefix checkpoint"
                if args.logit_distillation_mode == "trajectory-consensus" else
                "valid options plus each rationale prefix, without the question, followed by Answer: ("
                if args.logit_distillation_mode == "trajectory-pmi" else
                "reference solution only, followed by Answer: ("
                if args.logit_distillation_mode == "purified-pmi" or args.pmi_aux_loss_weight > 0 else None
            ),
            "pmi_target": (
                "per-layer softmax(log p_base + centered soft-clipped(log p_teacher - log p_reference) / beta), then arithmetic probability average over trajectory prefixes"
                if args.logit_distillation_mode == "trajectory-consensus" else
                "softmax(mean-prefix-base-log-option + mean-prefix-centered-soft-clipped(log p_teacher - log p_reference) / beta), over valid options"
                if args.logit_distillation_mode == "trajectory-pmi" else
                "softmax(log p_base + tanh(center(log p_teacher - log p_reference) / c) * c / beta)"
                if args.logit_distillation_mode == "purified-pmi" or args.pmi_aux_loss_weight > 0 else None
            ),
            "trajectory_contrast": args.trajectory_contrast if args.logit_distillation_mode == "trajectory-consensus" else None,
            "teacher_privileged_answer": args.teacher_privileged_answer,
            "teacher_privileged_answer_position": (
                args.teacher_privileged_answer_position if args.teacher_privileged_answer else None
            ),
            "hidden_distillation": (
                (
                    "gold-option-advantage-weighted JSD between student per-layer valid-option distributions and depth-interpolated teacher prefix distributions"
                    if args.trajectory_hidden_confidence_gate and args.trajectory_layer_interpolate else
                    "gold-option-advantage-weighted JSD between student per-layer valid-option distributions and depth-matched teacher prefix distributions"
                    if args.trajectory_hidden_confidence_gate and args.trajectory_layer_progress else
                    "gold-option-advantage-weighted JSD between student per-layer valid-option distributions and probability-averaged teacher prefix distributions"
                    if args.trajectory_hidden_confidence_gate else
                    "JSD between student question-only per-layer valid-option distributions and depth-interpolated teacher distributions across adjacent rationale prefixes"
                    if args.trajectory_layer_interpolate else
                    "JSD between student question-only per-layer valid-option distributions and teacher distributions at depth-matched rationale prefixes"
                    if args.trajectory_layer_progress else
                    "JSD between student question-only per-layer valid-option distributions and the probability-averaged teacher distributions at evenly spaced full-rationale prefixes"
                )
                if args.logit_distillation_mode == "trajectory-answer" else
                "reliability-weighted JSD between student question-only per-layer valid-option distributions and probability-marginalized, reference-debiased teacher trajectory targets"
                if args.logit_distillation_mode == "trajectory-consensus" else
                "KL between centered cross-GPU pairwise cosine distributions, excluding self; averaged across selected layers"
                if args.hidden_loss_type == "relational-kl" else
                "KL between valid-option distributions projected from selected hidden layers through the frozen original teacher RMSNorm and answer readout; final hidden state is already normalized"
                if args.hidden_loss_type == "answer-kl" else
                "cross-context cosine alignment at the next-answer-token position, averaged across selected transformer layers"
            ),
            "hidden_loss_type": (
                "trajectory-answer-interpolated-progress-jsd" if args.logit_distillation_mode == "trajectory-answer" and args.trajectory_layer_interpolate else
                "trajectory-answer-progress-jsd" if args.logit_distillation_mode == "trajectory-answer" and args.trajectory_layer_progress else
                "trajectory-answer-jsd"
                if args.logit_distillation_mode == "trajectory-answer" else args.hidden_loss_type
            ),
            "trajectory_layer_progress": args.trajectory_layer_progress,
            "trajectory_layer_interpolate": args.trajectory_layer_interpolate,
            "trajectory_hidden_confidence_gate": args.trajectory_hidden_confidence_gate,
            "relation_temperature": args.relation_temperature,
            "relation_batch_size": world_size * args.batch_size if args.hidden_loss_type == "relational-kl" else None,
            "hidden_layer_selection": hidden_layer_selection,
            "hidden_layers_1_indexed": hidden_layers,
            "hidden_loss_weight": args.hidden_loss_weight,
            "truncated_examples": truncated_examples,
            "max_grad_norm": 1.0,
            "status": "complete" if completed_epochs == args.epochs else "stopped_after_requested_epoch",
        }
        (args.output / "experiment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
