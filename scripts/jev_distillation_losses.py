"""Losses for answer-slot distillation experiments."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def answer_projected_hidden_kl(student_hidden, teacher_hidden, projection_weight,
                               norm_weight, norm_epsilon, valid_options,
                               temperature=2.0, already_normalized=False):
    """Compare intermediate answer beliefs using the SAME frozen teacher readout."""
    def project(hidden):
        if not already_normalized:
            hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + norm_epsilon)
            hidden = hidden * norm_weight
        return F.linear(hidden, projection_weight).masked_fill(~valid_options, -1e9)

    return F.kl_div(
        F.log_softmax(project(student_hidden) / temperature, dim=-1),
        F.softmax(project(teacher_hidden) / temperature, dim=-1), reduction="batchmean",
    ) * temperature ** 2


def reasoning_gain_mask(teacher_option_logits, baseline_option_logits, targets):
    """Keep examples where the rationale increases gold-option probability (T=1)."""
    with torch.no_grad():
        gold = targets[:, None]
        gain = (
            teacher_option_logits.softmax(-1).gather(1, gold)
            - baseline_option_logits.softmax(-1).gather(1, gold)
        ).squeeze(1)
    return gain > 0, gain


def gated_logit_kl(student_logits, teacher_logits, temperature, mask):
    """Normalize by selected examples across GPUs; return differentiable zero if none."""
    per_example = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1), reduction="none",
    ).sum(-1) * temperature ** 2
    selected_count = mask.sum().to(per_example.dtype)
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(selected_count)
    # FSDP averages parameter gradients across ranks.
    return (per_example * mask).sum() * world_size / selected_count.clamp_min(1)


def gain_weighted_logit_kl(student_logits, teacher_logits, temperature, gains):
    """Weight each example's KL by its positive gold-option probability gain."""
    per_example = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1), reduction="none",
    ).sum(-1) * temperature ** 2
    weights = gains.clamp_min(0).to(per_example.dtype)
    weight_sum = weights.sum()
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(weight_sum)
    return (per_example * weights).sum() * world_size / weight_sum.clamp_min(1e-12)


def relational_hidden_kl(student_hidden, teacher_hidden, temperature=0.1):
    """Match centered pairwise cosine distributions, excluding each query itself.

    Gather the two GPU microbatches (eight examples in the current setup).
    Student gathering is differentiable so key and centering gradients are kept.
    Each rank evaluates only its own query rows before FSDP gradient averaging.
    """
    rank = 0
    batch_size = student_hidden.shape[0]
    if dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather

        rank = dist.get_rank()
        student_global = torch.cat(all_gather(student_hidden), dim=0)
        teacher_parts = [torch.empty_like(teacher_hidden) for _ in range(dist.get_world_size())]
        dist.all_gather(teacher_parts, teacher_hidden.contiguous())
        teacher_global = torch.cat(teacher_parts, dim=0)
    else:
        student_global, teacher_global = student_hidden, teacher_hidden
    if student_global.shape[0] < 3:
        raise ValueError("Relational hidden KL needs at least three examples across GPUs")
    student_global = F.normalize(student_global - student_global.mean(0), dim=-1)
    teacher_global = F.normalize(teacher_global - teacher_global.mean(0), dim=-1)
    start = rank * batch_size
    student_scores = student_global[start:start + batch_size] @ student_global.T / temperature
    teacher_scores = teacher_global[start:start + batch_size] @ teacher_global.T / temperature
    self_mask = torch.arange(student_global.shape[0], device=student_hidden.device)[None, :] == (
        torch.arange(batch_size, device=student_hidden.device)[:, None] + start
    )
    # Finite masking avoids 0 * inf in KL and assigns zero probability to self.
    student_scores = student_scores.masked_fill(self_mask, -1e4)
    teacher_scores = teacher_scores.masked_fill(self_mask, -1e4)
    return F.kl_div(
        F.log_softmax(student_scores, dim=-1),
        F.softmax(teacher_scores, dim=-1), reduction="batchmean",
    )
