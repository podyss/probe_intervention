import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.jev_distillation_losses import (
    answer_projected_hidden_kl, gain_weighted_logit_kl, gated_logit_kl,
    reasoning_gain_mask, relational_hidden_kl,
)


def relation_inputs():
    generator = torch.Generator().manual_seed(7)
    return (torch.randn(8, 12, generator=generator, dtype=torch.float64),
            torch.randn(8, 12, generator=generator, dtype=torch.float64))


def distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    student, teacher = relation_inputs()
    local_student = student[rank * 4:(rank + 1) * 4].clone().requires_grad_()
    loss = relational_hidden_kl(local_student, teacher[rank * 4:(rank + 1) * 4])
    loss.backward()
    # Also exercise the case where only one rank has selected KD examples.
    local_logits = local_student[:, :5].detach().clone().requires_grad_()
    selected = torch.full((4,), rank == 1, dtype=torch.bool)
    gate_loss = gated_logit_kl(local_logits, teacher[rank * 4:(rank + 1) * 4, :5], 2, selected)
    gate_loss.backward()
    torch.save({"loss": loss.detach(), "grad": local_student.grad,
                "gate_loss": gate_loss.detach(), "gate_grad": local_logits.grad},
               Path(output_dir) / f"rank{rank}.pt")
    dist.destroy_process_group()


class DistillationLossTests(unittest.TestCase):
    def test_answer_projection_matches_manual_readout_and_masks_invalid_options(self):
        student, teacher = relation_inputs()
        student.requires_grad_()
        projection = torch.eye(12, dtype=torch.float64)[:5]
        norm_weight = torch.ones(12, dtype=torch.float64)
        valid = torch.arange(5)[None, :] < torch.tensor([5, 4, 3, 5, 5, 4, 3, 5])[:, None]
        loss = answer_projected_hidden_kl(student, teacher, projection, norm_weight, 1e-6, valid,
                                          already_normalized=True)
        expected = torch.nn.functional.kl_div(
            (student[:, :5].masked_fill(~valid, -1e9) / 2).log_softmax(-1),
            (teacher[:, :5].masked_fill(~valid, -1e9) / 2).softmax(-1), reduction="batchmean",
        ) * 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[:, 5:].abs().sum().item(), 0)
        same = answer_projected_hidden_kl(teacher, teacher, projection, norm_weight, 1e-6, valid)
        self.assertLess(abs(same.item()), 1e-10)

    def test_gain_uses_gold_probability_and_excludes_ties(self):
        baseline = torch.tensor([[0., 0.], [0., 0.], [0., 0.]])
        teacher = torch.tensor([[2., 0.], [2., 0.], [0., 0.]])
        mask, gain = reasoning_gain_mask(teacher, baseline, torch.tensor([0, 1, 0]))
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertGreater(gain[0], 0)
        self.assertLess(gain[1], 0)

    def test_gating_matches_full_kl_or_returns_zero(self):
        student, teacher = relation_inputs()
        student.requires_grad_()
        all_mask = torch.ones(8, dtype=torch.bool)
        expected = torch.nn.functional.kl_div(
            (student / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction="batchmean",
        ) * 4
        torch.testing.assert_close(gated_logit_kl(student, teacher, 2, all_mask), expected)
        zero_loss = gated_logit_kl(student, teacher, 2, ~all_mask)
        zero_loss.backward()
        self.assertEqual(zero_loss.item(), 0)
        self.assertEqual(student.grad.abs().sum().item(), 0)

    def test_gain_weighting_uses_positive_gain_and_returns_zero_without_gain(self):
        student, teacher = relation_inputs()
        student.requires_grad_()
        gains = torch.tensor([0.1, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        weighted = gain_weighted_logit_kl(student, teacher, 2, gains)
        per_example = torch.nn.functional.kl_div(
            (student / 2).log_softmax(-1), (teacher / 2).softmax(-1), reduction="none",
        ).sum(-1) * 4
        expected = (per_example[:2] * gains[:2]).sum() / gains[:2].sum()
        torch.testing.assert_close(weighted, expected)
        zero = gain_weighted_logit_kl(student, teacher, 2, torch.zeros_like(gains))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_relations_allow_rotation_scale_and_translation(self):
        teacher, _ = relation_inputs()
        rotation, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))
        student = (3 * teacher @ rotation + 7).requires_grad_()
        loss = relational_hidden_kl(student, teacher)
        self.assertLess(abs(loss.item()), 1e-10)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_wrong_sample_relations_produce_learning_signal(self):
        student, teacher = relation_inputs()
        student.requires_grad_()
        loss = relational_hidden_kl(student, teacher)
        self.assertGreater(loss.item(), .01)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_two_rank_losses_and_gradients_match_global_reference(self):
        student, teacher = relation_inputs()
        student.requires_grad_()
        expected = relational_hidden_kl(student, teacher)
        expected.backward()
        logits = student[:, :5].detach().clone().requires_grad_()
        expected_gate = gated_logit_kl(logits, teacher[:, :5], 2, torch.arange(8) >= 4)
        expected_gate.backward()
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / "init"), directory), nprocs=2)
            results = [torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
        torch.testing.assert_close(sum(row["loss"] for row in results) / 2, expected.detach())
        torch.testing.assert_close(sum(row["gate_loss"] for row in results) / 2, expected_gate.detach())
        for rank, result in enumerate(results):
            sl = slice(rank * 4, (rank + 1) * 4)
            torch.testing.assert_close(result["grad"] / 2, student.grad[sl])
            torch.testing.assert_close(result["gate_grad"] / 2, logits.grad[sl])


if __name__ == "__main__":
    unittest.main()
