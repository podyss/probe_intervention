import unittest
import multiprocessing
from unittest.mock import Mock, patch

from probe_int.vllm_dp import _rank_progress_factory, generate_with_dp, partition_prompts


class DataParallelPartitionTests(unittest.TestCase):
    def test_round_robin_partition_preserves_each_request_once(self):
        shards = partition_prompts(["a", "b", "c", "d", "e"], 2)

        self.assertEqual(shards, [
            [(0, "a"), (2, "c"), (4, "e")],
            [(1, "b"), (3, "d")],
        ])

    def test_empty_rank_gets_placeholder_for_collective_progress(self):
        shards = partition_prompts(["only"], 2)

        self.assertEqual(shards[0], [(0, "only")])
        self.assertEqual(shards[1], [(None, "only")])

    def test_rank_progress_updates_shared_count_and_passes_render_iterable(self):
        counts = multiprocessing.get_context("spawn").Array("q", 2, lock=True)
        progress_factory = _rank_progress_factory(1, counts, max_count=2)

        prompts = ["prompt"]
        self.assertIs(progress_factory(prompts, desc="Rendering prompts"), prompts)
        progress = progress_factory(total=2, desc="Processed prompts")
        progress.update(1)

        self.assertEqual(progress.n, 1)
        self.assertGreater(progress.format_dict["elapsed"], 0)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 1)

    def test_progress_excludes_placeholder_completions(self):
        counts = multiprocessing.get_context("spawn").Array("q", 2, lock=True)
        real = _rank_progress_factory(0, counts, max_count=2)(total=2)
        placeholder = _rank_progress_factory(1, counts, max_count=0)(total=2)
        real.update(2)
        placeholder.update(2)

        self.assertEqual(list(counts[:]), [2, 0])

    def test_immediate_native_worker_failure_reports_rank_and_exit_code(self):
        # Native aborts may kill every child before the parent's polling loop
        # begins, leaving no Python error file. They must not count as success.
        context = Mock()
        context.Array.return_value = multiprocessing.get_context("spawn").Array(
            "q", 2, lock=True,
        )
        processes = [Mock(exitcode=-6), Mock(exitcode=-9)]
        for process in processes:
            process.is_alive.return_value = False
        context.Process.side_effect = processes
        options = {"seed": 42, "async_scheduling": True}
        with patch("probe_int.vllm_dp.multiprocessing.get_context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "DP rank 0 exited with code -6"):
                generate_with_dp(["a", "b"], {"max_tokens": 16}, options, 2, False)
        # Guard the external-launcher workaround without modifying caller config.
        for call in context.Process.call_args_list:
            self.assertFalse(call.kwargs["args"][5]["async_scheduling"])
        self.assertTrue(options["async_scheduling"])


if __name__ == "__main__":
    unittest.main()
