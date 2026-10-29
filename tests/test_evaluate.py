import json
import tempfile
import unittest
from pathlib import Path

from probe_int.evaluate import merge_dp_shards, visible_gpu_ids


class EvaluateMergeTests(unittest.TestCase):
    def test_merges_shards_in_source_order(self):
        source = [{"id": f"row-{index}"} for index in range(5)]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / f"dp{rank}.jsonl" for rank in range(2)]
            shards = [source[:2], source[2:]]
            for path, rows in zip(paths, shards):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            merged = merge_dp_shards(source, paths, dp_size=2)

        self.assertEqual([row["id"] for row in merged], [row["id"] for row in source])

    def test_rejects_incomplete_shard(self):
        source = [{"id": f"row-{index}"} for index in range(4)]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / f"dp{rank}.jsonl" for rank in range(2)]
            paths[0].write_text('{"id":"row-0"}\n')
            paths[1].write_text('{"id":"row-2"}\n{"id":"row-3"}\n')

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                merge_dp_shards(source, paths, dp_size=2)

    def test_rejects_duplicate_ids_inside_shard(self):
        source = [{"id": f"row-{index}"} for index in range(2)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dp0.jsonl"
            path.write_text('{"id":"row-0"}\n{"id":"row-0"}\n')

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                merge_dp_shards(source, [path], dp_size=1)


class VisibleGpuTests(unittest.TestCase):
    def test_uses_configured_gpu_identifiers(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,4,6"}):
            self.assertEqual(visible_gpu_ids(2), ["2", "4", "6"])

    def test_requires_enough_gpus(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            with self.assertRaisesRegex(ValueError, "Need 2 visible GPUs"):
                visible_gpu_ids(2)


if __name__ == "__main__":
    unittest.main()
