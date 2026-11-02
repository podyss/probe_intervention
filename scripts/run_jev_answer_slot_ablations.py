#!/usr/bin/env python3
"""Run matched one-epoch ablations and direct-answer MathQA evaluation on GPUs 2,3."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-ablations-epoch1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "scripts"
    source_dir.mkdir(exist_ok=True)
    scripts = [
        "train_jev_answer_slot_distill.py", "train_jev_full_trajectory.py",
        "evaluate_jev_zero_shot.py", "run_jev_answer_slot_ablations.py",
    ]
    for name in scripts:
        source = Path("scripts") / name
        snapshot = source_dir / name
        if snapshot.exists() and snapshot.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Source changed since suite was started: {name}")
        shutil.copyfile(source, snapshot)

    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", CUDA_VISIBLE_DEVICES="2,3",
               PYTHONUNBUFFERED="1")
    train_path = Path("data/jev-mathqa/train.jsonl")
    test_path = Path("data/jev-mathqa/test.jsonl")
    expected = {
        "model": args.model, "train": str(train_path), "epochs": 4,
        "completed_epochs": 1, "learning_rate": 2e-5, "global_batch_size": 32,
        "batch_size_per_gpu": 4, "gradient_accumulation_steps": 4,
        "examples": 2048, "max_length": 512,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    }
    common = [
        "--model", args.model, "--train", str(train_path),
        "--epochs", "4", "--stop-after-epoch", "1", "--learning-rate", "2e-5",
        "--batch-size", "4", "--gradient-accumulation-steps", "4",
        "--max-length", "512", "--max-train-examples", "2048", "--seed", "42",
    ]
    jobs = [
        ("ce-only", "CE only", "train_jev_answer_slot_distill.py", 0.0, 0.0),
        ("ce-logits", "CE + logits KL", "train_jev_answer_slot_distill.py", 1.0, 0.0),
        ("ce-hidden", "CE + hidden alignment", "train_jev_answer_slot_distill.py", 0.0, 0.1),
        ("full-trajectory-sft", "Full trajectory SFT", "train_jev_full_trajectory.py", None, None),
    ]
    results = []
    manifest = {
        "configuration": expected, "gpus": "2,3",
        "note": "Each new model is trained for exactly one epoch (64 updates), retaining the first epoch of a four-epoch LR schedule to match the existing combined-distillation checkpoint.",
        "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in scripts},
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "results": results,
    }

    def persist():
        (args.output / "results.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        lines = [
            "All runs: actual 1 epoch, 2048 examples, seed 42, global batch 32, GPUs 2,3.",
            "LR: first epoch of a four-epoch linear schedule, matching the reused combined checkpoint.",
            "Evaluation: direct answer-option readout on MathQA test; no generated rationale.",
            "", "| Method | Correct / total | Accuracy |",
            "|---|---:|---:|",
        ]
        for item in results:
            lines.append(f"| {item['method']} | {item['correct']}/{item['total']} | {item['accuracy']:.2%} |")
        (args.output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def validate_metadata(checkpoint, logit_weight=None, hidden_weight=None):
        metadata = json.loads((checkpoint / "experiment.json").read_text())
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Configuration mismatch in {checkpoint}: {key}: {metadata.get(key)!r} != {value!r}")
        if logit_weight is not None and metadata.get("logit_loss_weight", 1.0) != logit_weight:
            raise ValueError(f"Logit weight mismatch in {checkpoint}")
        if hidden_weight is not None and metadata.get("hidden_loss_weight") != hidden_weight:
            raise ValueError(f"Hidden weight mismatch in {checkpoint}")

    def add_result(name, method, checkpoint, predictions, **extra):
        rows = read_jsonl(predictions)
        expected_rows = read_jsonl(test_path)
        if [row["id"] for row in rows] != [row["id"] for row in expected_rows]:
            raise ValueError(f"Test IDs/order mismatch: {predictions}")
        if any(row["model"] != str(checkpoint) for row in rows):
            raise ValueError(f"Model mismatch: {predictions}")
        correct = sum(row["prediction_index"] == row["gold_index"] for row in rows)
        results.append({
            "name": name, "method": method, "checkpoint": str(checkpoint),
            "predictions": str(predictions), "correct": correct, "total": len(rows),
            "accuracy": correct / len(rows), **extra,
        })
        persist()
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    def run(command, log_path):
        started = time.monotonic()
        print(f"START {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(command)} log={log_path}", flush=True)
        with log_path.open("w") as log:
            log.write(shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        duration = time.monotonic() - started
        print(f"DONE {log_path} seconds={duration:.1f}", flush=True)
        return duration

    combined = Path("outputs/jev-answer-slot-distill-epoch1")
    validate_metadata(combined, 1.0, 0.1)
    add_result("ce-logits-hidden", "CE + logits KL + hidden alignment", combined,
               Path("outputs/jev-answer-slot-distill-epoch1-mathqa-test.jsonl"), reused=True)

    for name, method, script, logit_weight, hidden_weight in jobs:
        checkpoint = args.output / name
        train_seconds = None
        if not (checkpoint / "experiment.json").exists():
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc_per_node=2", str(source_dir / script), *common,
                       "--output", str(checkpoint)]
            if logit_weight is not None:
                command.extend(["--logit-loss-weight", str(logit_weight),
                                "--hidden-loss-weight", str(hidden_weight), "--temperature", "2.0"])
            train_seconds = run(command, args.output / f"{name}-train.log")
        validate_metadata(checkpoint, logit_weight, hidden_weight)
        predictions = args.output / f"{name}-mathqa-test.jsonl"
        eval_seconds = None
        if not predictions.exists():
            command = [sys.executable, str(source_dir / "evaluate_jev_zero_shot.py"),
                       "--model", str(checkpoint), "--input", str(test_path),
                       "--output", str(predictions), "--batch-size", "8",
                       "--data-parallel-size", "2", "--dtype", "bfloat16"]
            eval_seconds = run(command, args.output / f"{name}-eval.log")
        add_result(name, method, checkpoint, predictions, logit_loss_weight=logit_weight,
                   hidden_loss_weight=hidden_weight, train_seconds=train_seconds,
                   eval_seconds=eval_seconds, reused=False)
    manifest["status"] = "complete"
    persist()
    print("ALL ABLATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
