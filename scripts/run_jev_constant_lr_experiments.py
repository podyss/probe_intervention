#!/usr/bin/env python3
"""Train and evaluate five matched MathQA experiments at a constant learning rate."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/jev-constant-lr-2e-5"))
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--stop-on-collapse", action="store_true",
                        help="Stop the suite if at least 95%% of predictions select one option")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "scripts"
    source_dir.mkdir(exist_ok=True)
    names = ["train_jev_answer_slot_distill.py", "train_jev_full_trajectory.py",
             "evaluate_jev_zero_shot.py", Path(__file__).name]
    for name in names:
        source = Path("scripts") / name
        target = source_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"Source changed since suite was started: {name}")
        shutil.copyfile(source, target)

    train_path = Path("data/jev-mathqa/train.jsonl")
    test_path = Path("data/jev-mathqa/test.jsonl")
    expected_test = read_rows(test_path)
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", CUDA_VISIBLE_DEVICES="2,3",
               PYTHONUNBUFFERED="1")
    common = ["--model", args.model, "--train", str(train_path),
              "--learning-rate", str(args.learning_rate), "--lr-scheduler", "constant",
              "--batch-size", "4", "--gradient-accumulation-steps", "4",
              "--max-length", "512", "--max-train-examples", "2048", "--seed", "42"]
    jobs = [
        ("ce-only-epoch1", "直接监督答案 token", 1, "train_jev_answer_slot_distill.py", 0.0),
        ("full-sft-epoch1", "完整轨迹标准 SFT", 1, "train_jev_full_trajectory.py", None),
        ("full-sft-epoch3", "完整轨迹标准 SFT", 3, "train_jev_full_trajectory.py", None),
        ("ce-logits-epoch1", "Ours（CE＋logits KL）", 1, "train_jev_answer_slot_distill.py", 1.0),
        ("ce-logits-epoch3", "Ours（CE＋logits KL）", 3, "train_jev_answer_slot_distill.py", 1.0),
    ]
    manifest_path = args.output / "results.json"
    manifest = {
        "status": "running", "model": args.model, "gpus": "2,3",
        "learning_rate": args.learning_rate, "lr_scheduler": "constant", "warmup_steps": 0,
        "stop_on_collapse": args.stop_on_collapse, "collapse_threshold": 0.95,
        "global_batch_size": 32, "batch_size_per_gpu": 4, "gradient_accumulation_steps": 4,
        "examples": 2048, "seed": 42, "max_length": 512,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in names},
        "note": "Every job starts from the original model; fixed LR from the first to final update. Evaluation reads answer-option probabilities without generating a rationale.",
        "results": [],
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in manifest.keys() - {"status", "results"}:
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        previous_results = {row["name"]: row for row in previous["results"]}
    else:
        previous_results = {}

    def persist():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        lines = [f"所有实验：固定学习率 {args.learning_rate:g}，无 warmup、无衰减；从原始 Qwen3-4B 初始化。",
                 "2048 条训练样本，seed 42，max length 512，卡 2、3，全局 batch 32。",
                 "MathQA test 直接读取答案选项概率，不生成推理链。", "",
                 "| 方法 | 实际训练 epoch | 正确数 | 准确率 |", "|---|---:|---:|---:|"]
        for row in manifest["results"]:
            lines.append(f"| {row['method']} | {row['epochs']} | {row['correct']}/{row['total']} | {row['accuracy']:.2%} |")
        if manifest["status"] == "stopped_on_collapse":
            lines += ["", "检测到预测塌缩（至少 95% 样本选择同一选项），已停止后续实验。"]
        (args.output / "results.md").write_text("\n".join(lines) + "\n")

    def run(command, log_path):
        print(f"START {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(command)}", flush=True)
        started = time.monotonic()
        with log_path.open("w") as log:
            log.write(shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        seconds = time.monotonic() - started
        print(f"DONE {log_path} seconds={seconds:.1f}", flush=True)
        return seconds

    persist()
    try:
        for name, method, epochs, script, logit_weight in jobs:
            manifest["active_job"] = name
            persist()
            checkpoint = args.output / name
            previous_row = previous_results.get(name, {})
            train_seconds = previous_row.get("train_seconds")
            train_command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                             "--nproc_per_node=2", str(source_dir / script), *common,
                             "--epochs", str(epochs), "--output", str(checkpoint)]
            if logit_weight is not None:
                train_command += ["--temperature", "2.0", "--logit-loss-weight", str(logit_weight),
                                  "--hidden-loss-weight", "0.0"]
            metadata_path = checkpoint / "experiment.json"
            if not metadata_path.exists():
                train_seconds = run(train_command, args.output / f"{name}-train.log")
            metadata = json.loads(metadata_path.read_text())
            expected = {
                "model": args.model, "train": str(train_path), "epochs": epochs,
                "completed_epochs": epochs, "optimizer_steps": epochs * 64,
                "learning_rate": args.learning_rate, "lr_scheduler": "constant", "warmup_steps": 0,
                "final_learning_rate": args.learning_rate, "global_batch_size": 32,
                "batch_size_per_gpu": 4, "gradient_accumulation_steps": 4,
                "examples": 2048, "max_length": 512, "seed": 42,
                "train_sha256": manifest["train_sha256"], "status": "complete",
            }
            if logit_weight is not None:
                expected.update(logit_loss_weight=logit_weight, hidden_loss_weight=0.0,
                                temperature=2.0)
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"{name}: {key}: {metadata.get(key)!r} != {value!r}")
            logged_rates = re.findall(r"\blr=([\deE.+-]+)",
                                      (args.output / f"{name}-train.log").read_text())
            if not logged_rates or any(float(rate) != args.learning_rate for rate in logged_rates):
                raise ValueError(f"Learning rate log mismatch: {name}")
            predictions = args.output / f"{name}-mathqa-test.jsonl"
            eval_command = [sys.executable, str(source_dir / "evaluate_jev_zero_shot.py"),
                            "--model", str(checkpoint), "--input", str(test_path),
                            "--output", str(predictions), "--batch-size", "8",
                            "--data-parallel-size", "2", "--dtype", "bfloat16"]
            eval_seconds = previous_row.get("eval_seconds")
            if not predictions.exists():
                eval_seconds = run(eval_command, args.output / f"{name}-eval.log")
            rows = read_rows(predictions)
            if [row["id"] for row in rows] != [row["id"] for row in expected_test]:
                raise ValueError(f"Test IDs/order mismatch: {name}")
            if any(row["model"] != str(checkpoint) or row["gold_index"] != target["gold_index"]
                   for row, target in zip(rows, expected_test)):
                raise ValueError(f"Prediction model/gold mismatch: {name}")
            correct = sum(row["prediction_index"] == row["gold_index"] for row in rows)
            distribution = Counter(row["prediction_index"] for row in rows)
            dominant_share = max(distribution.values()) / len(rows)
            result = dict(name=name, method=method, epochs=epochs, correct=correct,
                          total=len(rows), accuracy=correct / len(rows), checkpoint=str(checkpoint),
                          predictions=str(predictions), train_seconds=train_seconds,
                          eval_seconds=eval_seconds, train_command=train_command,
                          eval_command=eval_command,
                          prediction_distribution=dict(distribution),
                          dominant_prediction_share=dominant_share,
                          collapsed=dominant_share >= manifest["collapse_threshold"])
            manifest["results"].append(result)
            persist()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if args.stop_on_collapse and result["collapsed"]:
                manifest.update(status="stopped_on_collapse", active_job=None,
                                stop_reason=f"{name}: dominant option share {dominant_share:.2%}")
                persist()
                print("STOPPED: PREDICTION COLLAPSE", flush=True)
                return
        manifest.update(status="complete", active_job=None)
        persist()
        print("ALL FIVE CONSTANT-LR EXPERIMENTS COMPLETE", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        persist()
        raise


if __name__ == "__main__":
    main()
