"""Independently verify each labeled prefix without exposing its paired label."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .core import read_jsonl, write_jsonl


AUDIT_PROMPT = """Check only the mathematical reasoning prefix shown below.
It may be incomplete. A missing final answer is not an error. Do not infer
unshown future reasoning. Judge definite mathematical/logical claims, not
whether the attempt agrees with the reference answer yet. An error explicitly
corrected WITHIN THIS PREFIX is resolved. An error corrected outside the
prefix must still count here. Mere tentative questions are not definite claims.

Return only one JSON object with these keys. Do not include analysis or Markdown.
"verdict": "error" if this prefix contains a definite unresolved error,
           "clean" if it contains no definite unresolved error,
           "uncertain" if you cannot verify it confidently;
"error_quote": for "error", an exact verbatim substring containing the first
               definite unresolved error; otherwise an empty string;
"reason": one sentence of at most 30 words explaining the verdict.

Problem:
{question}

Reference answer (consistency check only):
{answer}

Reasoning prefix:
{prefix}
"""


def parse_audit(text, prefix):
    # Permit a Markdown wrapper, but do not accept incomplete or invented evidence.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    verdict = value.get("verdict")
    quote, reason = value.get("error_quote"), value.get("reason")
    if verdict not in {"error", "clean", "uncertain"}:
        return None
    if not isinstance(reason, str) or not reason.strip() or not isinstance(quote, str):
        return None
    if verdict == "error" and (not quote.strip() or quote not in prefix):
        return None
    if verdict == "clean" and quote.strip():
        return None
    return {"verdict": verdict, "error_quote": quote, "reason": reason}


def verified_pairs(rows, audits):
    if len(rows) != len(audits):
        raise ValueError("Every input prefix must have an audit result")
    groups = defaultdict(list)
    for row, audit in zip(rows, audits):
        groups[row["pair_id"]].append((row, audit))
    accepted = []
    for pair_id, pair in groups.items():
        if len(pair) != 2 or Counter(row["label"].lower() for row, _ in pair) != {"yes": 1, "no": 1}:
            raise ValueError(f"Expected one Yes and one No for {pair_id}")
        expected = {"yes": "error", "no": "clean"}
        if all(audit is not None and audit["verdict"] == expected[row["label"].lower()]
               for row, audit in pair):
            accepted.extend(dict(row, prefix_audit=audit) for row, audit in pair)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    audit_output = output.with_suffix(output.suffix + ".audit.jsonl")
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    for path in (output, audit_output, manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError("Input contains no pairs")

    from .backend import VLLMBackend

    backend = VLLMBackend(args)
    prompts = [backend.chat(AUDIT_PROMPT.format(
        question=row["question"], answer=row["reference_solution"],
        prefix=row["reasoning_prefix"],
    ), thinking=False) for row in rows]
    generations = backend.generate(prompts, args.max_tokens, temperature=0)
    audits, audit_rows = [], []
    for row, samples in zip(rows, generations):
        sample = samples[0]
        parsed = (parse_audit(sample["text"], row["reasoning_prefix"])
                  if sample["finish_reason"] == "stop" else None)
        audits.append(parsed)
        audit_rows.append({
            "id": row["id"], "pair_id": row["pair_id"], "original_label": row["label"],
            "audit": parsed, "teacher_response": sample["text"],
            "finish_reason": sample["finish_reason"],
        })
    accepted = verified_pairs(rows, audits)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, accepted)
    write_jsonl(audit_output, audit_rows)
    expected = {"yes": "error", "no": "clean"}
    summary = {
        "arguments": vars(args),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "input_pairs": len(rows) // 2,
        "accepted_pairs": len(accepted) // 2,
        "rejected_pairs": (len(rows) - len(accepted)) // 2,
        "unparsed_or_truncated": sum(x is None for x in audits),
        "label_disagreements": sum(x is not None and x["verdict"] != expected[row["label"].lower()]
                                   for row, x in zip(rows, audits)),
        "status": "complete",
    }
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
