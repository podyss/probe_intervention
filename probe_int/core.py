import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            count += 1
    return count


def boxed_answer(text):
    text = text.rsplit("</think>", 1)[-1]
    answers = []
    for match in re.finditer(r"\\boxed\{", text):
        depth = 1
        for position in range(match.end(), len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                depth -= 1
            if depth == 0:
                answers.append(text[match.end():position].strip())
                break
    return answers[-1] if answers else None


def reference_answer(solution):
    if "####" in solution:
        return solution.rsplit("####", 1)[1].strip()
    return boxed_answer(solution) or solution.strip()


def as_fraction(text):
    text = text.strip().replace(",", "").replace("$", "")
    latex = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if latex:
        text = f"{latex[1]}/{latex[2]}"
    try:
        if "/" in text:
            numerator, denominator = text.split("/")
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(text))
    except (ValueError, InvalidOperation, ZeroDivisionError, OverflowError):
        return None


def grade(text, solution, finish_reason="stop"):
    answer = boxed_answer(text)
    if finish_reason != "stop":
        return {"answer": answer, "correct": None, "status": "incomplete"}
    if not answer:
        return {"answer": None, "correct": None, "status": "unparseable"}
    gold = reference_answer(solution)
    predicted_number, gold_number = as_fraction(answer), as_fraction(gold)
    if predicted_number is not None and gold_number is not None:
        correct = predicted_number == gold_number
    elif re.sub(r"\s+", "", answer) == re.sub(r"\s+", "", gold):
        correct = True
    else:
        try:
            from math_verify import parse, verify
        except ImportError as error:
            raise RuntimeError("Symbolic grading requires math-verify; install the generation extra") from error
        gold_parsed, predicted_parsed = parse(gold), parse(answer)
        if not gold_parsed or not predicted_parsed:
            return {"answer": answer, "correct": None, "status": "unparseable"}
        correct = bool(verify(gold_parsed, predicted_parsed))
    return {"answer": answer, "correct": correct, "status": "valid"}


def tokenizer_fingerprint(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template,
               "special_tokens": tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reasoning_boundaries(token_ids, tokenizer, mode, interval, delimiter="\n\n"):
    if interval < 1 or not delimiter:
        raise ValueError("interval must be positive and delimiter must be nonempty")
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    reasoning_end = len(token_ids)
    for position in range(len(token_ids)):
        if token_ids[position:position + len(end_ids)] == end_ids:
            reasoning_end = position
            break
    if mode == "token":
        boundaries = list(range(interval, reasoning_end, interval))
    elif mode == "step":
        boundaries = []
        last_delimiter = -len(delimiter)
        for position in range(reasoning_end):
            piece = tokenizer.decode(token_ids[position:position + 1], skip_special_tokens=False)
            if any(character in piece for character in delimiter):
                prefix = tokenizer.decode(token_ids[:position + 1], skip_special_tokens=False)
                delimiter_position = prefix.rfind(delimiter)
                if delimiter_position >= 0 and delimiter_position > last_delimiter:
                    content = prefix[last_delimiter + len(delimiter):delimiter_position]
                    if content.replace("<think>", "").strip():
                        boundaries.append(position + 1)
                    last_delimiter = delimiter_position
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")
    return sorted(set([0] + boundaries + ([reasoning_end] if reasoning_end else [])))


def probe_state(probe, correct_threshold, wrong_threshold):
    samples = probe["samples"]
    if not samples or any(sample["correct"] is None for sample in samples):
        return "uncertain"
    rate = sum(sample["correct"] for sample in samples) / len(samples)
    if rate >= correct_threshold:
        return "correct"
    if rate <= wrong_threshold:
        return "wrong"
    return "uncertain"


def localize(row, correct_threshold=0.8, wrong_threshold=0.2):
    """Find adjacent probe regression regardless of the trajectory's final grade."""
    if not 0 <= wrong_threshold < correct_threshold <= 1:
        raise ValueError("Require 0 <= wrong threshold < correct threshold <= 1")
    result = dict(row, candidate=None)
    probes = row["probes"]
    offsets = [probe["offset"] for probe in probes]
    if offsets != sorted(set(offsets)):
        raise ValueError("Probe offsets must be strictly increasing")
    states = [probe_state(probe, correct_threshold, wrong_threshold) for probe in probes]
    for index in range(1, len(probes)):
        if states[index - 1:index + 1] == ["correct", "wrong"]:
            result["candidate"] = {
                "start": offsets[index - 1], "end": offsets[index],
                "left_probe": index - 1, "right_probe": index,
                "meaning": "first_observed_adjacent_answer_regression",
            }
            result["localization_status"] = "candidate"
            return result
    result["localization_status"] = "no_adjacent_regression"
    return result


def repair_rejection(text, solution, allow_answer=False):
    if not text.strip():
        return "empty_step"
    if any(marker in text for marker in ["<think>", "</think>", "<|", "<reference", "<intervention"]):
        return "unexpected_tags"
    if "\n\n" in text.strip():
        return "multiple_steps"
    if not allow_answer:
        if boxed_answer(text) is not None or "####" in text:
            return "final_answer_in_step"
        gold = reference_answer(solution)
        if gold and re.search(r"(?<![\w.])" + re.escape(gold) + r"(?!\w|\.\d)", text):
            return "possible_answer_leakage"
    return None


def sft_example(row, target="prefix_step"):
    if row["split"] != "train":
        raise ValueError(f"Refusing non-training sample {row['id']}")
    if row.get("repair_status") != "accepted":
        return None
    if row.get("repair_success_criterion") == "probe":
        if row.get("repair_probe", {}).get("correct") is not True:
            return None
        successful = []
    else:
        successful = [sample for sample in row["continuations"] if sample["grade"]["correct"] is True]
        if not successful:
            return None
    prefix = row["trajectory_ids"][:row["candidate"]["start"]]
    replacement = row["replacement_ids"]
    if target == "prefix_step":
        context = row["prompt_ids"]
        completion = prefix + replacement
    elif target == "step_only":
        context = row["prompt_ids"] + prefix
        completion = replacement
    elif target == "full":
        context = row["prompt_ids"]
        completion = prefix + replacement + successful[0]["token_ids"]
    else:
        raise ValueError(f"Unknown target: {target}")
    return {
        "schema_version": 1, "id": row["id"], "split": "train", "target": target,
        "tokenizer_fingerprint": row["tokenizer_fingerprint"],
        "input_ids": context + completion,
        "labels": [-100] * len(context) + completion,
        "candidate": row["candidate"], "source_model": row["model"],
    }


def transition_sft_examples(row, correct_threshold=0.8, wrong_threshold=0.2):
    """Build SFT targets from original-trajectory wrong-to-correct probe transitions."""
    if row["split"] != "train":
        raise ValueError(f"Refusing non-training sample {row['id']}")
    probes = row["probes"]
    offsets = [probe["offset"] for probe in probes]
    if offsets != sorted(set(offsets)):
        raise ValueError("Probe offsets must be strictly increasing")
    states = [probe_state(probe, correct_threshold, wrong_threshold) for probe in probes]
    for index in range(1, len(probes)):
        if states[index - 1:index + 1] != ["wrong", "correct"]:
            continue
        start, end = offsets[index - 1], offsets[index]
        trajectory = row["trajectory_ids"]
        if start < 0 or end > len(trajectory):
            raise ValueError(f"Probe offset outside trajectory for {row['id']}")
        context = row["prompt_ids"]
        completion = trajectory[:end]
        yield {
            "schema_version": 1, "id": f"{row['id']}:wrong-to-correct:{start}-{end}",
            "source_id": row["id"], "split": "train", "target": "prefix_recovery",
            "tokenizer_fingerprint": row["tokenizer_fingerprint"],
            "input_ids": context + completion,
            "labels": [-100] * len(context) + completion,
            "candidate": {"start": start, "end": end,
                          "left_probe": index - 1, "right_probe": index},
            "source_model": row["model"],
        }
