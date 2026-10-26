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


def generated_answer(text):
    answer = boxed_answer(text)
    if answer is not None:
        return answer
    match = re.search(r"####\s*([^\r\n]*)", text)
    return match.group(1).strip() if match else None


def reference_answer(solution):
    if "####" in solution:
        return solution.rsplit("####", 1)[1].strip()
    return boxed_answer(solution) or solution.strip()


def as_fraction(text):
    text = text.strip().replace("$", "")
    text = re.sub(r"^(\d+)\^\{?\\(?:text|mathrm)\{(?:st|nd|rd|th)\}\}?$", r"\1", text)
    if "," in text:
        # A comma-separated answer list is not a single integer.
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
            return None
        text = text.replace(",", "")
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


def parse_math_answer(answer):
    from math_verify import LatexExtractionConfig, parse

    # Like verl's Math-Verify scorer, retain a LaTeX anchor so fractions,
    # matrices and intervals are parsed as a whole instead of isolated numbers.
    return parse("\\boxed{" + answer + "}", (LatexExtractionConfig(),))


def choice_label(answer, allow_description=False):
    answer = answer.strip()
    wrapper = re.fullmatch(r"\\text\{(.*)\}", answer, flags=re.DOTALL)
    if wrapper:
        answer = wrapper[1].strip()
    suffix = r"(?:\s+[A-Za-z][A-Za-z\s-]*)?" if allow_description else ""
    match = re.fullmatch(r"\(([A-Z])\)" + suffix, answer)
    return match[1] if match else None


def grade(text, solution, finish_reason="stop"):
    answer = generated_answer(text)
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
    elif choice_label(gold) and choice_label(answer, allow_description=True):
        correct = choice_label(gold) == choice_label(answer, allow_description=True)
    else:
        try:
            from math_verify import parse, verify
        except ImportError as error:
            raise RuntimeError("Symbolic grading requires math-verify; install the generation extra") from error
        gold_parsed = parse_math_answer(gold)
        # Keep the GSM8K #### marker's existing plain-expression extraction.
        predicted_parsed = (parse_math_answer(answer) if boxed_answer(text) is not None
                            else parse(answer))
        if not gold_parsed or not predicted_parsed:
            return {"answer": answer, "correct": None, "status": "unparseable"}
        correct = bool(verify(gold_parsed, predicted_parsed))
        if not correct and (answer.endswith("%") or gold.endswith("%")):
            # MATH references sometimes omit the requested percent unit.
            # Try the numeric percentage as well, as verl's math_reward does.
            predicted_percent = as_fraction(answer.removesuffix(r"\%").removesuffix("%"))
            gold_percent = as_fraction(gold.removesuffix(r"\%").removesuffix("%"))
            correct = (predicted_percent is not None and gold_percent is not None
                       and predicted_percent == gold_percent)
    return {"answer": answer, "correct": correct, "status": "valid"}


def grade_probe(text, solution, finish_reason="stop"):
    result = grade(text, solution, finish_reason)
    if result["correct"] is not None:
        return result
    if finish_reason != "stop":
        return result
    try:
        from math_verify import parse, verify
    except ImportError as error:
        raise RuntimeError("Probe grading requires math-verify; install the generation extra") from error
    predicted, gold = parse(text), parse_math_answer(reference_answer(solution))
    if not predicted or not gold:
        return result
    return {"answer": str(predicted[0]), "correct": bool(verify(gold, predicted)), "status": "valid"}


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
    if not samples:
        return "uncertain"
    if all("p_correct" in sample for sample in samples):
        rate = sum(sample["p_correct"] for sample in samples) / len(samples)
    else:
        if any(sample.get("correct") is None for sample in samples):
            return "uncertain"
        rate = sum(sample["correct"] for sample in samples) / len(samples)
    if correct_threshold == wrong_threshold and rate == correct_threshold:
        return "uncertain"
    if rate >= correct_threshold:
        return "correct"
    if rate <= wrong_threshold:
        return "wrong"
    return "uncertain"


def localize(row, correct_threshold=0.8, wrong_threshold=0.2):
    """Find adjacent probe regression regardless of the trajectory's final grade."""
    if not 0 <= wrong_threshold <= correct_threshold <= 1:
        raise ValueError("Require 0 <= wrong threshold <= correct threshold <= 1")
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
    elif row.get("repair_success_criterion") == "probe_and_continuation":
        if row.get("repair_probe", {}).get("correct") is not True:
            return None
        successful = [sample for sample in row.get("continuations", [])
                      if sample.get("grade", {}).get("correct") is True]
        if not successful:
            return None
    else:
        successful = [sample for sample in row["continuations"] if sample["grade"]["correct"] is True]
        if not successful:
            return None
    cross_model = "repair_model" in row
    prefix = row["repair_prefix_ids"] if cross_model else row["trajectory_ids"][:row["candidate"]["start"]]
    replacement = row["replacement_ids"]
    prompt = row["repair_prompt_ids"] if cross_model else row["prompt_ids"]
    if target == "prefix_step":
        context = prompt
        completion = prefix + replacement
    elif target == "step_only":
        context = prompt + prefix
        completion = replacement
    elif target == "full":
        context = prompt
        completion = prefix + replacement + successful[0]["token_ids"]
    else:
        raise ValueError(f"Unknown target: {target}")
    return {
        "schema_version": 1, "id": row["id"], "split": "train", "target": target,
        "tokenizer_fingerprint": row.get("repair_tokenizer_fingerprint", row["tokenizer_fingerprint"]),
        "input_ids": context + completion,
        "labels": [-100] * len(context) + completion,
        "candidate": row["candidate"], "source_model": row["model"],
        **({"repair_model": row["repair_model"]} if cross_model else {}),
    }


def standard_sft_examples(rows, tokenizer, fingerprint, limit, excluded_ids=()):
    excluded_ids = set(excluded_ids)
    selected, seen_questions = [], set()
    for row in rows:
        if row["split"] != "train":
            raise ValueError(f"Refusing non-training sample {row['id']}")
        if row["id"] in excluded_ids:
            continue
        question = row["question"].strip()
        if question in seen_questions:
            raise ValueError(f"Duplicate training question {row['id']}")
        seen_questions.add(question)
        messages = [{"role": "user", "content": row["question"]}]
        context = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row["solution"]}],
            tokenize=True, add_generation_prompt=False,
        )
        if full[:len(context)] != context:
            raise ValueError(f"Chat template does not preserve prompt prefix for {row['id']}")
        completion = full[len(context):]
        if not completion:
            raise ValueError(f"Empty assistant target for {row['id']}")
        selected.append({
            "schema_version": 1, "id": row["id"], "split": "train",
            "target": "full_solution", "tokenizer_fingerprint": fingerprint,
            "input_ids": full, "labels": [-100] * len(context) + completion,
        })
        if len(selected) == limit:
            return selected
    raise ValueError(f"Found only {len(selected)} eligible training examples; need {limit}")


def correct_trajectory_sft_examples(rows, fingerprint, limit, excluded_ids=()):
    excluded_ids = set(excluded_ids)
    selected, seen_ids = [], set()
    for row in rows:
        if row["split"] != "train":
            raise ValueError(f"Refusing non-training sample {row['id']}")
        if row["tokenizer_fingerprint"] != fingerprint:
            raise ValueError(f"Tokenizer mismatch for {row['id']}")
        if row["id"] in excluded_ids or row["id"] in seen_ids:
            continue
        if row.get("trajectory_finish_reason") != "stop":
            continue
        if row.get("final_grade", {}).get("correct") is not True:
            continue
        prompt, trajectory = row["prompt_ids"], row["trajectory_ids"]
        if not trajectory:
            continue
        selected.append({
            "schema_version": 1, "id": row["id"], "split": "train",
            "target": "generated_correct_trajectory", "source_id": row["id"],
            "source_model": row["model"], "tokenizer_fingerprint": fingerprint,
            "input_ids": prompt + trajectory,
            "labels": [-100] * len(prompt) + trajectory,
        })
        seen_ids.add(row["id"])
        if len(selected) == limit:
            return selected
    raise ValueError(f"Found only {len(selected)} eligible correct generated trajectories; need {limit}")


def transition_sft_examples(row, correct_threshold=0.8, wrong_threshold=0.2,
                            target="prefix_recovery", allow_multiple=False,
                            consecutive_transitions=1, skip_uncertain=False,
                            min_target_tokens=0, max_target_tokens=None,
                            min_probe_gap=1, require_stable_correct=False,
                            require_correct_final_grade=False):
    """Build SFT targets from original-trajectory wrong-to-correct probe transitions."""
    if target not in {"prefix_recovery", "step_only"}:
        raise ValueError(f"Unknown transition SFT target: {target}")
    if row["split"] != "train":
        raise ValueError(f"Refusing non-training sample {row['id']}")
    probes = row["probes"]
    offsets = [probe["offset"] for probe in probes]
    if offsets != sorted(set(offsets)):
        raise ValueError("Probe offsets must be strictly increasing")
    states = [probe_state(probe, correct_threshold, wrong_threshold) for probe in probes]
    emitted = 0
    if consecutive_transitions < 1:
        raise ValueError("consecutive_transitions must be positive")
    if min_target_tokens < 0 or (max_target_tokens is not None and max_target_tokens < min_target_tokens):
        raise ValueError("Invalid target token limits")
    if min_probe_gap < 1:
        raise ValueError("min_probe_gap must be positive")
    if require_correct_final_grade and row.get("final_grade", {}).get("correct") is not True:
        return
    if skip_uncertain:
        decided = [(index, state) for index, state in enumerate(states)
                   if state in {"wrong", "correct"}]
        transitions = (
            (left[0], right[0]) for left, right in zip(decided, decided[1:])
            if left[1] == "wrong" and right[1] == "correct"
        )
    elif consecutive_transitions == 1:
        transitions = ((index - 1, index) for index in range(1, len(probes))
                       if states[index - 1:index + 1] == ["wrong", "correct"])
    else:
        # Ignore uncertain probes when checking for repeated W->C transitions.
        decided = [(index, state) for index, state in enumerate(states)
                   if state in {"wrong", "correct"}]
        pattern = [state for _ in range(consecutive_transitions)
                   for state in ("wrong", "correct")]
        transitions = (
            (decided[index][0], decided[index + 1][0])
            for index in range(len(decided) - len(pattern) + 1)
            if [state for _, state in decided[index:index + len(pattern)]] == pattern
        )
    for left_index, right_index in transitions:
        if require_stable_correct and any(
            state == "wrong" for state in states[right_index + 1:]
        ):
            continue
        if right_index - left_index < min_probe_gap:
            continue
        start, end = offsets[left_index], offsets[right_index]
        trajectory = row["trajectory_ids"]
        if not 0 <= start < end <= len(trajectory):
            raise ValueError(f"Probe offset outside trajectory for {row['id']}")
        target_tokens = end - start
        if target_tokens < min_target_tokens:
            continue
        if max_target_tokens is not None and target_tokens > max_target_tokens:
            continue
        context = row["prompt_ids"]
        completion = trajectory[:end]
        labels = ([-100] * (len(context) + start) + trajectory[start:end]
                  if target == "step_only" else [-100] * len(context) + completion)
        yield {
            "schema_version": 1, "id": f"{row['id']}:wrong-to-correct:{start}-{end}",
            "source_id": row["id"], "split": "train", "target": target,
            "tokenizer_fingerprint": row["tokenizer_fingerprint"],
            "input_ids": context + completion,
            "labels": labels,
            "candidate": {"start": start, "end": end,
                          "left_probe": left_index, "right_probe": right_index},
            "source_model": row["model"],
        }
        emitted += 1
        if not allow_multiple and emitted == 1:
            return


def regression_reconstruction_candidate(row, correct_threshold=0.6, wrong_threshold=0.4):
    """Return the first C-to-W transition while skipping uncertain probes."""
    if not 0 <= wrong_threshold <= correct_threshold <= 1:
        raise ValueError("Require 0 <= wrong threshold <= correct threshold <= 1")
    if row["split"] != "train":
        raise ValueError(f"Refusing non-training sample {row['id']}")
    probes = row["probes"]
    offsets = [probe["offset"] for probe in probes]
    if offsets != sorted(set(offsets)):
        raise ValueError("Probe offsets must be strictly increasing")
    decided = [(index, probe_state(probe, correct_threshold, wrong_threshold))
               for index, probe in enumerate(probes)]
    decided = [(index, state) for index, state in decided if state in {"correct", "wrong"}]
    for (left, left_state), (right, right_state) in zip(decided, decided[1:]):
        if left_state == "correct" and right_state == "wrong":
            start, end = offsets[left], offsets[right]
            if not 0 <= start < end <= len(row["trajectory_ids"]):
                raise ValueError(f"Probe offset outside trajectory for {row['id']}")
            return {"start": start, "end": end, "left_probe": left,
                    "right_probe": right}
    return None
