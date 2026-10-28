# Bundled datasets

## AMC 2023

`amc23/` contains the 40-question test split from
[`zwhe99/amc23`](https://hf-mirror.com/datasets/zwhe99/amc23), pinned at
revision `f9810c0439cd3c670ec885d328a2f06a87f3694a`.

- `amc23/source/test-00000-of-00001.parquet`: the original Hugging Face file.
- `amc23/test.jsonl`: normalized rows ready for `probe-int-eval`.
- `amc23/prepare.py`: reproducible conversion from the source Parquet.
- `amc23/manifest.json`: source revision, checksum, field mapping, and row count.

The normalized schema is `id`, `split`, `question`, `solution`, and `url`;
the source's numeric `answer` is converted to a string in `solution`. Run an
evaluation directly with:

```bash
probe-int-eval --model "$MODEL" --input data/amc23/test.jsonl \
  --output outputs/amc23/eval.jsonl --temperature 0
```

The upstream dataset card does not declare a license. Check the source and
underlying AMC/AoPS terms before redistributing these files.

`gsm8k-full/` contains the complete GSM8K train/test parquet files from
the parent OPSD workspace. The dataset is included here so
this standalone project can be copied to a separate GitHub repository without
an implicit dependency on parent-directory files.

`aime-2024/` contains the HuggingFaceH4 AIME 2024 dataset and a prepared
evaluation split:

- `aime-2024/train.parquet`: the source dataset file from
  `HuggingFaceH4/aime_2024` (30 problems, source split `train`).
- `aime-2024/test.jsonl`: normalized evaluation rows accepted by
  `probe-int-eval` (30 problems, labeled `test` for this repository's
  evaluator).
- Source columns `problem`, `answer`, and `id` map to `question`, `solution`,
  and `id`. When both `answer` and `solution` exist, `answer` is used as the
  authoritative grading reference; detailed worked solutions are not used as
  the gold answer.

Run evaluation from the repository root with:

```bash
probe-int-eval --model "$MODEL" \
  --input data/aime-2024/test.jsonl \
  --output outputs/aime-2024/eval.jsonl \
  --temperature 0
```

The source dataset card describes these as the 30 problems from AIME I and
AIME II 2024 and identifies the original source as `AI-MO/aimo-validation-aime`.
Check the upstream dataset card and license before redistribution.

- `gsm8k-full/{train,test}.parquet`: complete GSM8K train/test splits.
- Schema: `question`/`problem`, `answer`/`solution`, as emitted by the source parquet.
- These are source data, not generated project artifacts. Verify the original
  dataset license and redistribution terms before publishing them publicly.

For a quick run from the `probe_intervention/` directory, prepare a small subset
directly from the full dataset:

```bash
probe-int prepare --input data/gsm8k-full/train.parquet --split train \
  --limit 8 --output outputs/train.jsonl
probe-int prepare --input data/gsm8k-full/test.parquet --split test \
  --limit 8 --output outputs/test.jsonl
```

Remove `--limit` to prepare the complete split.

## Teacher First-Error Step Labels

`teacher-first-error-short-200/` contains the short, externally audited
first-error examples used for the R1 1.5B Yes/No LoRA experiment.

- `all.jsonl`: all 208 labeled prefixes (104 Yes, 104 No).
- `train.jsonl`: 168 examples (84 Yes, 84 No).
- `test.jsonl`: 40 examples (20 Yes, 20 No), held out by source problem.
- `split-manifest.json`: seed and source IDs assigned to each split.
- `manifest.json`: provenance, schema, counts, and the matching experiment.

Each row includes the source problem, reference solution, teacher's first-error
location and rationale, the reasoning prefix, and a `label` of `Yes` or `No`.
The training and evaluation scripts normalize label case. Yes/No examples for
one source problem are kept in the same split to avoid pair leakage. The
annotations came from the external first-error audit stored under
`outputs/teacher-first-error-short-200/` in the working experiment; inspect
those audit records before redistributing the annotations.

The experiment trained on `train.jsonl` and evaluated on `test.jsonl` with
`probe_int/train_step_labels.py` and `probe_int/evaluate_step_labels.py`.

## DeepMath-103K

`deepmath-103k/` contains the complete DeepMath-103K dataset from
[`zwhe99/DeepMath-103K`](https://hf-mirror.com/datasets/zwhe99/DeepMath-103K)
at revision `5cf055d1fe3d7a2eb19719ac020211469736ae44` (MIT license).

- `deepmath-103k/source/data/`: all 10 original Parquet shards (103,022 rows).
- `deepmath-103k/all.jsonl`: complete normalized dataset.
- `deepmath-103k/train.jsonl`: 1,000 training problems.
- `deepmath-103k/test.jsonl`: 500 test problems.
- `deepmath-103k/manifest.json`: source revision, field mapping, sample seed, and selected IDs.
- `deepmath-103k/prepare.py`: reproducible conversion and sampling script.

The JSONL format uses `id`, `question`, and `solution`; `solution` maps from
the source `final_answer` field. `difficulty`, `topic`, and `source_split` are
retained. The train and test sets are disjoint uniform samples (seed 42) from
rows with nonempty questions and answers. Two source rows have empty answers;
they remain in the complete files but are excluded from both samples.

Use the prepared subsets directly with the project commands, for example:

```bash
probe-int collect --model "$MODEL" \
  --input data/deepmath-103k/train.jsonl \
  --output outputs/deepmath-103k/probes.jsonl
probe-int-eval --model "$MODEL" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/deepmath-103k/eval.jsonl
```

Rebuild the normalized files from the downloaded source shards with:

```bash
uv run python data/deepmath-103k/prepare.py
```
