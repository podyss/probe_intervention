# Bundled datasets

`gsm8k-full/` contains the complete GSM8K train/test parquet files from
the parent OPSD workspace. The dataset is included here so
this standalone project can be copied to a separate GitHub repository without
an implicit dependency on parent-directory files.

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
