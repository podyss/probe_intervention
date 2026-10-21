# Bundled datasets

`gsm8k-500/` and `gsm8k-full/` contain the GSM8K parquet files already available
in the parent OPSD workspace. The complete small dataset is included here so
this standalone project can be copied to a separate GitHub repository without
an implicit dependency on parent-directory files.

- `gsm8k-500/{train,test}.parquet`: 500-row train/test subset used by local experiments.
- `gsm8k-full/{train,test}.parquet`: complete GSM8K train/test splits.
- Schema: `question`/`problem`, `answer`/`solution`, as emitted by the source parquet.
- These are source data, not generated project artifacts. Verify the original
  dataset license and redistribution terms before publishing them publicly.

Use `probe-int prepare --input data/gsm8k-500/train.parquet --split train ...`
from the `probe_intervention/` directory for a quick run.
