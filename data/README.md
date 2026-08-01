# Data

The dataset contract, the full audit, and the privacy terms are in
[`docs/DATA.md`](../docs/DATA.md).

This directory holds `fawkes-training-graph-embedded-260615/`: 4,000 embedded
admission graphs, 234 MB. It is **not committed** — `.gitignore` excludes
`data/**/*.jsonl` under the PhysioNet data use agreement — and
`_download_manifest.json` records where it came from.

Run:

```bash
python scripts/audit_data.py
```

before selecting a file for a model. With no `--path` it audits the dataset
above; pass `--path YOUR.jsonl` for another.
