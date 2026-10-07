# Stage 1 labeler

Run everything from the project root with the `knee` venv active and Ollama running.

```powershell
cd "D:\omer files\RSNA Knee Abnormality Detection - Kaggle"
$env:PYTHONUTF8 = "1"

# 1. Split (uses study_meta_chunk*.csv for patient grouping if present)
python -m src.labeler.split

# 2. Dev loop: run, read errors, edit prompt.py (bump PROMPT_VERSION), repeat
python -m src.labeler.run --subset dev
python -m src.labeler.evaluate data\labels\raw_v3_dev.jsonl --errors --save data\labels\eval_v3_dev.txt

# 3. Once definitions are frozen: score the held-out 38 (summary only)
python -m src.labeler.run --subset heldout
python -m src.labeler.evaluate data\labels\raw_v3_heldout.jsonl

# 4. Overnight full run in its own window (resumable; just rerun if it stops)
python -m src.labeler.run --subset all
python -m src.labeler.build_labels data\labels\raw_v3_all.jsonl
```

### Running 2 reports in parallel (faster full run)

Ollama handles one request at a time unless told otherwise. To allow 2:
```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "2", "User")
# then quit Ollama from the tray icon and start it again
python -m src.labeler.run --subset dev --workers 2 --tag w2
ollama ps   # PROCESSOR must still say 100% GPU; if it shows CPU, go back to 1
```
Results are identical either way; only the speed changes.

| File | What it does |
|---|---|
| `split.py` | Dev 20 / held-out 38, seeded search for the most balanced split, ≥2 positives per label per side |
| `prompt.py` | System prompt, label definitions, JSON schema passed to Ollama `format`, `PROMPT_VERSION` |
| `run.py` | Ollama runner: `think: false`, temp 0, `num_ctx` 8192, schema-constrained output, 3 retries, resumable JSONL + log |
| `evaluate.py` | Per-label acc / sens / spec / κ / soft AUC vs the human labels; `--errors` lists disagreements (dev only) |
| `build_labels.py` | `soft_labels.parquet`: 58 human labels + LLM 1 / 0.5 / 0, with a `source` column |

Outputs go to `data\labels\` as `raw_<version>_<subset>.jsonl` and `run_<version>_<subset>.log`.
