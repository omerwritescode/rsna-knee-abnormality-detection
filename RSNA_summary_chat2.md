# RSNA Knee Abnormality Detection — Progress Log & Handoff

*Last updated: Wed 7 Oct 2026, 14:30 IST · Covers the sessions on 6–7 Oct 2026 · Work done by Omer*

This file records everything done so far, every decision and finding, how the setup looks on the laptop and on Kaggle, and what is left. It goes **alongside** `RSNA_Knee_Game_Plan.md`. The game plan stays the reference for strategy, and this file overrides it wherever they disagree (section 9 lists every deviation).

**To continue in a new chat:** say "read `RSNA_Knee_Progress.md` and the game plan, then start Stage 1 (the labeler)."

---

## 1. Status at a glance

| Stage | Status | Notes |
|---|---|---|
| Laptop environment | ✅ Done | Native Windows (not WSL2). PyTorch cu128 + CUDA on the RTX 5050 · Ollama + `qwen3.5:9b` at 100% GPU · Kaggle API |
| DICOM inspection on Kaggle | ✅ Done | Every geometry tag present, all sampled files uncompressed, decoding is fast |
| `kneeprep` (shared preprocessing) | ✅ Done and tested | Synthetic tests pass; checked on real data with the orientation-check image |
| Stage 2: DICOM → shards | ✅ **Done** | **4,407 / 4,407** studies with 0 errors, extracted to `data\shards\` (~23.4 GB) |
| Stage 1: report → labels (LLM) | ⏭️ **Next** | Nothing written yet |
| Slice embeddings (DINOv2-S) | ⬜ Not started | |
| M1: frozen DINOv2-S + aggregator | ⬜ Not started | |
| Submission notebook + first LB score | ⬜ Not started | Target: by **11 Oct** |
| M2: fine-tuned DINOv2-S | ⬜ Not started | |
| Team merge (Rishi, Arfeen) | ⬜ **Deadline 15 Oct 23:59 UTC** | Omer's account has accepted the rules (`userHasEntered: True`) |
| Ensemble + final 2 submissions | ⬜ Deadline 22 Oct 23:59 UTC | |

**We are about 2 days behind the original game-plan timeline.** The revised timeline is in section 8.

---

## 2. Key facts and identifiers

| Item | Value |
|---|---|
| Competition slug | `rsna-knee-abnormality-detection` |
| Competition data path on Kaggle | `/kaggle/input/competitions/rsna-knee-abnormality-detection/` |
| Kaggle username | `omerraza2202` |
| Deadlines (23:59 UTC) | 15 Oct entry + team merger · 22 Oct final submission · 5 Nov winners' code/video/write-up |
| Teams on LB (7 Oct) | 5,324 · prize pool $77,000 |
| Submission limits | Notebook, internet off, ≤ 9 h, writes `submission.csv` |

> ⚠️ An old personal dataset `omerraza2202/train-series-csv` contains **only** `train_series.csv`, without images. Don't attach it to notebooks; it once caused the data-path search to pick the wrong folder. All notebooks now search for the folder that has both `train_series.csv` **and** the `train_series/` image folder.

---

## 3. Laptop environment (as set up)

**Hardware:** i7-13620H · RTX 5050 8 GB (Blackwell) · 24 GB RAM · Windows 11.

**Decision: native Windows instead of WSL2.** This saves setup time, and everything in the plan runs on Windows. Switch to WSL2 later only if M2 training runs into DataLoader-worker or library problems. All code is written with `pathlib` paths and `if __name__ == "__main__":` guards so it works on both.

| Component | Detail |
|---|---|
| Project folder | `D:\omer files\RSNA Knee Abnormality Detection - Kaggle` (**has spaces, so always quote it in PowerShell**) |
| Shell | **PowerShell**, not Command Prompt (cmd opens `.ps1` files in Notepad) |
| Python | 3.12 via winget; venv named **`knee`**, activated with `<venv path>\Scripts\Activate.ps1` (instructions used `C:\venvs\knee`). The prompt shows `(knee)` when active |
| Packages | torch + torchvision (cu128 wheels), timm, transformers, peft, pydicom, pylibjpeg (+ libjpeg, openjpeg), python-gdcm, numpy, pandas, pyarrow, scikit-learn, iterative-stratification, opencv-python-headless, tqdm, kaggle, requests |
| CUDA check | `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `+cu128 True NVIDIA GeForce RTX 5050…` |
| Ollama | Windows tray app, serves on `localhost:11434`. Model **`qwen3.5:9b`** (Q4_K_M, ~6.6 GB download, 5.6 GB loaded) |
| Ollama speed (measured) | **100% GPU · ~52 tokens/s generation · ~77 tokens/s prompt eval** · ~10 s cold load · unloads after 5 idle minutes |
| Kaggle API | Key at `<project>\.kaggle\kaggle.json` · env var `KAGGLE_CONFIG_DIR` set permanently at user level to that folder · `.kaggle/` is in `.gitignore` |
| UTF-8 | `PYTHONUTF8=1` set (permanently at user level). Needed for the multilingual reports, and fixes the Kaggle CLI `'charmap' codec` error |
| Git | `git init` done in the project folder. **No commits yet; commit soon** |
| Power | Sleep/hibernate on AC disabled (`powercfg`). Before overnight runs: pause Windows Update, plug in, Best performance mode, cooling pad, `nvidia-smi -l 5` to watch thermals |

### Folder layout (actual)

```
D:\omer files\RSNA Knee Abnormality Detection - Kaggle\
├── .git\  .kaggle\ (kaggle.json)  .gitignore
├── RSNA_Knee_Game_Plan.md, RSNA_Knee_Abnormality_Detection.md, dataset-info.png
├── configs\
├── data\
│   ├── raw\          train.csv, train_series.csv, test.csv, test_series.csv, sample_submission.csv
│   │                 study_meta_chunk0.csv, study_meta_chunk1.csv      ← per-study metadata from Stage 2
│   │                 series_meta_chunk0.csv, series_meta_chunk1.csv    ← per-series metadata from Stage 2
│   ├── labels\       (empty: Stage 1 output goes here)
│   ├── shards\       4,407 × <StudyInstanceUID>.npz   (~23.4 GB)
│   ├── embeddings\   (empty: DINOv2 features go here)
│   └── download\     12 × shards_c{0,1}_p0{0-5}.tar  (duplicate of shards; kept as a local backup)
├── kaggle\           kaggle_dicom_inspect.py / .ipynb, prep_notebook.ipynb
├── scripts\          build_prep_notebook.py
├── src\
│   ├── kneeprep\     __init__.py, core.py
│   ├── labeler\  models\  utils\   (empty)
└── tests\            test_kneeprep_geometry.py
```

**Still to add to `.gitignore`** (if not done): `data/raw/`, `data/shards/`, `data/embeddings/`, `data/download/`, `__pycache__/`.

---

## 4. DICOM inspection results (Kaggle, 40 random training studies, 219 series, 657 headers)

Notebook: `kaggle_dicom_inspect.ipynb`. Outputs saved to `/kaggle/working`: `series_sample.csv`, `tag_presence.csv`, `decode_timing.csv`, `estimate.json`, `wheels/`.

| Finding | Result | Consequence |
|---|---|---|
| Distinct tags seen | 83 (of the 86 allowlisted) | — |
| `ImagePositionPatient`, `ImageOrientationPatient`, `InstanceNumber`, `PixelSpacing`, `SliceThickness` | **100%** | Sort slices by true 3D position; put every image in a standard orientation |
| `SpacingBetweenSlices` | 94% | — |
| `Laterality` | **44% of series** (52 L, 44 R, 123 missing) | Can't normalize left/right from tags alone; see laterality plan in section 7 |
| `RescaleSlope/Intercept` | 45% | Applied when present |
| `PatientID` | 100% | Recorded per study; group CV folds by patient if any patient has several studies |
| Plane from orientation tags vs CSV `Anatomical_Plane` | **100% agreement** | The CSV plane can be trusted for series selection |
| Transfer syntax | **All 6,713 sampled files uncompressed** (Explicit VR Little Endian) | Decoding is not a bottleneck. JPEG/J2K may still exist in other studies or the test set, so decoders stay installed |
| Decode cost | ~1.2–1.6 ms decode + 1.5 ms read per file; gdcm fastest, all 3 backends worked | — |
| Slices per series | mean 30.7, median 30, min 11, max 144 | Resample to 32 |
| Matrix sizes | 256² to 1024² (512², 640², 256², 384², 320² most common) | Resize to 256 |
| Frames per file | 1 wherever present | Single-frame only |
| Photometric | 100% MONOCHROME2 | MONOCHROME1 still handled defensively |
| Manufacturers | GE, Siemens/Siemens Healthineers, Philips, Toshiba, Canon | Multi-vendor, so standard orientation and intensity scaling matter |
| Field strength | 1.5 T (137), 3 T (71), missing (11) | — |
| Multi-echo / stacks | `EchoNumbers` 85%, `StackID` 21%, `TemporalPositionIndex` 6% | Some series have several images per position, so `kneeprep` drops duplicate positions |
| Offline wheels | `pip download` of pydicom, pylibjpeg (+plugins), python-gdcm into `/kaggle/working/wheels` (Kaggle's Python 3.13) | **TODO:** check that the inspection notebook was saved as a version with this output; it's needed for the internet-off submission notebook |

---

## 5. `kneeprep`: the shared preprocessing module

**Source of truth:** `src/kneeprep/core.py`. The Kaggle notebooks embed it with a `%%writefile kneeprep.py` cell generated by `scripts/build_prep_notebook.py`. **Edit only `core.py`, then run `python scripts\build_prep_notebook.py` to rebuild.** The submission notebook must embed the same file, so train and test get identical preprocessing.

### Series selection (up to 4 per study, the "kinds")

| Kind | Plane | `Fluid_Sensitive` | Preference |
|---|---|---|---|
| `sag_fl` | Sagittal | 1 | `Fat_Suppression == 1` first, then most slices |
| `cor_fl` | Coronal | 1 | same |
| `ax_fl` | Axial | 1 | same |
| `sag_t1` | Sagittal | 0 | `Fat_Suppression == 0` first, then most slices |

A missing kind is simply left out of the study's file. Coronal non-FS and axial non-FS are not kept.

### Per-series pipeline
1. Read every file with `pydicom.dcmread(force=True)`; apply rescale; invert MONOCHROME1; drop slices whose shape differs from the series' most common shape.
2. **Sort by position** along the slice normal (from IPP + IOP), falling back to `InstanceNumber` when geometry is missing. **Drop duplicate positions** (within 0.01 mm), keeping the lowest echo.
3. **Standard orientation** using LPS patient coordinates (+x = patient left, +y = posterior, +z = superior):

| Plane | Slice order | Image up | Image right |
|---|---|---|---|
| Sagittal | patient right → left | superior | posterior (front of knee on the **left**) |
| Coronal | anterior → posterior | superior | patient left |
| Axial | superior → inferior | anterior | patient left |

   This is done by transposing and/or flipping each image as needed; oblique scans are handled. **Laterality is NOT normalized**: a left knee stays a left knee.
4. **Resize keeping the physical aspect ratio** (uses `PixelSpacing`) and centre-pad to **256×256**. INTER_AREA when shrinking.
5. **Resample to 32 slices:** block-average when there are ≥ 64 source slices, then linear interpolation.
6. **Intensity:** clip to the 0.5–99.5 percentiles of the image region (padding excluded) and scale to uint8. Padding stays 0.

### Output format
- One file per study: `data\shards\<StudyInstanceUID>.npz` (compressed), with keys from `sag_fl`, `cor_fl`, `ax_fl`, `sag_t1`, each **uint8 `[32, 256, 256]`**.
- Load with `kneeprep.load_study(path)` → `{kind: array}`.
- `study_meta_chunk*.csv` columns: `StudyInstanceUID, PatientID, laterality (L/R/blank, strict majority vote over all series), lat_votes_L, lat_votes_R, kinds, n_series_csv, n_series_errors, npz_bytes, error, seconds`.
- `series_meta_chunk*.csv` columns: `StudyInstanceUID, kind, SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive, Fat_Suppression, error, plane_iop, sorted_by, n_slices_used, n_unique, transposed, flip_lr, flip_ud, slice_gap_mm, rows, cols, row_spacing, col_spacing, slice_thickness, spacing_between, fov_center_x, laterality, patient_id, manufacturer, field_strength, n_files, n_failed`.
- `fov_center_x` is the patient-x (left/right) position of the series centre, in mm. It's recorded for inferring laterality (section 7).

### Testing done
- `tests\test_kneeprep_geometry.py` (run `python tests\test_kneeprep_geometry.py`) slices a synthetic phantom with a marker in all 8 orientation variants per plane, with shuffled slices, extra echoes and anisotropic pixels. All variants come out **identical**, with the marker in the expected position. It also checks: fallback when IPP is missing, 1–144 source slices → 32, padding stays 0, series selection rules, and 10–15° oblique scans.
- A dry run of the whole prep notebook against a fake competition tree (missing plane, corrupt file, tar rollover) completed correctly.
- On real data: the orientation-check image looked correct (sagittal: femur up, patella left; coronal: femur up; axial: patella up). **All 8,551 chunk-0 series were already in the standard orientation** (no flips needed); the flips are a safety net for the test set.

---

## 6. Stage 2 run: DICOM → shards ✅

**Notebook:** `kaggle\prep_notebook.ipynb` (CPU, Internet on). Settings cell: `CHUNK_ID`, `N_CHUNKS = 2`, `TIMING_ONLY`, `N_TIMING = 50`, `MAX_OUTPUT_GB = 19`, `PART_GB = 2`.

- Studies go to chunk `i % N_CHUNKS` by sorted StudyInstanceUID.
- Per-study npz files are written to `/tmp/shards`, then appended to **~2 GB `.tar` parts** in `/kaggle/working`. This works around **Kaggle's limit of at most 500 output files per notebook version**.
- Committed as two separate notebooks: **`omerraza2202/rsna-knee-prep-chunk0`** and **`omerraza2202/rsna-knee-prep-chunk1`**.
- Success is the line `ALL SAVED` in the log. It's *not* the last line, because the summary cells and Kaggle's nbconvert output follow it.

### Results
| | Timing run (50) | Chunk 0 (2,204) | Total |
|---|---|---|---|
| Wall time | 0.85 s/study (4 workers) | **32.7 min** (0.89 s/study) | ~1.1 h for both chunks |
| Output | 5.13 MB/study | **11.73 GB in 6 tars** | ~23.4 GB, 12 tars |
| Study / series errors | 0 / 0 | **0 / 0**, 0 failed slice reads | 4,407 / 4,407 saved |
| All 4 kinds present | 39/50 | **1,939 / 2,204 (88%)** | — |
| Missing kind (chunk 0) | — | no `sag_fl` 123 · no `cor_fl` 72 · no `sag_t1` 70 | the model must handle missing views |
| Laterality known (chunk 0) | 22/50 | **1,121 / 2,204 (51%)**: L 562, R 559 | — |
| Unique slices per series | median 30, max 320 | mean 34.6, median 30, max 320 (3D scans, block-averaged to 32) | — |

### Download (done)
```powershell
$env:PYTHONUTF8 = "1"
kaggle kernels output omerraza2202/rsna-knee-prep-chunk0 -p data\download
kaggle kernels output omerraza2202/rsna-knee-prep-chunk1 -p data\download
(Get-ChildItem data\download\*.tar | ForEach-Object { tar -tf $_.FullName } | Measure-Object).Count   # 4407
Get-ChildItem data\download\shards_*.tar | ForEach-Object { tar -xf $_.FullName -C data\shards }
Move-Item data\download\*_meta_chunk*.csv data\raw\
(Get-ChildItem data\shards\*.npz).Count   # 4407 ✅
```
The tars are kept in `data\download` as a backup (disk space is fine). They can also be re-downloaded any time from the committed notebooks.

---

## 7. Decisions made, and design notes for the coming stages

### Stage 1 labeler (next)
- **Model:** `qwen3.5:9b` through the local Ollama HTTP API. **Pass `think: false` explicitly**, because recent Ollama starts qwen3.5 with thinking on. Use temperature 0, JSON output, and **`num_ctx` 8192** (Ollama's default context is 4096, which is too small for the system prompt plus a long report). Use a long `keep_alive` so the model stays loaded.
- **Output size is what limits speed:** at ~52 tokens/s, a full 12-field JSON with evidence (~400 tokens) takes ~8 s per report, or ~10 h for 4,407. **Decision:** the model lists **only findings that are present or uncertain**, each with a short evidence quote; anything not listed counts as absent. This should keep the full run to one night. Ollama reuses the processed system prompt across requests, so only the report text is processed each time.
- The runner must be **resumable** (append to JSONL, skip done IDs), log to a file, and run in its own PowerShell window.
- Plan unchanged from the game plan: dev 20 / held-out 38 split of the 58 labeled studies (multilabel-stratified so each split has positives for every label); precise multilingual definitions; map present=1, absent=0, uncertain=0.5; score per label against the held-out 38 (accuracy, sensitivity, specificity, κ); weak-label fixes as in game plan §6.
- Output: `data\labels\` (raw JSONL plus `soft_labels.parquet` with a source flag: human vs LLM).

### Laterality
- Tags give laterality for ~51% of studies.
- **Planned inference for the rest:** only the scanned knee is in the coil, so the knee sits off-centre left/right on the scanner. A negative `fov_center_x` (patient right in LPS) suggests a right knee, positive a left knee. **Validate this against the ~2,200 studies with known laterality before using it.** Anatomy (the fibula is lateral) is a fallback.
- Whether to mirror left knees to look like right knees during training is still open; flip augmentation is the alternative.

### Modelling notes (carried over plus new)
- In M1, embeddings are computed once, so random crop/flip augmentation can't happen on the fly. Options: precompute a flipped copy of the embeddings, and rely on slice/series dropout.
- Feed `Fluid_Sensitive` and `Fat_Suppression` to the model separately (equal in train, may differ in test).
- Check for patients with several studies (`PatientID` in `study_meta`) before building CV folds; group by patient if any exist.
- 3D series with up to 320 slices are compressed to 32 thick averaged slices. Acceptable for now; revisit if those studies score badly.

---

## 8. What's left: revised timeline (from 7 Oct)

| Dates | Work | Done when |
|---|---|---|
| **7–8 Oct** | Commit code to git · split the 58 (dev 20 / held-out 38) · write the definitions prompt · iterate on dev · score on held-out 38 · **full labeling run overnight on 8 Oct** (on 7 Oct if ready) | `soft_labels.parquet` for all 4,407 studies; per-label scores on the 38 |
| **9 Oct** | Laterality inference check · patient-duplicate check · CV folds · **DINOv2-S slice embedding extraction** (~500k slices) · M1 training + 5-fold CV | M1 OOF macro AUC vs LLM labels and vs the 58 |
| **10–11 Oct** | **Submission notebook** (embed `kneeprep`, offline wheels dataset, DINOv2 weights dataset, M1 checkpoints) · runtime test · **first submission by 11 Oct** | LB score; runtime well under 9 h |
| **12–15 Oct** | Fix weak labels (re-prompt / two-prompt agreement / keyword rules / bigger model on Kaggle 2×T4) · start M2 fine-tuning · **Rishi and Arfeen accept the rules and merge the team by 15 Oct 23:59 UTC** | Team of 3 on Kaggle; M2 running |
| **16–20 Oct** | M2 folds (overnight) · rank ensemble · per-label analysis | M2 beats M1 in CV |
| **21–22 Oct** | Final runtime checks · pick 2 final submissions (best CV + best blend) | Submitted by 22 Oct 23:59 UTC |
| by 5 Nov | If placing: clean code, write-up, video | — |

**Dropped unless time frees up:** M3 (BiomedCLIP / DINOv2-B diversity model).

### Kaggle GPU budget
~30 h/week on Omer's account. Not used yet. Don't log into Rishi's or Arfeen's accounts (account sharing breaks Kaggle's rules).

---

## 9. Deviations from the original game plan

| Game plan said | What we did | Why |
|---|---|---|
| WSL2 + Ubuntu | **Native Windows + PowerShell** | Faster setup; nothing in the plan needs Linux. WSL2 stays a fallback for M2 |
| Develop `kneeprep` on the 3 example test studies locally first | Inspected 40 real training studies on Kaggle, wrote `kneeprep`, tested it on synthetic phantoms, then did a 50-study timing run | Quicker, and covered far more variety |
| One output folder of npz per chunk, saved as private datasets | **npz packed into ~2 GB tar parts**; downloaded with `kaggle kernels output` (no datasets made) | Kaggle keeps at most 500 output files per version |
| 2–3 chunks, JPEG 2000 risk | **2 chunks, ~33 min each** | Data was uncompressed in practice; decoding was fast |
| Optional laterality flip in preprocessing | Not done in `kneeprep`; laterality recorded in metadata instead | Only ~51% known from tags; keeps preprocessing identical for unknown cases |
| Labeler outputs all 12 fields | Outputs only present/uncertain findings | Generation speed (~52 tok/s) would otherwise make the run ~10 h |
| Timeline: first submission ~8 Oct | First submission ~11 Oct | Started 6 Oct instead of 1 Oct |

---

## 10. Gotchas we hit (so they don't recur)

- **PowerShell paths:** when building paths from a variable, use `Join-Path` (or keep the `\`). Dropping it once created sibling folders like `...KaggleConfigs` next to the project folder.
- **Command Prompt vs PowerShell:** running `Activate.ps1` in cmd opens it in Notepad. Use PowerShell (prompt starts with `PS`).
- **A failed `pip install` line installs nothing:** if one package fails, none of the line is installed. Check with `pip show <pkg>`.
- **Kaggle CLI `'charmap' codec` error:** harmless (it happens while saving the log file, after all outputs have downloaded). Fixed with `PYTHONUTF8=1`.
- **`kaggle kernels output` skips files** that already exist locally with the same name. That's fine for the shared `kneeprep.py` and png; use `--force` to overwrite.
- **Kaggle notebook output:** at most 500 files per version, and roughly a 20 GB limit on `/kaggle/working`.
- **Kaggle commit logs:** look for the success line with Ctrl+F; the last lines are always Kaggle's nbconvert step plus harmless `SyntaxWarning`s and debugger warnings.
- **Data path:** the competition is mounted at `/kaggle/input/competitions/rsna-knee-abnormality-detection/`. Search for the folder with both the CSV and the `train_series/` folder.
- **Pandas groupby on a column that may not exist:** the inspection notebook crashed when there were zero decode failures (the `err` column was missing). Fixed with a guard.
- **Ollama:** the first `ollama run` loads 6.6 GB (~2 min the very first time, ~10 s after). The model unloads after 5 idle minutes.

---

## 11. Immediate next steps (start of the next chat)

1. **Commit the code:** add the data folders and `__pycache__/` to `.gitignore`, then `git add` → `git commit -m "kneeprep + Stage 2 prep notebook"`.
2. **Check the offline wheels:** confirm the `kaggle_dicom_inspect` notebook version saved `/kaggle/working/wheels`. If not, regenerate them later in the submission-prep step.
3. **Stage 1, the labeler:**
   - a. Script to split the 58 labeled studies into dev 20 / held-out 38 (multilabel-stratified, fixed seed, saved to `data\labels\split.csv`).
   - b. Write the definitions prompt, deciding the open definition questions in game plan §6 from the dev reports.
   - c. Ollama runner (`src\labeler\`): `think: false`, `num_ctx` 8192, temperature 0, JSON output listing present/uncertain findings only, resumable JSONL, log file.
   - d. Evaluation script against the held-out 38, per label.
   - e. When the scores are acceptable: full overnight run over all 4,407 reports, then build `soft_labels.parquet`.
4. In parallel (later the same day): laterality inference check using `series_meta`, and a check for patients with several studies.
5. Remind Rishi and Arfeen to accept the competition rules on their own accounts ahead of the 15 Oct merger deadline.
