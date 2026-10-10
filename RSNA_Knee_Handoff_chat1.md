# RSNA Knee Abnormality Detection — Handoff (end of 8 Oct 2026)

*Written Thu 8 Oct 2026, 22:15 IST · Team: Omer Raza (does the work), Rishi Sharma, Arfeen Khan · Covers everything up to and including chat 3 (7–8 Oct).*

**This file replaces the older handoffs as the starting point.** The older docs in the Project are still accurate for details:
`claude/RSNA_Knee_Progress.md` (environment, Stage 2 DICOM prep, 6–7 Oct), `claude/RSNA_Stage1_Labeler.md` (report labeler in detail), `claude/RSNA_Models_Progress.md` (models, 8 Oct), `RSNA_Knee_Game_Plan.md` (original strategy, partly outdated — see §12).

**To continue in a new chat:** "Read `claude/RSNA_Knee_Handoff_2026-10-08.md` in the Project. The overnight M2 run `ft2` has finished — here is its log: …". The new chat cannot see the laptop: **paste outputs**, and attach the current code if it needs to edit files (zip `src/` + `scripts/`, or let it attach the private GitHub repo `omerwritescode/rsna-knee-abnormality-detection`). The repo on GitHub is the source of truth for code.

---

## 1. Where we stand

| | Result |
|---|---|
| **Public LB (best)** | **0.860** — notebook Version 2 = M1 v2 + M2 ft1 blend (8 Oct ~21:00 IST) |
| Public LB (first) | 0.800 — Version 1 = M1 v2 only (8 Oct ~13:30 IST) |
| Best CV (OOF `auc_llm`, 4,407 studies) | **0.8569** (blend 0.1·M1 + 0.9·M2) · M2 alone 0.8565 · M1 alone 0.7936 |
| CV ↔ LB | 0.794 → 0.800, 0.857 → 0.860. **`auc_llm` predicts the LB well** — use it to choose between ideas before spending submissions |
| Submission runtime | ~25 min for the full hidden test (limit 9 h) → lots of headroom |
| Running now | **M2 `ft2`** overnight on the laptop (5 folds, 10 epochs, last 6 blocks trained) — see §9 |
| Public reference points | another team: LB 0.664; a cited public baseline: 0.809 |
| Timeline | ~3 days ahead of the revised plan (first submission target was 11 Oct) |

| Stage | Status |
|---|---|
| Environment (native Windows, venv `knee`, CUDA, Ollama, Kaggle API, git + private GitHub) | ✅ |
| Stage 2: DICOM → 4,407 study shards (`kneeprep`) | ✅ (6–7 Oct) |
| Stage 1: report → soft labels (local LLM) | ✅ (7–8 Oct) |
| DINOv2-S slice embeddings + CV folds + feature cache | ✅ |
| M1 (frozen features + aggregator) | ✅ v2: CV 0.794, LB 0.800 |
| Submission pipeline (offline Kaggle notebook) | ✅ working |
| M2 (fine-tuned DINOv2-S) | ✅ ft1: CV 0.8565 · LB 0.860 with blend |
| M2 ft2 (longer, more blocks) | ⏳ training overnight 8→9 Oct |
| Team merge (Rishi, Arfeen) | ⬜ **deadline 15 Oct 23:59 UTC** |
| Final 2 submissions | ⬜ deadline 22 Oct 23:59 UTC |

---

## 2. Key facts

| Item | Value |
|---|---|
| Competition | `rsna-knee-abnormality-detection` (Kaggle code competition; RSNA host; $77k) |
| Task / metric | 12 binary labels per knee MRI study; **macro ROC AUC** (only ranking matters) |
| Labels | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Data | 4,407 train studies, only **58 human-labelled**; every train study has a free-text report (multilingual); test ≈1,300 studies, **images only** |
| Submission | Kaggle notebook, internet off, ≤ 9 h, writes `submission.csv` |
| Kaggle user | `omerraza2202` (rules accepted). Don't log into Rishi's/Arfeen's accounts (account sharing breaks the rules) |
| **Deadlines** (23:59 UTC = 05:29 IST next morning) | **15 Oct**: entry + team merger (= 16 Oct 05:29 IST) · **22 Oct**: final submission (= 23 Oct 05:29 IST) · 5 Nov: winners' code/video/write-up |
| GitHub | `omerwritescode/rsna-knee-abnormality-detection` — **private** (sharing code publicly before the end would breach Kaggle rules; competition data must not be published) |
| Kaggle dataset | `omerraza2202/rsna-knee-models` (private): M1 + M2 checkpoints, DINOv2 weights, priors, wheels |
| Kaggle notebook | `submit_notebook.ipynb` (Version 1 = 0.800, Version 2 = 0.860) |

---

## 3. Laptop & environment

- HP Victus · i7-13620H · **RTX 5050 8 GB** · 24 GB RAM · Windows 11 · native Windows (no WSL2).
- Project folder: `D:\omer files\RSNA Knee Abnormality Detection - Kaggle` (spaces → always quote). **PowerShell**, not cmd.
- venv: `C:\venvs\knee\Scripts\Activate.ps1` (prompt shows `(knee)`); always `$env:PYTHONUTF8 = "1"` (set permanently too).
- Python 3.12, torch 2.11.0+cu128, timm 1.0.30, pandas, pyarrow, scikit-learn, opencv, pydicom, Ollama 0.35.1 + `qwen3.5:9b`.
- Kaggle CLI configured (`KAGGLE_CONFIG_DIR` = `<project>\.kaggle`).
- DINOv2 weights cached by Hugging Face in `C:\Users\Omer\.cache\huggingface\hub` (symlink warning on Windows is harmless).
- Run every module from the project root: `python -m src.models.xxx`.

### Repo layout (as of 8 Oct)
```
src/
  kneeprep/        __init__.py, core.py            DICOM → uint8 [32,256,256] per series (shared train/test)
  labeler/         split.py prompt.py run.py evaluate.py postprocess.py build_labels.py README.md
  models/          embed.py data.py folds.py metrics.py baseline.py m1.py m2.py predict.py blend.py
scripts/           build_prep_notebook.py  build_submit_notebook.py  make_model_dataset.py
tests/             test_kneeprep_geometry.py
kaggle/            prep_notebook.ipynb  submit_notebook.ipynb  model_dataset/ (git-ignored)
data/              raw/ shards/ embeddings/ labels/ download/   (all git-ignored except labels/split.csv, labels/folds.csv)
runs/              m1/<name>/  m2/<name>/                       (git-ignored)
```
`.gitignore` covers: data/raw, shards, embeddings, download, labels/*.jsonl|*.log|*.parquet|*.txt, `__pycache__/`, `*.zip`, `.kaggle/`, `runs/`, `kaggle/model_dataset/`.

---

## 4. Stage 2 — DICOM → shards (done 6–7 Oct)

`kneeprep` picks up to 4 series per study — `sag_fl`, `cor_fl`, `ax_fl` (fluid-sensitive, FS preferred) and `sag_t1` — sorts slices by 3D position, puts every image in a standard orientation (sagittal: right→left slices, anterior on the left; coronal/axial: patient-left on image right), resizes to 256×256 keeping aspect ratio, resamples to 32 slices, percentile-scales to uint8. **Laterality is not normalised.** Output `data\shards\<uid>.npz`, 4,407/4,407, 0 errors, ~23 GB. Details: `claude/RSNA_Knee_Progress.md` §4–6.

## 5. Stage 1 — report labeler (done 7–8 Oct)

- Local `qwen3.5:9b` via Ollama (`think:false`, temp 0, `num_ctx` 8192, JSON schema). Prompt **v3** (frozen): fixed 12-key output, each `{"q": short quote, "s": absent|present|uncertain}` (quote first), explicit negation rules, structure-specific rules. Runner rule: non-absent with empty quote → absent. Post-processing **p1**: Synovitis absent + moderate/large effusion or hemarthrosis quote → 0.5.
- Scores vs the 58 human labels: dev-20 macro soft AUC 0.88, **held-out-38 0.86** (κ 0.60). Weakest: Synovitis 0.69, Fracture 0.77.
- Full run 4,407/4,407, 0 failures, 8.6 s/report (~10.6 h). Output `data\labels\soft_labels.parquet` (12 labels as 1/0.5/0, `source` human/llm). Prevalences plausible (e.g. medial:lateral meniscus ≈ 2.6:1).
- **Key finding:** the 58 human labels were read from the images, not the reports (8/20 dev studies had a label contradicting its report) → report labels have a ceiling; image models can exceed them.
- Details: `claude/RSNA_Stage1_Labeler.md`.

## 6. Features, folds, M1 (8 Oct)

- **Embeddings** (`src/models/embed.py`): frozen DINOv2-S (`vit_small_patch14_dinov2.lvd142m`, timm), 224 px, fp16, per slice CLS+mean-patch (768) and 4×4 patch grid; mirrored coronal/axial copies. 4,407 studies in ~20 min. `data/embeddings/dinov2s_224_g4/`.
- **Knee mirroring (left↔right)** in kneeprep orientation: coronal/axial = flip image horizontally; sagittal = reverse slice order. Used as augmentation and test-time augmentation everywhere.
- **Folds** (`src/models/folds.py`): 5 folds, grouped by PatientID (**no patient has >1 study**), balanced on labels, report language and the 58 human studies. `data/labels/folds.csv` (committed).
- **Cache** (`src/models/data.py`): `cache_g2/` = feats [N,6,32,768] + grid pooled to 2×2 + present flags + per-channel `stats.npz`.
- **Scores (OOF):** M1 v1 0.753 · logistic-regression baseline 0.781 · **M1 v2 0.794** (per-channel standardisation + mean-pool path + 40 epochs) → LB 0.800. The baseline ≈ M1 showed the **frozen features were the bottleneck**.

## 7. M2 — fine-tuned DINOv2-S (8 Oct)

- `src/models/m2.py`: DINOv2-S with the **first 8 of 12 blocks frozen**, last 4 + final norm trained (lr 2e-5); the M1 aggregator on top, **initialised from the same fold's M1 v2 checkpoint** (lr 1e-4); same feature definition + standardisation; bf16. Training: 16 of 32 slices per series (random offset; positions given to the head), per-series affine/intensity augmentation, knee mirror, series dropout; 2 studies/step × 8 accumulation; 6 epochs. Eval: all 32 slices + mirror TTA. Code split into `stem` (frozen part), `tail` (trained blocks + pooling), `head_from_features`.
- Speed on the laptop: 6.7 min/epoch, ~435 img/s; ~1 h per fold incl. evaluation.
- **ft1 results:** fold 0: start 0.794 (= M1 v2, confirms setup) → ep2 0.823 → ep4 0.851 → **ep6 0.858** (still rising). **5-fold OOF auc_llm 0.8565.**

| Label | M1 v2 | M2 ft1 | Blend 0.9 |
|---|---|---|---|
| ACL | 0.812 | 0.866 | 0.868 |
| MCL | 0.738 | 0.764 | 0.768 |
| Medial Meniscus | 0.761 | 0.852 | 0.850 |
| Lateral Meniscus | 0.699 | 0.762 | 0.761 |
| Medial OA | 0.839 | 0.891 | 0.891 |
| Lateral OA | 0.782 | 0.843 | 0.842 |
| PF OA | 0.794 | 0.833 | 0.835 |
| Effusion | 0.938 | 0.953 | 0.954 |
| Synovitis | 0.872 | 0.884 | 0.887 |
| Baker's | 0.764 | 0.916 | 0.915 |
| Contusion | 0.785 | 0.872 | 0.871 |
| Fracture | 0.740 | 0.842 | 0.840 |
| **Macro** | **0.794** | **0.857** | **0.857** |

Blend sweep (auc_llm): w_M2 0.0 → 0.794 · 0.5 → 0.845 · 0.9 → **0.8569** · 1.0 → 0.8565. **M1 now adds almost nothing** → future gains come from better / more M2 models. Weakest labels now: **MCL 0.76, Lateral Meniscus 0.76**, then PF OA 0.83.

---

## 8. Submission pipeline (working)

1. `python -m src.models.blend --m1 v2 --m2 ft1` → `runs\m2\ft1\blend.json` (chosen weights).
2. `python scripts\make_model_dataset.py --run v2 --m2-run ft1` → `kaggle\model_dataset\` (m1/, m2/ + blend.json, dinov2/model.safetensors, priors.json, Linux cp3.13 wheels for pydicom + JPEG decoders, dataset-metadata.json).
3. `kaggle datasets version -p "kaggle\model_dataset" -m "<what changed>" --dir-mode zip` (first time was `datasets create`). M2 upload ≈ 500 MB.
4. `python scripts\build_submit_notebook.py` → `kaggle\submit_notebook.ipynb` (embeds kneeprep + models code via %%writefile; finds inputs by searching /kaggle/input).
5. Kaggle: open the existing notebook → **Edit → File → Import Notebook** (replace) → **Input panel: update `rsna-knee-models` to the latest version** (⋮ → check for updates) → GPU **T4** (not P100), Internet **off** → Run All (3 example studies; last cell must list both models) → **Save Version → Save & Run All** → submit that version.

`src/models/predict.py` `run_test()`: DICOM → kneeprep (fork CPU pool, started before CUDA) → M1 (Embedder + fold ensemble) and/or **M2Ensemble** (frozen stem run once per image, shared by all folds; fp16) → blend from `blend.json` → training prevalence (priors) for any study that fails. Uses whichever of `m1/` / `m2/` the dataset contains.
Local checks before uploading: `python -m src.models.predict --check --run v2` (M1 parity) and `python -m src.models.predict --check-m2 ft1` (M2 inference vs training OOF, shared-stem equality).

---

## 9. In progress right now

```powershell
python -m src.models.m2 --name ft2 --epochs 10 --frozen-blocks 6 --eval-every 5
```
Started 8 Oct ~22:00 IST, expected ~7–8 h (5 folds × 10 epochs; ~20% slower per epoch than ft1). If it ran out of memory: `--bs 1 --accum 16`. Outputs in `runs\m2\ft2\` (fold*.pt, oof_fold*.csv, train.log, metrics.csv at the end).

---

## 10. Immediate next steps (9 Oct)

1. **Check ft2**: final `OOF scores` table in the log (compare macro auc_llm with ft1's **0.8565**; per-fold val at ep 5 and 10). Commit.
2. **Extend the code for several M2 runs** (needs a new chat with the code attached):
   - `blend.py`: accept several M2 runs (e.g. `--m2 ft1,ft2`), search weights (or a simple average), write one `blend.json`.
   - `make_model_dataset.py`: copy each run into its own folder (`m2_ft1/`, `m2_ft2/`).
   - `predict.py`: build one `M2Ensemble` per run folder (runs with different `frozen_blocks` cannot share the stem, but each run's folds still can); blend all by `blend.json`. Consider dropping M1 (weight ~0.1, costs runtime).
   - Rebuild dataset + notebook → submit the ft1+ft2 blend.
3. Remind **Rishi and Arfeen** to accept the competition rules on their own Kaggle accounts now; then send/accept the team merge **before 15 Oct 23:59 UTC**. Check the team page shows all three.

## 11. Plan to the deadline

| Dates | Work |
|---|---|
| 9 Oct | ft2 results → multi-run blend → submission |
| 10–12 Oct | Next model improvement (pick by expected gain, test on fold 0 first, ~1–1.5 h each): **higher input resolution** (e.g. 336/448 px — small structures: menisci, MCL), **more slices** at train time, **more trainable blocks / longer**, a **second seed** for ensembling. Optional: Kaggle GPU (30 h/week) for parallel runs |
| by 15 Oct | **Team merge done** (hard deadline) |
| 13–18 Oct | Label improvements for the weakest labeler classes (Synovitis, Fracture) — e.g. second prompt or larger model on Kaggle 2×T4; **pseudo-labelling** with M2 (models now beat the report labels on human-label AUC) |
| 19–21 Oct | Final ensemble from the best runs (OOF-weighted), runtime check, robustness |
| 22 Oct | **Pick 2 final submissions** (best CV and best LB / safest blend) before 23:59 UTC (= 05:29 IST 23 Oct) |
| by 5 Nov | If placing: clean code, write-up, video |

## 12. Decisions & findings (keep these)

- **Use `auc_llm` (OOF vs LLM labels, uncertain rows excluded) as the main CV metric**; `auc_human` (58 studies) is noisy. Both moved together with the LB so far.
- Game plan said "aim for 0.946" — **not realistic**; public references are 0.66–0.81. We're at 0.860.
- Report labels match image-based ground truth only ~82% (also seen by another team) → label noise caps CV; image models trained on them can still beat the reports.
- Fine-tuning > frozen features (+0.06). Initialising M2's head from M1 makes training stable and fast.
- Random/site-mixed folds may inflate CV (another team saw ~0.05); for us CV and LB agreed, so we keep the current folds.
- The 58 human-labelled studies are included in training with their true labels (and spread over folds).

## 13. Gotchas we hit (don't repeat)

- PowerShell `Out-File` garbles non-Latin text → use the scripts' `--save` options.
- Ollama: `num_ctx` default 4096 is too small (we use 8192); `OLLAMA_NUM_PARALLEL=2` gave no speed-up on 8 GB and pushes the model partly onto the CPU — keep it unset. `ollama ps` shows nothing until a model is loaded.
- Version numbers: Ollama "0.35.1" is newer than "0.9" (compare part by part).
- `kneeprep` once lived only in an extracted `rsna-knee-stage2\` folder and was never committed → now in `src\kneeprep\`. Keep all code inside the repo.
- Jupyter `%%writefile` cells cannot be empty → empty `__init__.py` files are created in the notebook's setup cell.
- Reading CSVs: StudyInstanceUIDs must be read as strings (`dtype={"StudyInstanceUID": str}`).
- Don't unzip new code while a training run is going: DataLoader workers reload modules each epoch.
- Kaggle notebooks stay pinned to the dataset version they were created with → update the input after every `datasets version`.
- Kaggle: use **T4**, not P100 (newer torch no longer supports it). Import a new notebook version via File → Import Notebook into the existing notebook.
- First `kaggle datasets` upload needs a minute or two of processing before the notebook can use it.
- Windows: the HF "symlinks" warning is harmless; `Expand-Archive` prints nothing on success; zip timestamps show UTC.

## 14. Hardware care (laptop)

Long runs are safe: the GPU throttles itself near its limit (high 80s °C) and the laptop shuts down before damage; observed 78 °C at full clocks. To limit wear: cap battery charge at 80% (HP *Adaptive Battery Optimizer* / battery care in BIOS F10 → Configuration or OMEN Gaming Hub), keep it on the stand with vents clear, clean vents occasionally, use *Balanced* mode if it runs hot (≈10–15% slower). Check: `nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv`.

## 15. Command cheat-sheet

```powershell
cd "D:\omer files\RSNA Knee Abnormality Detection - Kaggle"
C:\venvs\knee\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# train / evaluate
python -m src.models.m2 --smoke                                  # quick pipeline check
python -m src.models.m2 --folds 0 --name X --eval-start          # one fold test
python -m src.models.m2 --name X --eval-every 6                  # all folds
python -m src.models.blend --m1 v2 --m2 ft1                      # OOF blend weight

# submission
python -m src.models.predict --check-m2 ft1
python scripts\make_model_dataset.py --run v2 --m2-run ft1
kaggle datasets version -p "kaggle\model_dataset" -m "msg" --dir-mode zip
python scripts\build_submit_notebook.py

# git
git add . ; git commit -m "msg" ; git push
```
