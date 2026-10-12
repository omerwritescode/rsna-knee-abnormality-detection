# RSNA Knee Abnormality Detection — Complete Handoff (12 Oct 2026)

*Written Mon 12 Oct 2026, 19:00 IST · Team: Omer Raza (does all the work), Rishi Sharma, Arfeen Khan · Covers everything from the start (6 Oct) to now.*

**This file replaces all earlier handoffs as the starting point.** Older docs in the Project are still accurate for fine detail: `claude/RSNA_Knee_Progress.md` (environment + DICOM prep, 6–7 Oct), `claude/RSNA_Stage1_Labeler.md` (report labeler), `claude/RSNA_Models_Progress.md` (M1/M2, 8 Oct), `RSNA_Knee_Handoff_chat1.md` (state on 8 Oct). `RSNA_Knee_Game_Plan.md` is the original strategy and is partly outdated (see §13).

---

## 0. How to continue in a new chat

1. Say: *"Read `claude/RSNA_Knee_Handoff_2026-10-12.md` in the Project."*
2. Paste the results of whatever ran since (see §10 for exactly what to paste).
3. The new chat **cannot see the laptop or the GitHub repo** (the repo is private and Claude's GitHub connection only reaches public repos). For code edits: zip `src\` + `scripts\` (+ `tests\`, `.gitignore`) and attach it:
   ```powershell
   cd "D:\omer files\RSNA Knee Abnormality Detection - Kaggle"
   Compress-Archive -Path src, scripts, tests, .gitignore -DestinationPath code_YYYY-MM-DD.zip -Force
   ```
   Claude sends back a zip of only the changed files → `Expand-Archive -Path "$HOME\Downloads\<zip>" -DestinationPath . -Force` → commit + push. The repo on GitHub is the source of truth.
4. Claude's own workspace cannot install PyTorch, so code is tested there with stand-ins; the real test is always a local check on the laptop (`--smoke`, `--check-m2`, fold-0 runs).

---

## 1. Where we stand

| | Value |
|---|---|
| **Best public LB** | **0.889** — M2 **ft2** alone (submitted 11 Oct) |
| Best CV (OOF `auc_llm`, 4,407 studies) | **0.8790** (ft2) · auc_human 0.855 |
| CV ↔ LB | 0.794→0.800 · 0.857→0.860 · 0.879→0.889. **CV tracks the LB closely (LB slightly higher)** → choose ideas by CV, not by spending submissions |
| Public reference points | another team 0.664; a cited public baseline 0.809 |
| In progress | Fold-0 experiments **ft3a** and **ft3b** (§10) — results not yet reported |
| ⚠️ Next hard deadline | **Team merge: 15 Oct 23:59 UTC = 16 Oct 05:29 IST** |

### Submission history
| Date | Notebook version | Models | CV | Public LB |
|---|---|---|---|---|
| 8 Oct ~13:30 IST | V1 | M1 v2 | 0.7936 | 0.800 |
| 8 Oct ~21:00 IST | V2 | 0.1·M1 v2 + 0.9·M2 ft1 | 0.8569 | 0.860 |
| 11 Oct | V3 | M2 ft2 only | 0.8790 | **0.889** |

### Stage status
| Stage | Status |
|---|---|
| Environment (native Windows, venv `knee`, CUDA, Ollama, Kaggle API, git + private GitHub) | ✅ |
| Stage 2: DICOM → 4,407 study shards (`kneeprep`) | ✅ 6–7 Oct |
| Stage 1: reports → soft labels (local LLM) | ✅ 7–8 Oct |
| DINOv2-S embeddings, CV folds, feature cache, M1 | ✅ 8 Oct |
| Submission pipeline (offline Kaggle notebook), multi-run blending | ✅ |
| M2 ft1 → ft2 | ✅ ft2 = current best |
| Fold-0 tests ft3a / ft3b | ⏳ |
| **Team merge (Rishi, Arfeen)** | ⬜ **by 15 Oct 23:59 UTC** |
| Final 2 submissions | ⬜ by 22 Oct 23:59 UTC (= 23 Oct 05:29 IST) |

---

## 2. Key facts

| Item | Value |
|---|---|
| Competition | `rsna-knee-abnormality-detection` (Kaggle code competition, RSNA host, $77k) |
| Task / metric | 12 binary labels per knee MRI study · **macro ROC AUC** (only the ranking matters) |
| Labels | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Data | 4,407 train studies, only **58 human-labelled**; every train study has a free-text (multilingual) report; test ≈ 1,300 studies, **images only**; raw data ~570 GB (not stored locally) |
| Submission | Kaggle notebook, internet **off**, ≤ 9 h, writes `submission.csv` |
| Kaggle user | `omerraza2202` (rules accepted). **Never log into Rishi's/Arfeen's accounts** (account sharing breaks the rules) |
| Deadlines (23:59 UTC = 05:29 IST next day) | **15 Oct**: entry + team merger · **22 Oct**: final submission · 5 Nov: winners' code / video / write-up |
| GitHub | `omerwritescode/rsna-knee-abnormality-detection` — **private** (public code before the end breaches the rules; competition data must never be published) |
| Kaggle dataset | `omerraza2202/rsna-knee-models` (private) — currently: `m2_ft2/` + `blend.json` + DINOv2 weights + priors + wheels |
| Kaggle notebooks | `submit_notebook` (submissions) · `rsna-knee-prep-chunk0` / `rsna-knee-prep-chunk1` (outputs = the 12 shard tars) · `kaggle_dicom_inspect` |
| Constraints | No paid APIs / spending. Train on the laptop (Omer chose this over Kaggle GPUs — see §9) |

---

## 3. Laptop & environment

- HP Victus · i7-13620H · **RTX 5050 8 GB** (Blackwell, has bf16) · 24 GB RAM · Windows 11 · **native Windows** (no WSL2).
- Project folder: `D:\omer files\RSNA Knee Abnormality Detection - Kaggle` (spaces → always quote). Use **PowerShell**, not cmd.
- Every new PowerShell window:
  ```powershell
  cd "D:\omer files\RSNA Knee Abnormality Detection - Kaggle"
  C:\venvs\knee\Scripts\Activate.ps1      # prompt shows (knee)
  $env:PYTHONUTF8 = "1"                   # also set permanently
  ```
- Python 3.12, torch 2.11.0+cu128, timm 1.0.30, pandas, pyarrow, scikit-learn, opencv, pydicom (+ JPEG decoders), Ollama 0.35.1 + `qwen3.5:9b`.
- Kaggle CLI: `KAGGLE_CONFIG_DIR` = `<project>\.kaggle` (key in `kaggle.json`, git-ignored).
- DINOv2 weights cached in `C:\Users\Omer\.cache\huggingface\hub` (the Windows symlink warning is harmless).
- Run every module from the project root: `python -m src.models.xxx`.
- **Power during long runs:** Windows power mode **Balanced** (Best Performance = hot; Best Energy Efficiency throttles the CPU and starves the data loader). Fans at max speed are fine (cooler = less wear; only cost is noise and some fan wear). Keep vents clear, laptop on a stand, cap battery charge at 80% if possible. Watch: `nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv -l 30` (60s–70s °C is good; earlier 78 °C on auto fans; it throttles itself in the high 80s).

### Repo layout (12 Oct)
```
src/
  kneeprep/   __init__.py core.py                DICOM → uint8 [32,256,256] per series (shared by train and test)
  labeler/    split.py prompt.py run.py evaluate.py postprocess.py build_labels.py README.md
  models/     embed.py data.py folds.py metrics.py baseline.py m1.py m2.py predict.py blend.py
scripts/      build_prep_notebook.py  build_submit_notebook.py  make_model_dataset.py
              build_train_notebook.py  make_train_inputs.py        (Kaggle training; built, unused — §9)
tests/        test_kneeprep_geometry.py
kaggle/       prep_notebook.ipynb  submit_notebook.ipynb  train_notebook.ipynb
              model_dataset/ train_inputs/                     (git-ignored)
data/         raw/ shards/ embeddings/ labels/ download/       (git-ignored except labels/split.csv, labels/folds.csv)
runs/         m1/<name>/  m2/<name>/  blend.json               (git-ignored)
```
`.gitignore`: data/raw, shards, embeddings, download, labels/*.jsonl|*.log|*.parquet|*.txt, `__pycache__/`, `*.zip`, `.kaggle/`, `runs/`, `kaggle/model_dataset/`, `kaggle/train_inputs/`.

---

## 4. Stage 2 — DICOM → shards (done 6–7 Oct)

- Ran on Kaggle CPU in two notebooks (`rsna-knee-prep-chunk0/1`, ~33 min each); output packed into ~2 GB tar parts (Kaggle keeps ≤ 500 output files per version); downloaded with `kaggle kernels output` and extracted to `data\shards\<StudyInstanceUID>.npz` — **4,407/4,407, 0 errors, ~23 GB**. Tars kept in `data\download\` as backup.
- `kneeprep` picks up to 4 series per study: `sag_fl`, `cor_fl`, `ax_fl` (fluid-sensitive, fat-suppressed preferred) and `sag_t1`. 88% of studies have all 4; the model handles missing ones.
- Per series: sort slices by 3D position (IPP/IOP), drop duplicate positions, put into a standard orientation (sagittal: patient right→left, anterior on the left; coronal/axial: patient-left on image right), resize to **256×256** keeping aspect ratio (centre-padded), resample to **32 slices**, clip 0.5–99.5 percentile → uint8. **Laterality is not normalised** (tags give it for only ~51% of studies).
- Synthetic-phantom tests (`tests\test_kneeprep_geometry.py`) pass for all orientation variants; real-data orientation image checked.

## 5. Stage 1 — report labeler (done 7–8 Oct)

- Local `qwen3.5:9b` via Ollama (`think:false`, temperature 0, `num_ctx` 8192, JSON schema). Prompt **v3** (frozen): fixed 12-key output, each `{"q": short quote, "s": absent|present|uncertain}` (quote first), explicit negation and structure-specific rules. Runner rule: non-absent with an empty quote → absent. Post-processing **p1**: Synovitis absent + moderate/large effusion or hemarthrosis quote → 0.5.
- Split of the 58 human-labelled studies: dev 20 / held-out 38. Scores vs human labels: dev macro soft AUC 0.88, **held-out 0.86** (κ 0.60). Weakest: Synovitis 0.69, Fracture 0.77.
- Full run 4,407/4,407, 0 failures, 8.6 s/report (~10.6 h). Output `data\labels\soft_labels.parquet` (12 labels as 1 / 0.5 / 0, `source` = human | llm). The 58 human studies keep their true labels.
- **Key finding:** the human labels were read from the images, not the reports (8/20 dev studies had a label contradicting the report) → report labels are noisy (~82% agreement, also seen by another team) → this caps CV, but image models can still beat the reports.

## 6. Features, folds, M1 (8 Oct)

- **Embeddings** (`embed.py`): frozen DINOv2-S (`vit_small_patch14_dinov2.lvd142m`, timm), 224 px, fp16; per slice CLS + mean-patch (768) and a 4×4 patch grid; mirrored coronal/axial copies. `data/embeddings/dinov2s_224_g4/`, cache `cache_g2/` (grid pooled to 2×2, per-channel `stats.npz`).
- **Knee mirroring (left↔right):** coronal/axial = flip image horizontally; sagittal = reverse slice order. Used as augmentation and as test-time augmentation everywhere.
- **Folds** (`folds.py`): 5 folds grouped by PatientID (no patient has >1 study), balanced on labels, report language and the 58 human studies → `data/labels/folds.csv` (committed).
- **Metrics** (`metrics.py`): `auc_llm` = OOF AUC vs LLM labels with uncertain (0.5) rows left out per label — **main CV metric**; `auc_human` = the 58 human studies (noisy).
- **M1** (`m1.py`): transformer aggregator over frozen slice features. v1 0.753 · logistic-regression baseline 0.781 · **v2 0.794** (per-channel standardisation + mean-pool path + 40 epochs) → LB 0.800. The baseline ≈ M1 showed frozen features were the bottleneck → M2.

## 7. M2 — fine-tuned DINOv2-S (8–10 Oct)

`src/models/m2.py`: DINOv2-S with the first `--frozen-blocks` of 12 blocks frozen, the rest + final norm trained (lr 2e-5); the M1 aggregator on top, **initialised from the same fold's M1 v2 checkpoint** (`--init v2`, lr 1e-4) + same feature standardisation. Training: 16 of 32 slices per series (random offset; positions given to the head), per-series affine/intensity augmentation, knee mirror, series dropout (p 0.2); `--bs 2 --accum 8`; warmup + cosine schedule. Eval: all 32 slices + mirror TTA. The code is split into `stem` (frozen part, shared across folds at inference), `tail`, `head_from_features`.

| Run | Command | Time (laptop) | OOF auc_llm | auc_human | LB |
|---|---|---|---|---|---|
| ft1 | `--name ft1` (8 frozen, 6 epochs) | ~6.7 min/epoch, ~1 h/fold | 0.8565 | 0.828 | 0.860 (blend with M1) |
| **ft2** | `--name ft2 --epochs 10 --frozen-blocks 6 --eval-every 5` | ~8.3 h total (Balanced mode, 9→10 Oct) | **0.8790** | **0.855** | **0.889** |

Per label (auc_llm):

| Label | M1 v2 | ft1 | **ft2** |
|---|---|---|---|
| ACL | 0.812 | 0.866 | **0.902** |
| MCL | 0.738 | 0.764 | **0.790** |
| Medial Meniscus | 0.761 | 0.852 | **0.911** |
| Lateral Meniscus | 0.699 | 0.762 | **0.799** |
| Medial OA | 0.839 | 0.891 | **0.905** |
| Lateral OA | 0.782 | 0.843 | **0.863** |
| PF OA | 0.794 | 0.833 | **0.842** |
| Effusion | 0.938 | 0.953 | **0.958** |
| Synovitis | 0.872 | 0.884 | **0.888** |
| Baker's | 0.764 | 0.916 | **0.926** |
| Contusion | 0.785 | 0.872 | **0.904** |
| Fracture | 0.740 | 0.842 | **0.861** |
| **Macro** | **0.794** | **0.857** | **0.879** |

ft2 auc_human per label: ACL 0.893, MCL 0.742, Med Men 0.888, Lat Men 0.758, Med OA 0.955, Lat OA 0.764, PF OA 0.775, Effusion 0.943, Synovitis 0.753, Baker's 0.967, Contusion 0.929, Fracture 0.894.

**Weakest labels now:** MCL 0.79, Lateral Meniscus 0.80, PF OA 0.84 (and on human labels Synovitis, Lat OA, PF OA, Lat Men, MCL ≈ 0.74–0.78).

**Blend test (10 Oct)** — `python -m src.models.blend --m2 ft1,ft2`:

| Blend | auc_llm | auc_human |
|---|---|---|
| M1 alone | 0.7936 | 0.732 |
| ft1 alone | 0.8565 | 0.828 |
| ft2 alone | 0.8790 | 0.855 |
| equal weights | 0.8672 | 0.832 |
| best grid (M1 0.05 / ft1 0.05 / ft2 0.90) | 0.8792 | 0.854 |

→ ft1 and M1 add nothing; **submitted ft2 alone**. Ensembling only pays off with models that make different mistakes.

---

## 8. Submission pipeline (current)

```powershell
# 1. choose weights (writes runs\blend.json; keys = folder names in the Kaggle dataset)
python -m src.models.blend --m1 none --m2 ft2            # several runs: --m2 ft2,ft3 ; include M1: --m1 v2
# 2. local check of the M2 inference path (must print "M2 CHECK PASSED")
python -m src.models.predict --check-m2 ft2
# 3. build + upload the dataset (packs only runs with weight > 0 into m1/ and m2_<run>/)
python scripts\make_model_dataset.py
kaggle datasets version -p "kaggle\model_dataset" -m "<what changed>" --dir-mode zip
# 4. rebuild the notebook (embeds kneeprep + models code via %%writefile)
python scripts\build_submit_notebook.py
```
5. Wait 1–2 min after the upload. On Kaggle: open `submit_notebook` → **File → Import Notebook** (`kaggle\submit_notebook.ipynb`, replace) → **Input panel: update `rsna-knee-models` to the latest version** → accelerator **GPU T4** (not P100), Internet **off** → **Run All** (3 example studies; the log lists each model folder with its weight, e.g. `m2_ft2: 5 folds, weight 1.0`) → **Save Version → Save & Run All** → submit that version.

How inference works (`predict.py` `run_test()`): DICOM → kneeprep (forked CPU pool, started before CUDA) → M1 (if weighted) + one `M2Ensemble` per `m2*/` folder (frozen stem run once per image, shared by that run's folds; fp16) → weighted average by `blend.json` → training prevalence (`priors.json`) for any study that fails. Old-format `blend.json` (`{"m1": w, "m2": w}`) still works. Runtime with M1 + ft1 was ~25 min for the full hidden test (limit 9 h) → lots of headroom for more models. Each M2 run adds ~5 checkpoints of ~90 MB to the upload.

Other local checks: `python -m src.models.predict --check --run v2` (M1 feature parity).

---

## 9. Code changes 9–12 Oct

| File | Change |
|---|---|
| `src/models/blend.py` | Rewritten: `--m1 v2\|none --m2 run1,run2,...`; prints each model alone, equal weights, best grid (step 0.05 for ≤ 3 models, 0.1 above), per-label table; writes `runs\blend.json` = `{"weights": {"m1"\|"m2_<run>": w}, "runs": {...}, "auc_llm", "auc_human", "n_studies"}` |
| `scripts/make_model_dataset.py` | Reads `runs\blend.json` (or `--blend <file>`); removes old `m1/` / `m2*/` folders; copies each weighted run into `m1/` or `m2_<run>/`; `--run v2` = M1 only |
| `src/models/predict.py` | Any number of M2 runs, weighted by `blend.json` (`blend_weights()`) |
| `scripts/build_submit_notebook.py` | Description text only |
| `src/models/m2.py` | `--amp auto\|bf16\|fp16` (auto = bf16 on the laptop, fp16 + GradScaler on a T4); when `--folds` is a subset, log/metrics/config get a suffix (`train_f0.log`, `metrics_f0.csv`); `--collect` scores all `oof_fold*.csv` of a run together |
| `scripts/build_train_notebook.py`, `scripts/make_train_inputs.py` | Kaggle M2 training on **T4 ×2** (one fold per GPU, time-budgeted queue, benchmark mode; shards come from the prep-notebook outputs, labels from a small private dataset `rsna-knee-train-inputs`). **Built and dry-run tested but not used** — Omer prefers laptop training. Usable later if laptop time becomes the bottleneck |

⚠️ The `m2.py` / Kaggle-training files came in a separate zip (`kaggle_training_update.zip`) that may not have been applied. Check: `python -m src.models.m2 --help` lists `--amp` and `--collect` if it was. Either way, laptop training is unaffected (auto stays bf16). Fold-subset runs write `train_f0.log` with the update, `train.log` without.

---

## 10. In progress — fold-0 experiments (started 12 Oct)

Goal: find what makes the single model stronger, cheaply (one fold instead of five), then train only the winner on all folds.

```powershell
python -m src.models.m2 --folds 0 --name ft3a --frozen-blocks 3 --epochs 12 --eval-every 4 ; python -m src.models.m2 --folds 0 --name ft3b --img-size 252 --frozen-blocks 6 --epochs 10 --eval-every 5
```
- **ft3a** — more of the network trained (9 of 12 blocks) and longer. That direction gave ft1 → ft2 its +0.022.
- **ft3b** — higher resolution: the shards are 256 px but the model sees 224, losing detail for small structures (MCL, menisci). 252 = 18 × 14 (DINOv2 patch size). Positional embeddings are resized by timm; inference reads `img_size` from each checkpoint's config.
- Expected ~2.5–3 h each on Balanced. If ft3a runs out of GPU memory: add `--bs 1 --accum 16`.

**What to paste into the new chat:**
```powershell
Select-String -Path "runs\m2\ft2\train.log" -Pattern "fold 0 ep .*val auc_llm"
Select-String -Path "runs\m2\ft3a\*.log","runs\m2\ft3b\*.log" -Pattern "val auc_llm"
```
**How to judge:** compare each run's final fold-0 `val auc_llm` with ft2's fold-0 value at epoch 10. A gain under ~0.005 on one fold is within noise. Also glance at MCL / Lateral Meniscus. Then run the winner on all 5 folds (same arguments without `--folds 0`, new `--name`, e.g. `ft3`), and if both help, try them combined.

---

## 11. Plan / idea list (ranked by expected gain)

1. **Stronger single model** (§10): more trainable blocks / longer / 252 px, then the full 5-fold run of the winner → blend check with ft2 → submit if CV improves.
2. **Pseudo-labels to reduce label noise:** our models beat the report labels on human-label AUC, so train on a mix of the LLM labels and ft2's out-of-fold predictions (e.g. 0.5·LLM + 0.5·OOF, or replace only uncertain / strongly contradicted labels). Cheap: OOF predictions already exist. Needs a small change to `m2.py` to read a different label file.
3. **A genuinely different second model** for ensembling: another seed with a different slice/augmentation setup, different resolution, or a different backbone (e.g. DINOv2-B with more blocks frozen / gradient checkpointing; slower). Blending helps only if the models disagree.
4. **Label improvements** for the weakest labeler classes (Synovitis, Fracture): second prompt / larger model.
5. **Real higher resolution** (e.g. shards at 384–448 px) would mean re-running Stage 2 on Kaggle and ~4× compute — only if 252 px clearly helps.
6. Endgame (19–22 Oct): final ensemble from the best runs (OOF-weighted), runtime check, choose **2 final submissions** (best CV + best/safest LB) before **22 Oct 23:59 UTC**.

**Team merge (not optional):** Rishi and Arfeen must accept the competition rules on their own Kaggle accounts, then the team merge must be sent/accepted **before 15 Oct 23:59 UTC (16 Oct 05:29 IST)**. Check the team page shows all three.

---

## 12. Decisions & findings (keep these)

- `auc_llm` is the main CV metric; it has predicted the LB three times in a row (LB slightly above CV).
- The game plan's "aim for 0.946" is not realistic; public references are 0.66–0.81. We're at 0.889.
- Report labels match image-based truth only ~82% → label noise caps CV.
- Fine-tuning ≫ frozen features (+0.06). Initialising M2's head from M1 makes training stable and fast. Training more blocks for longer gave another +0.022 (ft1 → ft2).
- Blending near-identical models gives nothing (ft1 + ft2 + M1: +0.0002). M1 is retired.
- Random/site-mixed folds may inflate CV (another team saw ~0.05); for us CV and LB agree, so keep the current folds.
- The 58 human-labelled studies are in training with their true labels, spread over folds.
- Laptop training (Balanced mode) rather than Kaggle GPUs — Omer's choice to avoid the hassle.

## 13. Gotchas (don't repeat)

- PowerShell `Out-File` garbles non-Latin text → use the scripts' `--save` options.
- Ollama: default `num_ctx` 4096 is too small (use 8192); `OLLAMA_NUM_PARALLEL=2` gave no speed-up on 8 GB — keep it unset; `ollama ps` shows nothing until a model is loaded. Ollama "0.35.1" is newer than "0.9".
- Keep all code inside the repo (kneeprep once lived only in an extracted folder).
- Jupyter `%%writefile` cells cannot be empty → empty `__init__.py` files are created in the notebook's setup cell.
- Read StudyInstanceUIDs as strings (`dtype={"StudyInstanceUID": str}`).
- Don't unzip new code while a training run is going: DataLoader workers reload modules every epoch.
- Kaggle notebooks stay pinned to the dataset version they were created with → update the input after every `datasets version`; the first upload needs a minute or two of processing.
- Kaggle: use **T4**, not P100 (newer torch dropped it). T4 has no fast bf16 (the inference path uses fp16).
- Kaggle notebook outputs: ≤ 500 files per version, ~20 GB in `/kaggle/working`.
- The old dataset `omerraza2202/train-series-csv` has only the CSV — never attach it (it confused the data-path search).
- Windows: HF symlink warning harmless; `Expand-Archive` prints nothing on success; zip timestamps are UTC.
- Claude cannot reach the private GitHub repo (no "Claude" entry under GitHub *Installed GitHub Apps*; claude.ai Settings → Connectors → GitHub might allow it). Use zips instead.
- Closing PowerShell / shutting down kills training → re-activate the venv and set `PYTHONUTF8` in each new window.

## 14. Command cheat-sheet

```powershell
cd "D:\omer files\RSNA Knee Abnormality Detection - Kaggle"
C:\venvs\knee\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# train / evaluate
python -m src.models.m2 --smoke                                          # quick pipeline check
python -m src.models.m2 --folds 0 --name X --eval-start                  # one fold (start = M1 head score)
python -m src.models.m2 --name X --epochs 10 --frozen-blocks 6 --eval-every 5   # all folds (ft2 settings)
python -m src.models.m2 --name X --collect                               # score OOF of a run (if update applied)
python -m src.models.blend --m1 none --m2 ft2,X                          # blend weights -> runs\blend.json

# submission
python -m src.models.predict --check-m2 X
python scripts\make_model_dataset.py
kaggle datasets version -p "kaggle\model_dataset" -m "msg" --dir-mode zip
python scripts\build_submit_notebook.py

# logs
Select-String -Path "runs\m2\X\*.log" -Pattern "val auc_llm"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv -l 30

# git
git add . ; git commit -m "msg" ; git push
```
