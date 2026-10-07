# RSNA Knee Abnormality Detection — Game Plan

*Team: Omer Raza, Rishi Sharma, Arfeen Khan (Omer does all the work) · Written 30 Sep 2026 · Final submission deadline 22 Oct 2026*

---

## 0. How to use this document

This is the handoff document for continuing the work in a new chat. Attach it and say which stage to start. **Stage 1, the report labeler, comes first.** Every decision below was agreed on already; the "Open questions" section lists what is still undecided.

---

## 1. The competition in brief

| Item | Detail |
|---|---|
| Task | Multi-label classification at the study level: 12 binary findings per knee MRI study |
| Labels | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Metric | Macro-averaged ROC AUC across the 12 labels. Every label counts equally, so rare labels matter as much as common ones. Only ranking matters, so calibration does not |
| Test input | Images only. The `Report` column is **not** provided at test time. Test set is about 1,300 studies |
| Submission | Kaggle notebook, internet off, 9 h or less on CPU or GPU, writing `submission.csv` |
| External data and models | Allowed if freely and publicly available, including pretrained weights |
| Prevalence | May differ between train, public LB and private LB. AUC is fairly robust to this |

### Key dates (23:59 UTC)
- **15 Oct:** entry deadline and **team merger deadline**. Check on Kaggle that all three accounts are on the team.
- **22 Oct:** final submission deadline. Choose 2 final submissions.
- **5 Nov:** winners must submit code, a video and a method write-up. Keep the code clean and reproducible from the start.

---

## 2. What the data showed (EDA already done)

### train.csv (4,407 studies, 5.7 MB)
- **Only 58 studies have labels.** The other 4,349 have NaN in all 12 columns.
- **Every study has a report,** including the 58 labeled ones. Mean length is about 1,100 characters (range 52–4,743).
- Positive counts among the 58 (enriched; this is **not** real prevalence):

| ACL | MCL | Med Men | Lat Men | Med OA | Lat OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 9 | 26 | 23 | 15 | 11 | 21 | 35 | 27 | 12 | 19 | 18 |

- **Report languages** (rough keyword-based estimate): English ~54%, Spanish ~18%, Turkish ~12%, German ~6%, Bulgarian (Cyrillic) ~5%, Dutch ~3.5%, French ~1.5%.
- Many reports end in an **Impression / Conclusion** section, which is the easiest part to extract labels from.

### train_series.csv (24,371 series)
- Series per study: 3–14, median 5, mean 5.5.
- Planes: Sagittal 9,864 · Coronal 8,609 · Axial 5,898.
- **In training, `Fluid_Sensitive` equals `Fat_Suppression` for every one of the 24,371 series.** The competition says they may differ in test, so feed **both** to the model and never derive one from the other.
- Coverage by series type (share of studies that have at least one):

| Series type | Coverage |
|---|---|
| Axial fluid-sensitive / fat-suppressed | 100% |
| Coronal fluid-sensitive / FS | 96.4% |
| Sagittal fluid-sensitive / FS | 94.2% |
| Sagittal non-FS (T1/PD) | 96.8% |
| Coronal non-FS | 77.3% |
| Axial non-FS | 19.4% |
| All three fluid-sensitive planes present | 90.6% |

The model must handle a **missing plane** for about 10% of studies.

### Images
- 570 GB, about 820k DICOM files, one slice per file. Series usually have 20–45 slices (median 30).
- Mixed transfer syntaxes: uncompressed, JPEG Lossless and **JPEG 2000 (slow to decode)**. Decoding needs `pylibjpeg` and its plugins, and/or `gdcm`.
- DICOMs are reduced to 86 allowlisted tags. **Check early** whether ImagePositionPatient, ImageOrientationPatient, PixelSpacing and Laterality are present.

### What public notebooks use (Kaggle model list, best public LB)
DINOv2-small **0.946** · DINOv2-base 0.919 · BiomedCLIP 0.906 · EfficientNet-B0 0.81.
Two takeaways: a strong self-supervised 2D slice encoder is the right backbone family, and *small* is enough. Aim for **about 0.94+** macro AUC. These are notebook scores on the public LB, so they reflect whole pipelines, not the backbone alone.

---

## 3. Core strategy

1. **Labels come first.** The real training set is 4,349 reports, not 58 labels. The quality of the report-to-label step limits every model built on top of it.
2. **Use the 58 human labels to learn the definitions, not to train.** They show how the organizers define each finding, and they are the only ground truth available for checking anything.
3. **Frozen DINOv2-S baseline first, fine-tuning second.** Precompute per-slice embeddings once, then train a small study-level aggregator in minutes. That allows fast iteration on an 8 GB laptop GPU.
4. **Keep the 570 GB on Kaggle.** Convert it to about 35–40 GB of compact shards with **CPU** notebooks, which have no weekly quota, and download those.
5. **Save Kaggle GPU hours for submissions.** Train locally.
6. **Everything stays free.** No paid APIs and no free-tier commercial APIs, since those may keep medical report data.

---

## 4. What runs where, and why

| Part | Where | Why |
|---|---|---|
| Report → label extraction (LLM) | **Laptop** | Text only, unlimited time, no quota used, report data stays local |
| DICOM → training shards | **Kaggle CPU notebooks** | The raw data can't leave Kaggle; CPU notebooks don't use the 30 h/week GPU quota |
| Slice embedding extraction | **Laptop** | About 500k slices × DINOv2-S takes roughly 30 min on the RTX 5050 |
| Aggregator training, CV, experiments | **Laptop** | Free, unlimited, fast iteration |
| DINOv2-S fine-tuning | **Laptop** (overnight) | Fits in 8 GB with AMP, gradient checkpointing and slice subsampling |
| Submission notebook | **Kaggle GPU** | Required by the rules (notebook, internet off) |
| Larger LLM relabeling, if needed | Kaggle GPU (2×T4, internet on) | Free fallback; costs a few GPU hours |
| Heavy training, if the laptop is too slow | Kaggle GPU | Fallback only |

### Hardware (Omer's laptop)
Intel i7-13620H · **RTX 5050 8 GB** (Blackwell) · 24 GB RAM · 300+ GB free · Windows 11.

### Kaggle GPU budget
- 30 h/week on Omer's account, about **90 h** through 22 Oct. Use it first.
- Rishi's and Arfeen's accounts: **do not log into them yourself.** That is account sharing, which Kaggle's terms prohibit, and it can get the team disqualified. They may run notebooks themselves from their own accounts if ever needed. With this plan they shouldn't be needed.

| Use | Estimated GPU hours |
|---|---|
| Submission runs (≈8–12 runs × 1–3 h) | 20–35 |
| Optional larger-LLM relabel | 3–6 |
| Optional heavy fine-tune runs | 10–30 |
| Buffer | remainder |

---

## 5. Local environment setup (do once)

1. **Use WSL2 with Ubuntu**, not native Windows. DICOM decoders, DataLoader workers and llama.cpp/Ollama are all far more reliable there, and CUDA works inside WSL2.
   - `wsl --install -d Ubuntu`, then install the latest NVIDIA Windows driver. The WSL CUDA driver comes with it; don't install a separate Linux driver.
2. **PyTorch for Blackwell:** the RTX 5050 needs a PyTorch build for **CUDA 12.8 or newer** (the `cu128` wheels or later). Check with `torch.cuda.is_available()` and `torch.cuda.get_device_name()`.
3. Python environment (conda or venv), roughly: `torch torchvision timm transformers peft pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm numpy pandas pyarrow scikit-learn iterative-stratification opencv-python tqdm kaggle`.
4. **Ollama** (or llama.cpp) inside WSL2 for the LLM.
5. **Kaggle API:** put `kaggle.json` in `~/.kaggle/`. It's used to download the shard datasets and to upload code and weights as private datasets.
6. **Thermals:** use a cooling pad for overnight runs and watch for throttling (`nvidia-smi`).

### Suggested repository layout
```
rsna-knee/
├── data/
│   ├── raw/            # train.csv, train_series.csv, test*.csv, sample_submission.csv
│   ├── labels/         # LLM outputs (JSONL), final soft-label table (parquet)
│   ├── shards/         # downloaded preprocessed npz shards
│   └── embeddings/     # frozen DINOv2 features
├── src/
│   ├── kneeprep/       # SHARED DICOM → array preprocessing (used in Kaggle prep AND submission)
│   ├── labeler/        # prompts, runner, rules, evaluation vs the 58
│   ├── models/         # aggregator, fine-tune model, datasets, training loops
│   └── utils/
├── kaggle/
│   ├── prep_notebook.ipynb      # CPU: DICOM → shards
│   └── submit_notebook.ipynb    # GPU: test DICOM → predictions
├── configs/
└── README.md
```
Use git from day one. The winners' deadline requires reproducible training code.

---

## 6. Stage 1 — Report → label extraction (laptop)

### Model
**Qwen3.5-9B** (Alibaba, released March 2026, Apache 2.0). At 4-bit it needs about 6 GB, so it fits the 8 GB GPU. It handles all the report languages directly, with no translation step. Check the exact model tag in the Ollama library before pulling.
Settings: **thinking off, temperature 0, JSON output.** Expected speed is about 3–5 s per report, so about 4–6 h overnight for 4,407 reports.

### Splitting the 58 labeled studies
- **Dev set (~20):** used to write and refine label definitions and pick few-shot examples.
- **Held-out set (~38):** used only to score the labeler. Never look at these while tuning prompts.
- Stratify so each split has positives for every label. MCL has only 9 positives in total, so its score will be noisy.

### Prompt design
- A system prompt defining each of the 12 findings **precisely**, in clinical terms, with multilingual synonyms and explicit inclusion and exclusion rules.
- Instruction: "a finding not mentioned counts as absent."
- Per-report output (JSON):
```json
{
  "ACL": {"status": "present|absent|uncertain", "evidence": "<short quote or empty>"},
  "MCL": {...}, "...": {...}
}
```
- The evidence field makes the labels auditable and lets you catch systematic mistakes.
- Mapping: present = 1, absent = 0, uncertain = 0.5 (soft label). The 58 keep their true labels.

### Definition questions to settle using the dev set
- ACL/MCL "injury": does it include sprain, partial tear, mucoid degeneration, or old reconstruction?
- Meniscus: tear only, or also degeneration ("meniscopatía", intrasubstance signal), prior meniscectomy, or a discoid meniscus without tear?
- OA per compartment: which cartilage grades or osteophyte descriptions count? How to handle "chondromalacia patellae" for PF OA?
- Effusion: does "small / physiological amount of fluid" count?
- Synovitis vs effusion-synovitis vs Hoffa's fat pad edema.
- Contusion vs bone marrow edema from OA or osteonecrosis vs fracture. Does a fracture also imply contusion?
- Fracture: does an osteochondral lesion, stress reaction or insufficiency fracture count?

### Evaluation
Compare per-label accuracy, sensitivity and specificity, and Cohen's κ against the held-out 38. Also count the uncertain outputs per label.

### Free fixes if some labels are weak
1. Rewrite the definitions for weak labels, add dev examples, or ask one label at a time.
2. **Run twice with different prompts.** Keep labels where the runs agree; set disagreements to 0.5.
3. **Keyword and negation rules** for Effusion, Baker's and Fracture (predictable vocabulary in every language), used to cross-check or override.
4. **A larger open-weight model on Kaggle 2×T4 (32 GB VRAM)** in a notebook with internet on. This is the strongest free upgrade and costs a few GPU hours.
5. Have the LLM translate the remaining disputed reports and label a few hundred by hand.

**Stage 1 output:** `labels/soft_labels.parquet` with StudyInstanceUID, 12 soft labels, and a source flag (human or LLM).

---

## 7. Stage 2 — DICOM → training shards (Kaggle CPU)

### First, test locally
Pull the 3 example test studies (`test_series/`) with `kaggle competitions download -c <slug> -f <path>` and develop `kneeprep` on them. Check every transfer syntax decodes and see which of the 86 tags exist.

### Which series to keep per study (about 4)
- The fluid-sensitive series in each plane (sagittal, coronal, axial). If there are duplicates, keep the one with the most slices or the best in-plane resolution.
- The sagittal non-FS series (T1/PD; useful for fracture and OA).
- Record which series types are missing.

### Per-series preprocessing (`kneeprep`, shared with the submission notebook)
1. Read all slices and decode with pydicom + pylibjpeg/gdcm.
2. **Sort slices by spatial position:** project ImagePositionPatient onto the slice normal from ImageOrientationPatient. Fall back to InstanceNumber if those tags are missing.
3. Apply RescaleSlope and RescaleIntercept if present.
4. **Resample to 32 slices** along the slice axis (interpolation, or uniform index sampling).
5. Intensity: clip each series to its 0.5–99.5 percentiles and scale to uint8.
6. Resize in-plane to **256×256**, padding to keep the aspect ratio.
7. Optional: normalize left vs right knee (horizontal flip) if the Laterality or orientation tags allow it. Otherwise rely on flip augmentation.
8. Save metadata per series: plane, Fluid_Sensitive, Fat_Suppression, original slice count, PixelSpacing, SliceThickness.

### Output and size
- One compressed `.npz` per study, plus a metadata parquet.
- Estimate: 4,407 studies × ~4 series × 32 × 256 × 256 bytes ≈ **35–40 GB** uncompressed, less compressed.
- A notebook's output is capped at roughly 20 GB (check the current limit), so **split the StudyInstanceUIDs into 2–3 chunks**, run one CPU notebook per chunk, and save each output as a **private Kaggle Dataset**.
- Parallelize decoding with multiprocessing across the notebook's CPU cores. **Time the first 50 studies** before launching full runs, since JPEG 2000 may be slow.
- Download the datasets to `data/shards/` with the Kaggle API.

---

## 8. Stage 3 — Modelling (laptop)

### Slice input format
Grayscale → 3 channels (replicated), ImageNet normalization, 224×224 (random crop from 256 in training, center crop at inference). DINOv2 uses 14-pixel patches, so 224 gives a 16×16 patch grid.

### M1 — frozen DINOv2-S with an attention aggregator (baseline)
1. **Extract embeddings once:** for every slice, keep CLS plus the mean of the patch tokens (768-d, fp16). About 500k slices take roughly 30 min and a few hundred MB.
2. **Aggregator:** each slice becomes a token:
   `token = Linear(embedding) + series_type_embedding(plane × fluid × FS) + slice_position_embedding`.
   All tokens from all series in the study go through a small transformer (2–4 layers, d≈256), with a padding/missing mask. Then attention pooling (or 12 per-label query tokens) → 12 logits.
3. **Training:** BCE with soft targets. Consider a label-wise positive weight for rare labels. AdamW with a cosine schedule. Regularize with slice dropout and whole-series dropout, which also teaches robustness to missing planes. Each run takes minutes.
4. Use this to test design choices cheaply: which series to include, pooling type, soft vs hard labels, using the patch tokens.

### M2 — fine-tuned DINOv2-S (main model)
- Train end-to-end: either LoRA (rank 8–16 on attention q/k/v) or unfreeze the last 3–4 blocks.
- To fit in 8 GB: **bf16 AMP**, gradient checkpointing, **sample 16 of 32 slices per series** during training (all slices at inference), and 1–2 studies per step with gradient accumulation.
- Augmentation: random resized crop, horizontal flip, brightness/contrast/gamma, small rotations, slice-offset jitter.
- Initialize the aggregator from M1. Run overnight per fold.

### M3 — diversity model (optional, if time allows)
Frozen BiomedCLIP vision encoder, or frozen DINOv2-B, with the same aggregator.

### Ensembling
**Rank-average** the fold models and model types, since AUC only depends on ranking. Per-label weights should come from CV only, and only if the gain is clear. With noisy labels, per-label weights overfit easily.

---

## 9. Validation

- **5-fold multilabel-stratified CV** (iterative stratification) on the soft labels. Balance the report languages across folds as well.
- For each experiment, report:
  1. **OOF macro AUC against the LLM labels** (the main signal, using hard labels at 0.5 for scoring), and
  2. **OOF macro AUC against the 58 true labels** (noisy, but grounded in the real definitions).
- Label noise caps the achievable CV score. If CV plateaus, improving the labels usually helps more than changing the architecture.
- Track per-label AUC. Macro AUC is dragged down by the weakest labels (likely MCL, Fracture, Synovitis).
- Correlate CV with the public LB after each submission. Don't overfit to the public LB, which uses only part of the test set and may have different prevalence.

---

## 10. Stage 4 — Submission notebook (Kaggle GPU)

- Inputs, all as private Kaggle Datasets: competition data, the `kneeprep` module, **offline wheels** for pylibjpeg/openjpeg/gdcm (and any packages not preinstalled), DINOv2-S weights, and trained checkpoints.
- Pipeline: read `test_series.csv` → select series → `kneeprep` → model forward passes (fp16; T4 does not support bf16) → rank-safe probabilities → `submission.csv` with the exact column order from `sample_submission.csv`.
- **Time budget:** 1,300 studies in 9 h is about 25 s per study. DICOM decoding is the bottleneck, so decode with a multiprocessing pool. Log the time per study.
- **Handle failures robustly:** if a series fails to decode or a plane is missing, mask it. If a whole study fails, output the training prevalence for it. Never crash.
- **Make the first submission early** (by about 8 Oct), even with a weak model, to test the whole pipeline and its runtime.

---

## 11. Timeline

| Dates | Work | Done when |
|---|---|---|
| **1–4 Oct** | WSL2, PyTorch cu128, Ollama setup · Stage 1 labeler: prompts, dev iteration, held-out score · Stage 2: build `kneeprep` on the 3 example studies, time it on Kaggle, launch CPU prep chunks | Labeler scored on the 38; prep notebooks running |
| **5–8 Oct** | Full LLM labeling run (overnight) · download shards · extract embeddings · M1 CV · **first submission** | End-to-end submission scores on the LB |
| **9–15 Oct** | Label fixes for weak labels · M2 fine-tuning (overnight folds) · improve series selection and augmentation · **confirm the team roster before 15 Oct** | M2 beats M1 in CV |
| **16–22 Oct** | Optional M3 · rank ensemble · final runtime checks · choose 2 final submissions (e.g., best CV and best blend) | Submitted by 22 Oct 23:59 UTC |
| by 5 Nov | If placing: clean code, method write-up, video | — |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Systematic LLM label errors, especially from definition mismatch | Held-out 38 score, evidence quotes, two-prompt agreement, keyword rules, free larger-model relabel |
| Too few positives for rare labels (MCL, Baker's) | Soft labels, positive weighting, check per-label AUC, don't over-tune on the 58 |
| JPEG 2000 decoding too slow (prep or inference) | Time it early, multiprocessing, test gdcm vs openjpeg, reduce slice count if needed |
| Submission exceeds 9 h | Log per-study time on the full test run; use fewer TTA passes or models |
| Missing planes / protocol variety across sites | Series-type embeddings + missing-series masking + series dropout |
| `Fluid_Sensitive` ≠ `Fat_Suppression` at test time | Feed both columns separately to the model |
| 8 GB VRAM limits | Frozen-embedding baseline, LoRA, AMP, checkpointing, slice subsampling |
| Laptop thermal throttling / crashes on overnight runs | Cooling pad, save checkpoints every epoch, make runs resumable |
| Public LB overfitting / prevalence shift | Trust CV + the 58, pick one CV-based final submission |
| Rule violations | No account sharing; only freely available weights (DINOv2 Apache 2.0, Qwen3.5 Apache 2.0, BiomedCLIP MIT); team roster fixed by 15 Oct |

---

## 13. Open questions (to settle as work proceeds)

- Exact label definitions (Section 6 list), decided from the dev 20.
- Which of the 86 DICOM tags are available (affects slice sorting and laterality handling).
- Kaggle's current notebook output size cap and CPU session limit.
- Whether patch-token features beyond CLS + mean help M1.
- Whether the Qwen3.5-9B labeler needs the larger-model fallback for any label.

---

## 14. Next action

**Start Stage 1:** set up WSL2 + Ollama + Qwen3.5-9B, split the 58 into dev and held-out sets, write the label-definition prompt, and score the labeler on the held-out 38. At the same time, start writing `kneeprep` against the 3 example test studies.
