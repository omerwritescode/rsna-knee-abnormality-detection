"""Build kaggle/prep_notebook.ipynb with the current src/kneeprep/core.py embedded.

Run from the project root after any change to kneeprep:
    python scripts/build_prep_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = (ROOT / "src" / "kneeprep" / "core.py").read_text(encoding="utf-8")
OUT = ROOT / "kaggle" / "prep_notebook.ipynb"

MD_INTRO = """# RSNA Knee — Stage 2: DICOM → training shards (CPU notebook)

**Settings:** Accelerator **None** · Internet **On** · competition data attached.

**1. Timing run (interactive):** leave `TIMING_ONLY = True` and click **Run All**. It processes 50 studies, \
reports speed and output size, suggests the number of chunks, and shows an orientation check image.

**2. Real runs (committed):** set `TIMING_ONLY = False` and `CHUNK_ID`, then **Save Version → Save & Run All**. \
Use one notebook per chunk (*Copy & Edit*, change `CHUNK_ID`), so each chunk's output can be downloaded separately.

**Output** in `/kaggle/working`: `shards_c<K>_p<NN>.tar` archives (Kaggle keeps at most 500 output files, so the \
per-study `<StudyInstanceUID>.npz` files are packed into ~2 GB tars; each npz has keys `sag_fl`, `cor_fl`, `ax_fl`, \
`sag_t1`, uint8 `[32, 256, 256]`), `study_meta_chunk<K>.csv`, `series_meta_chunk<K>.csv`, and `kneeprep.py`."""

CELL_INSTALL = """import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "pydicom>=3", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"], check=True)"""

CELL_CONFIG = """import os

CHUNK_ID = 0           # which chunk this run processes: 0 .. N_CHUNKS-1
N_CHUNKS = 2           # total number of chunks (the timing run suggested 2)
TIMING_ONLY = True     # True: only N_TIMING studies, then report speed/size + orientation check
N_TIMING = 50
MAX_OUTPUT_GB = 19.0   # stop cleanly before Kaggle's ~20 GB /kaggle/working limit
PART_GB = 2.0          # size of each output .tar part
WORKERS = os.cpu_count()"""

CELL_SETUP = """import sys, time, math, importlib
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, "/kaggle/working")
import kneeprep as kp
importlib.reload(kp)

# Competition data: the folder with train_series.csv AND the train_series/ image folder
ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    has_images = "train_series" in dirs
    dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    if "train_series.csv" in files and has_images and ROOT is None:
        ROOT = root
assert ROOT, "Competition data not found. Add it via Add Input -> Competitions."
print("Data root:", ROOT)

series_df = pd.read_csv(f"{ROOT}/train_series.csv")
cols = ["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
groups = {st: g[cols].to_dict("records") for st, g in series_df.groupby("StudyInstanceUID")}
studies = sorted(groups)
chunk = [s for i, s in enumerate(studies) if i % N_CHUNKS == CHUNK_ID]

TMP = "/tmp/shards"          # per-study npz land here briefly, then go into the tar parts
os.makedirs(TMP, exist_ok=True)
todo = chunk[:N_TIMING] if TIMING_ONLY else chunk
print(f"{len(studies)} studies total | chunk {CHUNK_ID}/{N_CHUNKS}: {len(chunk)} studies | "
      f"to process now: {len(todo)} | workers: {WORKERS} | timing only: {TIMING_ONLY}")"""

CELL_RUN = """import tarfile

def save_meta(study_rows, series_rows):
    pd.DataFrame(study_rows).to_csv(f"/kaggle/working/study_meta_chunk{CHUNK_ID}.csv", index=False)
    pd.DataFrame(series_rows).to_csv(f"/kaggle/working/series_meta_chunk{CHUNK_ID}.csv", index=False)

class TarParts:
    \"\"\"Append finished npz files to shards_c<K>_p<NN>.tar, starting a new part every PART_GB.\"\"\"
    def __init__(self):
        self.part, self.tar, self.total, self.where = -1, None, 0, {}
    def _roll(self):
        if self.tar:
            self.tar.close()
        self.part += 1
        self.path = f"/kaggle/working/shards_c{CHUNK_ID}_p{self.part:02d}.tar"
        self.tar = tarfile.open(self.path, "w")
    def add(self, npz, study):
        if self.tar is None or os.path.getsize(self.path) > PART_GB * 1e9:
            self._roll()
        self.tar.add(npz, arcname=f"{study}.npz")
        self.total += os.path.getsize(npz)
        self.where[study] = self.path
        os.remove(npz)
    def close(self):
        if self.tar:
            self.tar.close()

tasks = [(f"{ROOT}/train_series/{s}", groups[s], f"{TMP}/{s}.npz") for s in todo]
study_rows, series_rows, parts = [], [], TarParts()
every = 10 if TIMING_ONLY else 100
t0 = time.time()

with mp.get_context("fork").Pool(WORKERS) as pool:
    for i, (st, se) in enumerate(pool.imap_unordered(kp.process_and_save, tasks, chunksize=2), 1):
        if st.get("npz_bytes", 0) > 0:
            parts.add(f"{TMP}/{st['StudyInstanceUID']}.npz", st["StudyInstanceUID"])
        study_rows.append(st)
        series_rows.extend(se)
        if i % every == 0 or i == len(tasks):
            el = time.time() - t0
            n_err = sum(bool(r.get("error")) for r in study_rows)
            print(f"{i}/{len(tasks)} studies | {el/60:.1f} min | {i/el*60:.1f} studies/min | "
                  f"output {parts.total/1e9:.2f} GB in {parts.part + 1} tar part(s) | study errors {n_err}", flush=True)
            save_meta(study_rows, series_rows)
        if parts.total > MAX_OUTPUT_GB * 1e9:
            print(f"STOPPED: output reached {MAX_OUTPUT_GB} GB. Re-run the rest with more chunks.")
            break

parts.close()
save_meta(study_rows, series_rows)
elapsed = time.time() - t0
n_saved = len(parts.where)
print(f"\\nDone: {len(study_rows)} studies in {elapsed/60:.1f} min | saved {n_saved} | "
      f"chunk size {len(chunk)} | {'ALL SAVED' if n_saved == len(todo) else 'SOME MISSING - check errors'}")
print("Output files:", sorted(f for f in os.listdir("/kaggle/working") if not f.startswith(".")))"""

CELL_SUMMARY = """st_df, se_df = pd.DataFrame(study_rows), pd.DataFrame(series_rows)
n = max(len(st_df), 1)
print(f"Wall time: {elapsed/n:.2f} s/study with {WORKERS} workers")
print("Study errors:", int((st_df.error != "").sum()))
if (st_df.error != "").any():
    print(st_df[st_df.error != ""].error.value_counts().head(10).to_string())
if len(se_df):
    errs = se_df[se_df.error != ""]
    print("Series errors:", len(errs))
    if len(errs):
        print(errs.error.value_counts().head(10).to_string())
    ok = se_df[se_df.error == ""]
    print("\\nKinds per study:\\n" + st_df.kinds.value_counts().head(8).to_string())
    print("\\nLaterality:", st_df.laterality.replace("", "unknown").value_counts().to_dict())
    print("Slice order source:", ok.sorted_by.value_counts().to_dict())
    print("Plane from orientation tags != CSV plane:", int((ok.plane_iop != ok.Anatomical_Plane).sum()))
    print("Orientation ops (transposed, flip_lr, flip_ud):",
          ok.groupby(["transposed", "flip_lr", "flip_ud"]).size().to_dict())
    print("Unique slices per series:", ok.n_unique.describe().round(1).to_dict())
    print("Failed slice reads:", int(ok.n_failed.sum()))

mb = st_df.npz_bytes.mean() / 1e6
total_gb = mb * len(studies) / 1e3
print(f"\\nMean output: {mb:.2f} MB/study -> all {len(studies)} studies ≈ {total_gb:.1f} GB")
print(f"This chunk ({len(chunk)} studies): ≈ {mb*len(chunk)/1e3:.1f} GB, ≈ {elapsed/n*len(chunk)/3600:.2f} h")
print(f"Suggested N_CHUNKS (≤ {MAX_OUTPUT_GB-2:.0f} GB per chunk): {max(1, math.ceil(total_gb/(MAX_OUTPUT_GB-2)))}")"""

CELL_VIEW = """import io
import matplotlib.pyplot as plt

def load_from_tar(study):
    with tarfile.open(parts.where[study]) as t:
        with np.load(io.BytesIO(t.extractfile(f"{study}.npz").read())) as z:
            return {k: z[k] for k in z.files}

show = [s for s in st_df.StudyInstanceUID if s in parts.where][:3]
kinds = list(kp.KINDS)
fig, axes = plt.subplots(len(show), len(kinds), figsize=(3.2 * len(kinds), 3.4 * len(show)), squeeze=False)
for i, s in enumerate(show):
    arr = load_from_tar(s)
    lat = st_df.set_index("StudyInstanceUID").laterality.get(s, "") or "?"
    for j, k in enumerate(kinds):
        ax = axes[i][j]
        ax.axis("off")
        if k in arr:
            ax.imshow(arr[k][arr[k].shape[0] // 2], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"...{s[-6:]} ({lat}) {k}" + ("" if k in arr else "  MISSING"), fontsize=9)
plt.tight_layout()
plt.savefig("/kaggle/working/orientation_check.png", dpi=80)
plt.show()
print("Expected: sagittal = femur at TOP, patella/front of knee on the LEFT;"
      " coronal = femur at TOP; axial = patella at TOP.")"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md(MD_INTRO),
    md("## 1. Install DICOM decoders"), code(CELL_INSTALL),
    md("## 2. kneeprep module (generated from `src/kneeprep/core.py`; edit there, then rebuild)"),
    code("%%writefile kneeprep.py\n" + CORE),
    md("## 3. Configuration"), code(CELL_CONFIG),
    md("## 4. Locate data and pick this chunk's studies"), code(CELL_SETUP),
    md("## 5. Process (parallel, packs output into tar parts, stops before the output limit)"), code(CELL_RUN),
    md("## 6. Summary: speed, size, errors, geometry"), code(CELL_SUMMARY),
    md("## 7. Orientation check (middle slice of each series)"), code(CELL_VIEW),
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
