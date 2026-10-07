# =============================================================================
# RSNA Knee — DICOM inspection + decode timing (Kaggle CPU notebook)
#
# Notebook settings: Accelerator = None, Internet = ON,
# competition data attached (Add Input -> the competition).
# Paste this whole file into one cell and run it. It takes about 5-15 min.
#
# Answers three open questions from the game plan:
#   1. Which of the 86 allowlisted tags exist (IPP/IOP, PixelSpacing, Laterality...)
#   2. Transfer-syntax mix and decode speed per syntax and per backend
#   3. Rough wall-clock estimate for the full Stage 2 shard conversion
# It also saves offline wheels for the submission notebook later.
# =============================================================================

import subprocess, sys

PKGS = ["pydicom>=3", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *PKGS], check=True)
# Wheels for the internet-off submission notebook (save this output as a dataset later)
subprocess.run([sys.executable, "-m", "pip", "download", "-q", "-d", "/kaggle/working/wheels", *PKGS], check=True)

import glob, io, os, random, time, collections, json
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import pixel_array
from pydicom.uid import UID
from pydicom.multival import MultiValue

print("pydicom", pydicom.__version__, "| CPUs:", os.cpu_count())

# --- locate competition data (depth-limited: never glob the 820k-file tree) ---
cands = []
for depth in range(1, 4):
    cands += glob.glob("/kaggle/input/" + "*/" * depth + "train_series.csv")
assert cands, "Competition data not found. Attach it via Add Input."
ROOT = os.path.dirname(cands[0])
print("Data root:", ROOT)

series_df = pd.read_csv(f"{ROOT}/train_series.csv")
N_SAMPLE_STUDIES = 40
random.seed(0)
sample_studies = random.sample(list(series_df.StudyInstanceUID.unique()), N_SAMPLE_STUDIES)


def iop_plane(iop):
    """Plane from ImageOrientationPatient (LPS): normal along x/y/z -> sag/cor/ax."""
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:6], float)
    return ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(np.cross(r, c))))]


def g(ds, kw):
    v = ds.get(kw, None)
    if v is None:
        return None
    return list(v) if isinstance(v, MultiValue) else v


# --- 1. header pass -----------------------------------------------------------
rows, tag_counter, n_headers = [], collections.Counter(), 0
for st in sample_studies:
    for r in series_df[series_df.StudyInstanceUID == st].itertuples():
        sdir = f"{ROOT}/train_series/{st}/{r.SeriesInstanceUID}"
        try:
            files = sorted(os.listdir(sdir))
        except FileNotFoundError:
            rows.append(dict(study=st, series=r.SeriesInstanceUID, error="missing dir"))
            continue
        picks = list(dict.fromkeys([files[0], files[len(files) // 2], files[-1]]))
        ds0 = None
        for f in picks:
            try:
                ds = pydicom.dcmread(os.path.join(sdir, f), stop_before_pixels=True)
            except Exception as e:
                continue
            n_headers += 1
            tag_counter.update(el.keyword or str(el.tag) for el in ds)
            ds0 = ds0 or ds
        if ds0 is None:
            rows.append(dict(study=st, series=r.SeriesInstanceUID, error="unreadable"))
            continue
        iop = g(ds0, "ImageOrientationPatient")
        rows.append(dict(
            study=st, series=r.SeriesInstanceUID, n_files=len(files),
            plane_csv=r.Anatomical_Plane, plane_iop=iop_plane(iop) if iop else None,
            fluid=r.Fluid_Sensitive, fs=r.Fat_Suppression,
            ts=str(ds0.file_meta.TransferSyntaxUID), ts_name=UID(ds0.file_meta.TransferSyntaxUID).name,
            rows=g(ds0, "Rows"), cols=g(ds0, "Columns"), frames=g(ds0, "NumberOfFrames"),
            bits=g(ds0, "BitsAllocated"), photometric=g(ds0, "PhotometricInterpretation"),
            pixel_spacing=g(ds0, "PixelSpacing"), slice_thickness=g(ds0, "SliceThickness"),
            spacing_between=g(ds0, "SpacingBetweenSlices"),
            laterality=g(ds0, "Laterality") or g(ds0, "ImageLaterality"),
            manufacturer=g(ds0, "Manufacturer"), field_strength=g(ds0, "MagneticFieldStrength"),
            series_desc=g(ds0, "SeriesDescription"), has_ipp="ImagePositionPatient" in ds0,
            has_iop=iop is not None, has_instance_number="InstanceNumber" in ds0,
            rescale_slope=g(ds0, "RescaleSlope"), rescale_intercept=g(ds0, "RescaleIntercept"),
        ))

info = pd.DataFrame(rows)
info.to_csv("/kaggle/working/series_sample.csv", index=False)
tags = pd.DataFrame(
    [(k, v, round(100 * v / n_headers, 1)) for k, v in tag_counter.most_common()],
    columns=["keyword", "count", "pct_of_headers"],
)
tags.to_csv("/kaggle/working/tag_presence.csv", index=False)

pd.set_option("display.width", 200, "display.max_rows", 200, "display.max_columns", 50)
print(f"\n=== {len(info)} series from {N_SAMPLE_STUDIES} studies, {n_headers} headers read ===")
print(f"\nDistinct tags seen: {len(tags)}")
print(tags.to_string(index=False))

print("\n=== Key-tag availability (share of series) ===")
for c in ["has_ipp", "has_iop", "has_instance_number"]:
    print(f"{c:22s} {info[c].mean():.1%}")
for c in ["pixel_spacing", "slice_thickness", "spacing_between", "laterality", "rescale_slope"]:
    print(f"{c:22s} {info[c].notna().mean():.1%}")
print("\nLaterality values:", info.laterality.value_counts(dropna=False).to_dict())
if info.plane_iop.notna().any():
    m = info.plane_iop.notna()
    print(f"IOP plane agrees with CSV plane: {(info.plane_iop[m] == info.plane_csv[m]).mean():.1%}")

print("\n=== Transfer syntaxes (by series / by files) ===")
print(info.groupby("ts_name").agg(series=("series", "size"), files=("n_files", "sum")))
print("\nSlices per series:", info.n_files.describe().round(1).to_dict())
print("Matrix sizes:", (info.rows.astype(str) + "x" + info.cols.astype(str)).value_counts().head(10).to_dict())
print("Frames per file:", info.frames.value_counts(dropna=False).to_dict())
print("Manufacturers:", info.manufacturer.value_counts(dropna=False).to_dict())
print("Field strength:", info.field_strength.value_counts(dropna=False).to_dict())
print("Photometric:", info.photometric.value_counts(dropna=False).to_dict())


# --- 2. decode timing ---------------------------------------------------------
PLUGINS = ["", "pylibjpeg", "gdcm"]  # "" = pydicom's default choice
FILES_PER_TS = 20
timing = []
for ts_name, grp in info.dropna(subset=["ts"]).groupby("ts_name"):
    paths = []
    for r in grp.sample(min(len(grp), 5), random_state=0).itertuples():
        sdir = f"{ROOT}/train_series/{r.study}/{r.series}"
        fs = sorted(os.listdir(sdir))
        paths += [os.path.join(sdir, f) for f in random.sample(fs, min(len(fs), 4))]
    for p in paths[:FILES_PER_TS]:
        t0 = time.perf_counter()
        with open(p, "rb") as fh:
            data = fh.read()
        t_io = time.perf_counter() - t0
        for plugin in PLUGINS:
            try:
                t0 = time.perf_counter()
                arr = pixel_array(io.BytesIO(data), decoding_plugin=plugin)
                timing.append(dict(ts_name=ts_name, plugin=plugin or "default", ok=True,
                                   t_io=t_io, t_decode=time.perf_counter() - t0,
                                   shape=str(arr.shape), dtype=str(arr.dtype), mb=len(data) / 1e6))
            except Exception as e:
                timing.append(dict(ts_name=ts_name, plugin=plugin or "default", ok=False,
                                   t_io=t_io, err=f"{type(e).__name__}: {str(e)[:80]}"))

tdf = pd.DataFrame(timing)
tdf.to_csv("/kaggle/working/decode_timing.csv", index=False)
ok = tdf[tdf.ok]
summary = (ok.groupby(["ts_name", "plugin"])
             .agg(n=("t_decode", "size"), ms_decode=("t_decode", lambda x: 1000 * x.mean()),
                  ms_io=("t_io", lambda x: 1000 * x.mean()), mb=("mb", "mean"))
             .round(1))
print("\n=== Decode timing per file ===")
print(summary)
fails = tdf[~tdf.ok].groupby(["ts_name", "plugin"]).err.first()
if len(fails):
    print("\nFailures:\n", fails)


# --- 3. full-conversion estimate ------------------------------------------------
# Series kept per study: fluid-sensitive sag/cor/ax + sagittal non-FS (max 4)
s = series_df.assign(
    kind=np.where(series_df.Fluid_Sensitive == 1, "FL_" + series_df.Anatomical_Plane,
                  "NF_" + series_df.Anatomical_Plane))
keep_kinds = {"FL_Sagittal", "FL_Coronal", "FL_Axial", "NF_Sagittal"}
kept_series = s[s.kind.isin(keep_kinds)].groupby("StudyInstanceUID").kind.nunique().sum()
mean_slices = info.n_files.mean()
best = summary.reset_index().sort_values("ms_decode").groupby("ts_name").first()
file_share = info.groupby("ts_name").n_files.sum() / info.n_files.sum()
ms_per_file = float(sum(file_share.get(t, 0) * (best.loc[t, "ms_decode"] + best.loc[t, "ms_io"])
                        for t in best.index))
total_files = kept_series * mean_slices
hours = total_files * ms_per_file / 1000 / 3600 / os.cpu_count()
print("\n=== Stage 2 estimate ===")
print(f"Kept series: {kept_series:,}  x  {mean_slices:.1f} slices = ~{total_files:,.0f} files")
print(f"Weighted best-backend cost: {ms_per_file:.1f} ms/file (IO + decode)")
print(f"=> ~{hours:.1f} h of decode on {os.cpu_count()} cores with perfect parallelism "
      f"(add ~30-50% for resize/IO overhead)")
print("Best backend per syntax:", best.plugin.to_dict())

json.dump(dict(kept_series=int(kept_series), mean_slices=float(mean_slices),
               ms_per_file=ms_per_file, est_hours=hours, cpus=os.cpu_count()),
          open("/kaggle/working/estimate.json", "w"), indent=1)
print("\nSaved: series_sample.csv, tag_presence.csv, decode_timing.csv, estimate.json, wheels/")
