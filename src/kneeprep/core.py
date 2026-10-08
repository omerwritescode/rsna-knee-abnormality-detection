"""kneeprep: shared DICOM -> array preprocessing for RSNA Knee Abnormality Detection.

Used by BOTH the Kaggle prep notebook (training shards) and the submission notebook
(test studies), so train and test always go through identical preprocessing.
Edit this file only, then rebuild the notebooks (scripts/build_prep_notebook.py).

Output per series: uint8 volume [N_SLICES, SIZE, SIZE] in a canonical patient orientation
(DICOM patient coordinates are LPS: +x = patient left, +y = posterior, +z = superior):

  sagittal : slices patient right -> left | image up = superior | image right = posterior
  coronal  : slices anterior -> posterior | image up = superior | image right = patient left
  axial    : slices superior -> inferior  | image up = anterior | image right = patient left

Laterality is NOT normalized here (a left knee stays a left knee). Whether to mirror left
knees is a training-time choice; per-study laterality is recorded in the metadata.
"""
from __future__ import annotations

import os
from collections import Counter

import cv2
import numpy as np

N_SLICES = 32
SIZE = 256
PLANES = ("Sagittal", "Coronal", "Axial")

# plane -> ((image-right axis, sign), (image-down axis, sign), (slice axis, order sign)), LPS axes 0/1/2
CANON = {
    "Sagittal": ((1, +1), (2, -1), (0, +1)),
    "Coronal": ((0, +1), (2, -1), (1, +1)),
    "Axial": ((0, +1), (1, +1), (2, -1)),
}

# kind -> (plane, Fluid_Sensitive, preferred Fat_Suppression)
KINDS = {
    "sag_fl": ("Sagittal", 1, 1),
    "cor_fl": ("Coronal", 1, 1),
    "ax_fl": ("Axial", 1, 1),
    "sag_t1": ("Sagittal", 0, 0),
}

DUP_TOL_MM = 0.01  # slices closer than this along the normal count as the same position


# ----------------------------------------------------------------------------- DICOM I/O
def _get(ds, kw, default=None):
    v = ds.get(kw, None)
    if v is None or v == "":
        return default
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and kw in ("EchoNumbers",):
        return v[0] if len(v) else default
    return v


def _num(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_slice(path, with_pixels=True):
    """Read one DICOM file into a plain dict (geometry, a few tags, optional float32 pixels)."""
    import pydicom  # imported lazily so the geometry code can be tested without pydicom

    ds = pydicom.dcmread(path, force=True, stop_before_pixels=not with_pixels)
    iop = _get(ds, "ImageOrientationPatient")
    ipp = _get(ds, "ImagePositionPatient")
    sp = _get(ds, "PixelSpacing")
    lat = str(_get(ds, "Laterality", "") or _get(ds, "ImageLaterality", "") or "").strip().upper()[:1]
    rec = dict(
        iop=np.asarray(iop, float) if iop is not None and len(iop) == 6 else None,
        ipp=np.asarray(ipp, float) if ipp is not None and len(ipp) == 3 else None,
        spacing=(_num(sp[0], 1.0), _num(sp[1], 1.0)) if sp is not None and len(sp) == 2 else (1.0, 1.0),
        instance=int(_num(_get(ds, "InstanceNumber", 0), 0)),
        echo=int(_num(_get(ds, "EchoNumbers", 1), 1)),
        laterality=lat if lat in ("L", "R") else "",
        patient_id=str(_get(ds, "PatientID", "")),
        manufacturer=str(_get(ds, "Manufacturer", "")),
        field_strength=_num(_get(ds, "MagneticFieldStrength")),
        slice_thickness=_num(_get(ds, "SliceThickness")),
        spacing_between=_num(_get(ds, "SpacingBetweenSlices")),
    )
    if with_pixels:
        arr = ds.pixel_array
        if arr.ndim == 3:
            if int(_num(_get(ds, "SamplesPerPixel", 1), 1)) == 3:
                arr = arr.mean(-1)
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                raise ValueError(f"multi-frame file with {arr.shape[0]} frames is not supported")
        arr = arr.astype(np.float32)
        slope = _num(_get(ds, "RescaleSlope", 1), 1.0)
        inter = _num(_get(ds, "RescaleIntercept", 0), 0.0)
        if slope != 1.0 or inter != 0.0:
            arr = arr * slope + inter
        if str(_get(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = -arr  # inverted grayscale; the later percentile scaling handles the offset
        rec["pixels"] = arr
    return rec


# ----------------------------------------------------------------------------- geometry
def plane_of(iop):
    n = np.cross(iop[:3], iop[3:])
    return PLANES[int(np.argmax(np.abs(n)))]


def orient_ops(iop, plane):
    """(transpose, flip_lr, flip_ud) that bring a slice into the canonical orientation."""
    r, c = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    (ra, rs), (ca, cs), _ = CANON[plane]
    transpose = abs(c[ra]) > abs(r[ra])
    if transpose:
        r, c = c, r
    flip_lr = (np.sign(r[ra]) or 1.0) != rs
    flip_ud = (np.sign(c[ca]) or 1.0) != cs
    return bool(transpose), bool(flip_lr), bool(flip_ud)


def order_key(rec, plane):
    """Slice position along the plane's canonical slice axis (ascending = canonical order)."""
    _, _, (sa, so) = CANON[plane]
    n = np.cross(rec["iop"][:3], rec["iop"][3:])
    n = n / (np.linalg.norm(n) + 1e-12)
    s = np.sign(n[sa]) or 1.0
    return float(np.dot(rec["ipp"], n) * s * so)


def apply_ops(img, ops, spacing):
    t, lr, ud = ops
    rs, cs = spacing  # DICOM PixelSpacing = (row spacing, column spacing) in mm
    if t:
        img, rs, cs = img.T, cs, rs
    if lr:
        img = img[:, ::-1]
    if ud:
        img = img[::-1, :]
    return np.ascontiguousarray(img), (rs, cs)


def center_x(rec):
    """Patient x (left-right) of the slice centre; useful later for inferring laterality."""
    r, c = rec["iop"][:3], rec["iop"][3:]
    rows, cols = rec["pixels"].shape[:2]
    rs, cs = rec["spacing"]
    return float((rec["ipp"] + r * cs * (cols - 1) / 2 + c * rs * (rows - 1) / 2)[0])


# ----------------------------------------------------------------------------- image ops
def to_canvas(img, row_mm, col_mm, size=SIZE):
    """Resize keeping the physical aspect ratio, then centre-pad to size x size."""
    h, w = img.shape
    ph, pw = h * row_mm, w * col_mm
    scale = size / max(ph, pw)
    nh = int(min(size, max(1, round(ph * scale))))
    nw = int(min(size, max(1, round(pw * scale))))
    interp = cv2.INTER_AREA if (nh < h or nw < w) else cv2.INTER_LINEAR
    out = cv2.resize(img.astype(np.float32), (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = out
    return canvas, (y0, x0, nh, nw)


def resample_slices(vol, n=N_SLICES):
    """Resample the slice axis to n slices (block-average first when there are >= 2n slices)."""
    s = vol.shape[0]
    if s == n:
        return vol
    if s == 1:
        return np.repeat(vol, n, axis=0)
    k = s // n
    if k >= 2:
        start = (s - (s // k) * k) // 2
        vol = vol[start:start + (s // k) * k].reshape(-1, k, *vol.shape[1:]).mean(1)
        s = vol.shape[0]
        if s == n:
            return vol
    x = np.linspace(0, s - 1, n)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, s - 1)
    w = (x - i0)[:, None, None].astype(np.float32)
    return vol[i0] * (1 - w) + vol[i1] * w


def to_uint8(vol, box, lo=0.5, hi=99.5):
    """Clip to the [lo, hi] percentiles of the image region (padding excluded) and scale to 0-255."""
    y0, x0, h, w = box
    roi = vol[:, y0:y0 + h, x0:x0 + w]
    a, b = np.percentile(roi[:, ::2, ::2], [lo, hi])
    if b <= a:
        b = a + 1.0
    out = np.zeros(vol.shape, np.uint8)
    out[:, y0:y0 + h, x0:x0 + w] = np.round(np.clip((roi - a) / (b - a), 0, 1) * 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- series / study
def build_volume(recs, n_slices=N_SLICES, size=SIZE):
    """Slice records (from read_slice) -> (uint8 [n_slices, size, size] volume, meta dict)."""
    recs = [r for r in recs if r is not None and r.get("pixels") is not None]
    if not recs:
        raise ValueError("no readable slices")
    mode = Counter(r["pixels"].shape for r in recs).most_common(1)[0][0]
    recs = [r for r in recs if r["pixels"].shape == mode]

    has_geo = all(r["iop"] is not None and r["ipp"] is not None for r in recs)
    ref = recs[len(recs) // 2]
    plane = plane_of(ref["iop"]) if ref["iop"] is not None else None
    if has_geo and plane is not None:
        keys, sorted_by = [order_key(r, plane) for r in recs], "position"
    else:
        keys, sorted_by = [float(r["instance"]) for r in recs], "instance"

    order = sorted(range(len(recs)), key=lambda i: (keys[i], recs[i]["echo"], recs[i]["instance"]))
    kept = []
    for i in order:
        if kept and sorted_by == "position" and abs(keys[i] - keys[kept[-1]]) < DUP_TOL_MM:
            continue  # same position (other echo / duplicate): keep the first
        kept.append(i)

    ops = orient_ops(ref["iop"], plane) if plane is not None else (False, False, False)
    slices, box = [], None
    for i in kept:
        img, sp = apply_ops(recs[i]["pixels"], ops, recs[i]["spacing"])
        can, box = to_canvas(img, sp[0], sp[1], size)
        slices.append(can)
    n_unique = len(slices)
    vol = to_uint8(resample_slices(np.stack(slices), n_slices), box)

    gaps = np.diff([keys[i] for i in kept]) if sorted_by == "position" and len(kept) > 1 else np.array([])
    meta = dict(
        plane_iop=plane or "", sorted_by=sorted_by, n_slices_used=len(recs), n_unique=n_unique,
        transposed=ops[0], flip_lr=ops[1], flip_ud=ops[2],
        slice_gap_mm=float(np.median(gaps)) if gaps.size else np.nan,
        rows=int(mode[0]), cols=int(mode[1]),
        row_spacing=float(ref["spacing"][0]), col_spacing=float(ref["spacing"][1]),
        slice_thickness=ref.get("slice_thickness", np.nan), spacing_between=ref.get("spacing_between", np.nan),
        fov_center_x=float(np.mean([center_x(recs[i]) for i in kept])) if has_geo else np.nan,
        laterality=ref.get("laterality", ""), patient_id=ref.get("patient_id", ""),
        manufacturer=ref.get("manufacturer", ""), field_strength=ref.get("field_strength", np.nan),
    )
    return vol, meta


def list_files(series_dir):
    return sorted(f for f in os.listdir(series_dir) if not f.startswith("."))


def read_series(series_dir, n_slices=N_SLICES, size=SIZE):
    files = list_files(series_dir)
    recs, n_failed = [], 0
    for f in files:
        try:
            recs.append(read_slice(os.path.join(series_dir, f)))
        except Exception:
            n_failed += 1
    vol, meta = build_volume(recs, n_slices, size)
    meta.update(n_files=len(files), n_failed=n_failed)
    return vol, meta


def select_series(study_rows, study_dir):
    """Pick one series per kind. study_rows: dicts with the train/test_series.csv columns."""
    cands = []
    for r in study_rows:
        d = os.path.join(study_dir, str(r["SeriesInstanceUID"]))
        n = len(list_files(d)) if os.path.isdir(d) else 0
        cands.append({**r, "n_files": n})
    picks = {}
    for kind, (plane, fluid, fs_pref) in KINDS.items():
        c = [r for r in cands
             if r["Anatomical_Plane"] == plane and int(r["Fluid_Sensitive"]) == fluid and r["n_files"] > 0]
        if c:
            picks[kind] = max(c, key=lambda r: (int(r["Fat_Suppression"]) == fs_pref, r["n_files"]))
    return picks, cands


def process_study(study_dir, study_rows, n_slices=N_SLICES, size=SIZE):
    """-> (arrays {kind: uint8 volume}, list of per-series meta dicts, study meta dict). Never raises
    for a single bad series: its error is recorded and the kind is left out of `arrays`."""
    study = os.path.basename(os.path.normpath(study_dir))
    picks, cands = select_series(study_rows, study_dir)
    arrays, series_meta = {}, []
    for kind, r in picks.items():
        sm = dict(StudyInstanceUID=study, kind=kind, SeriesInstanceUID=r["SeriesInstanceUID"],
                  Anatomical_Plane=r["Anatomical_Plane"], Fluid_Sensitive=int(r["Fluid_Sensitive"]),
                  Fat_Suppression=int(r["Fat_Suppression"]), error="")
        try:
            vol, m = read_series(os.path.join(study_dir, str(r["SeriesInstanceUID"])), n_slices, size)
            arrays[kind] = vol
            sm.update(m)
        except Exception as e:
            sm["error"] = f"{type(e).__name__}: {e}"[:300]
        series_meta.append(sm)

    # Study-level laterality and PatientID: kept series + one header from every other series
    lat, pids = Counter(), Counter()
    for m in series_meta:
        if m.get("laterality"):
            lat[m["laterality"]] += 1
        if m.get("patient_id"):
            pids[m["patient_id"]] += 1
    picked = {str(r["SeriesInstanceUID"]) for r in picks.values()}
    for r in cands:
        if str(r["SeriesInstanceUID"]) in picked or r["n_files"] == 0:
            continue
        d = os.path.join(study_dir, str(r["SeriesInstanceUID"]))
        try:
            h = read_slice(os.path.join(d, list_files(d)[0]), with_pixels=False)
            if h["laterality"]:
                lat[h["laterality"]] += 1
            if h["patient_id"]:
                pids[h["patient_id"]] += 1
        except Exception:
            pass
    laterality = ""
    if lat:
        top, cnt = lat.most_common(1)[0]
        laterality = top if cnt > sum(lat.values()) - cnt else ""  # strict majority only

    study_meta = dict(
        StudyInstanceUID=study,
        PatientID=pids.most_common(1)[0][0] if pids else "",
        laterality=laterality, lat_votes_L=lat["L"], lat_votes_R=lat["R"],
        kinds=",".join(k for k in KINDS if k in arrays),
        n_series_csv=len(cands), n_series_errors=sum(bool(m["error"]) for m in series_meta),
    )
    return arrays, series_meta, study_meta


def process_and_save(task):
    """Multiprocessing worker. task = (study_dir, study_rows, out_npz_path). Never raises."""
    import time

    study_dir, study_rows, out_path = task
    t0 = time.time()
    study = os.path.basename(os.path.normpath(study_dir))
    try:
        arrays, series_meta, study_meta = process_study(study_dir, study_rows)
        if arrays:
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **arrays)
            os.replace(tmp, out_path)
        study_meta["npz_bytes"] = os.path.getsize(out_path) if arrays else 0
        study_meta["error"] = "" if arrays else "no series could be processed"
    except Exception as e:
        series_meta = []
        study_meta = dict(StudyInstanceUID=study, npz_bytes=0, error=f"{type(e).__name__}: {e}"[:300])
    study_meta["seconds"] = round(time.time() - t0, 2)
    return study_meta, series_meta


def load_study(npz_path):
    """-> {kind: uint8 [N_SLICES, SIZE, SIZE]} for the kinds present in the study."""
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}
