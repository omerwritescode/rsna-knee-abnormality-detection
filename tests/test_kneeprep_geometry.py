"""Synthetic tests for kneeprep's geometry, ordering, resizing and intensity logic.

No DICOM files or pydicom needed: a 3D phantom is sliced in every orientation variant a
scanner could produce, turned into the same records read_slice() returns, and fed to
build_volume(). Run:  python tests/test_kneeprep_geometry.py
"""
import itertools
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kneeprep.core import build_volume, select_series  # noqa: E402

DIMS = (40, 48, 56)  # phantom extent in mm along LPS x, y, z


def make_phantom():
    x, y, z = np.meshgrid(*[np.arange(d) for d in DIMS], indexing="ij")
    p = 0.2 + 0.5 * z / (DIMS[2] - 1) + 0.1 * np.sin(x / 3.0) + 0.05 * np.cos(y / 4.0)
    # bright marker: patient LEFT (high x), ANTERIOR (low y), SUPERIOR (high z)
    p[32:38, 2:8, 48:54] = 3.0
    return p.astype(np.float32)


PH = make_phantom()


def slice_records(normal_axis, r_axis, r_sign, c_sign, axis_step=None,
                  echo_dups=False, drop_ipp=False, shuffle=True, seed=0):
    """Slice the phantom with row direction r (image right) and column direction c (image down).
    axis_step: sampling step in mm per LPS axis (anisotropic pixels), e.g. {2: 2}."""
    c_axis = 3 - normal_axis - r_axis
    axis_step = axis_step or {}
    step_r, step_c = axis_step.get(r_axis, 1), axis_step.get(c_axis, 1)
    nr = DIMS[r_axis] // step_r
    nc = DIMS[c_axis] // step_c
    recs = []
    for k in range(DIMS[normal_axis]):
        img = np.zeros((nc, nr), np.float32)
        for i in range(nc):
            for j in range(nr):
                pt = [0, 0, 0]
                pt[normal_axis] = k
                # negative directions walk the SAME sample grid backwards, so variants match exactly
                pt[r_axis] = (j if r_sign > 0 else nr - 1 - j) * step_r
                pt[c_axis] = (i if c_sign > 0 else nc - 1 - i) * step_c
                img[i, j] = PH[tuple(pt)]
        r = np.zeros(3); r[r_axis] = r_sign
        c = np.zeros(3); c[c_axis] = c_sign
        ipp = np.zeros(3)
        ipp[normal_axis] = k
        ipp[r_axis] = 0 if r_sign > 0 else (nr - 1) * step_r
        ipp[c_axis] = 0 if c_sign > 0 else (nc - 1) * step_c
        base = dict(iop=np.concatenate([r, c]), ipp=None if drop_ipp else ipp,
                    spacing=(float(step_c), float(step_r)), instance=k + 1, echo=1,
                    laterality="", patient_id="P1", manufacturer="X",
                    field_strength=1.5, slice_thickness=1.0, spacing_between=1.0)
        recs.append({**base, "pixels": img})
        if echo_dups:  # second echo at the same position, different content: must be dropped
            recs.append({**base, "echo": 2, "instance": 1000 + k,
                         "pixels": np.random.default_rng(k).random(img.shape).astype(np.float32)})
    if shuffle:
        random.Random(seed).shuffle(recs)
    return recs


PLANE_NORMAL = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def variants(plane):
    n = PLANE_NORMAL[plane]
    inplane = [a for a in range(3) if a != n]
    for r_axis, rs, cs in itertools.product(inplane, (1, -1), (1, -1)):
        yield r_axis, rs, cs


def marker_location(vol):
    s, r, c = np.nonzero(vol == vol.max())
    return s.mean(), r.mean(), c.mean()


def test_all_orientations_agree_and_marker_is_where_expected():
    expect = {  # (slice index, row, col) relative to the middle: -1 = low half, +1 = high half
        "Sagittal": (+1, -1, -1),  # slices R->L (marker left: late), superior up, anterior on the left
        "Coronal": (-1, -1, +1),   # slices A->P (marker anterior: early), superior up, patient left on the right
        "Axial": (-1, -1, +1),     # slices S->I (marker superior: early), anterior up, patient left on the right
    }
    for plane in PLANE_NORMAL:
        outs, metas = [], []
        for r_axis, rs, cs in variants(plane):
            vol, meta = build_volume(slice_records(PLANE_NORMAL[plane], r_axis, rs, cs,
                                                     axis_step={max(a for a in range(3) if a != PLANE_NORMAL[plane]): 2}))
            assert vol.shape == (32, 256, 256) and vol.dtype == np.uint8
            assert meta["plane_iop"] == plane and meta["sorted_by"] == "position"
            outs.append(vol)
            metas.append(meta)
        for v in outs[1:]:
            assert np.array_equal(v, outs[0]), f"{plane}: orientation variants disagree"
        s, r, c = marker_location(outs[0])
        got = (np.sign(s - 15.5), np.sign(r - 127.5), np.sign(c - 127.5))
        assert got == expect[plane], f"{plane}: marker at {got}, expected {expect[plane]}"
        assert len({(m["transposed"], m["flip_lr"], m["flip_ud"]) for m in metas}) == 8
        print(f"ok  {plane}: 8 orientation variants identical, marker where expected")


def test_echo_duplicates_dropped():
    a, ma = build_volume(slice_records(0, 1, 1, -1))
    b, mb = build_volume(slice_records(0, 1, 1, -1, echo_dups=True))
    assert np.array_equal(a, b) and mb["n_unique"] == DIMS[0] == ma["n_unique"]
    print("ok  duplicate positions: second echo dropped")


def test_instance_fallback_without_position():
    a, _ = build_volume(slice_records(1, 0, 1, -1, shuffle=False))
    b, mb = build_volume(slice_records(1, 0, 1, -1, drop_ipp=True, shuffle=True))
    assert mb["sorted_by"] == "instance" and np.array_equal(a, b)
    print("ok  missing ImagePositionPatient: falls back to InstanceNumber order")


def test_resample_counts():
    for n_src in (1, 5, 31, 32, 33, 64, 70, 144):
        recs = []
        base = slice_records(2, 0, 1, 1, shuffle=False)
        for k in range(n_src):  # n_src slices at distinct positions
            r = dict(base[k % len(base)])
            r["ipp"] = np.array([0.0, 0.0, float(k)])
            recs.append(r)
        vol, meta = build_volume(recs)
        assert vol.shape == (32, 256, 256) and meta["n_unique"] == n_src
    print("ok  1..144 source slices all resample to 32")


def test_padding_stays_black_and_full_range_used():
    vol, _ = build_volume(slice_records(0, 1, 1, -1))
    assert vol[:, :, :5].max() == 0 and vol[:, :, -5:].max() == 0  # 48x56 mm FOV -> padded left/right
    assert vol.max() == 255 and vol[vol > 0].min() < 40
    print("ok  padding stays 0, intensities span 0-255")


def test_select_series_prefers_fat_suppression_then_slices():
    with tempfile.TemporaryDirectory() as d:
        rows = []
        spec = [("s1", "Sagittal", 1, 1, 20), ("s2", "Sagittal", 1, 1, 30), ("s3", "Sagittal", 1, 0, 60),
                ("s4", "Sagittal", 0, 0, 25), ("c1", "Coronal", 1, 0, 30), ("a1", "Axial", 0, 0, 30)]
        for uid, plane, fl, fs, n in spec:
            os.makedirs(os.path.join(d, uid))
            for i in range(n):
                open(os.path.join(d, uid, f"{i}.dcm"), "w").close()
            rows.append(dict(SeriesInstanceUID=uid, Anatomical_Plane=plane, Fluid_Sensitive=fl, Fat_Suppression=fs))
        picks, _ = select_series(rows, d)
        got = {k: v["SeriesInstanceUID"] for k, v in picks.items()}
        assert got == {"sag_fl": "s2", "cor_fl": "c1", "sag_t1": "s4"}, got
    print("ok  series selection: FS preferred, then most slices; missing kinds left out")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nall tests passed")
