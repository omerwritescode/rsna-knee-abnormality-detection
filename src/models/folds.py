"""5-fold CV split over all training studies, grouped by patient.

    python -m src.models.folds

* Patient grouping: PatientID from data/raw/study_meta_chunk*.csv; all studies of a
  patient go to the same fold. Prints how many patients have more than one study.
* Stratification (greedy, deterministic): the 12 labels (soft label >= 0.5 counts as
  positive) plus the report language, so every fold has a similar mix of findings and
  reporting sites. The 58 human-labelled studies are spread across folds too.

Output: data/labels/folds.csv  StudyInstanceUID, PatientID, fold, lang, source
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

_LANG_HINTS = {
    "es": r"\b(menisco|derrame|rotura|ligamento|rodilla|sin|con|del)\b",
    "de": r"\b(meniskus|kreuzband|erguss|kniegelenk|und|der|die|kein)\b",
    "hr": r"\b(menisk[a-z]*|ligament[a-z]*|zglob[a-z]*|uredn[a-z]*|bez|uz)\b|[čćđ]",
    "nl": r"\b(van|het|een|geen|gewricht[a-z]*|kniegewricht)\b",
    "fr": r"\b(ménisque|épanchement|genou|du|des|pas)\b",
    "tr": r"[ığş]|\b(menisküs|bağ|izlendi|mevcuttur|normaldir)\b",
}


def guess_language(text: str) -> str:
    """Rough report-language guess (used only to balance folds)."""
    t = str(text).lower()
    if re.search(r"[а-яё]", t):
        return "bg"
    if re.search(r"[α-ω]", t):
        return "el"
    scores = {lang: len(re.findall(pat, t)) for lang, pat in _LANG_HINTS.items()}
    lang, best = max(scores.items(), key=lambda kv: kv[1])
    en = len(re.findall(r"\b(the|of|and|is|with|no|meniscus|ligament|effusion)\b", t))
    return lang if best > max(2, en) else "en"


def read_labels(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_patients(raw: Path) -> pd.Series | None:
    files = sorted(raw.glob("study_meta_chunk*.csv"))
    if not files:
        return None
    meta = pd.concat([pd.read_csv(f, usecols=["StudyInstanceUID", "PatientID"], dtype=str)
                      for f in files])
    return meta.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")["PatientID"]


def greedy_group_folds(Y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each group to a fold, keeping every column's positives and the fold
    sizes as even as possible. Rare columns are placed first."""
    G = groups.max() + 1
    gY = np.zeros((G, Y.shape[1]))
    np.add.at(gY, groups, Y)
    gsize = np.bincount(groups, minlength=G).astype(float)
    totals = Y.sum(0)
    tgt = totals / k
    tgt_size = len(groups) / k
    w = 1.0 / np.maximum(tgt, 1.0)

    rng = np.random.default_rng(seed)
    with np.errstate(divide="ignore"):
        rarity = np.where(gY > 0, totals[None, :], np.inf).min(1)   # rarest column present
    order = np.lexsort((rng.random(G), -gsize, rarity))

    cnt = np.zeros((k, Y.shape[1]))
    size = np.zeros(k)
    fold_of_group = np.empty(G, dtype=int)
    for g in order:
        before = (w * (cnt - tgt) ** 2).sum(1) + ((size - tgt_size) / tgt_size) ** 2 * 50
        after = (w * (cnt + gY[g] - tgt) ** 2).sum(1) + ((size + gsize[g] - tgt_size) / tgt_size) ** 2 * 50
        f = int(np.argmin(after - before))
        fold_of_group[g] = f
        cnt[f] += gY[g]
        size[f] += gsize[g]
    return fold_of_group[groups]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    raw = args.root / "data" / "raw"
    lab_path = args.labels or args.root / "data" / "labels" / "soft_labels.parquet"
    labels = read_labels(lab_path)
    train = pd.read_csv(raw / "train.csv", usecols=["StudyInstanceUID", "Report"])
    df = labels.merge(train, on="StudyInstanceUID", how="left")
    df = df[df["source"] != "missing"].reset_index(drop=True)

    pids = load_patients(raw)
    if pids is None:
        print("WARNING: no study_meta_chunk*.csv in data/raw -> no patient grouping")
        df["PatientID"] = ""
    else:
        df["PatientID"] = df["StudyInstanceUID"].map(pids).fillna("")
    key = df["PatientID"].where(df["PatientID"] != "", "study:" + df["StudyInstanceUID"])
    groups = pd.factorize(key)[0]
    per_patient = pd.Series(groups).value_counts()
    print(f"studies {len(df)} · patients {groups.max() + 1} · "
          f"patients with >1 study: {(per_patient > 1).sum()} "
          f"(covering {per_patient[per_patient > 1].sum()} studies) · "
          f"missing PatientID: {(df['PatientID'] == '').sum()}")

    df["lang"] = df["Report"].map(guess_language)
    Yl = (df[LABELS].to_numpy() >= 0.5).astype(float)
    Ylang = pd.get_dummies(df["lang"]).to_numpy().astype(float)
    Yhuman = (df["source"] == "human").to_numpy()[:, None].astype(float)
    Y = np.hstack([Yl, Ylang, Yhuman])
    df["fold"] = greedy_group_folds(Y, groups, args.k, args.seed)

    out = args.root / "data" / "labels" / "folds.csv"
    df[["StudyInstanceUID", "PatientID", "fold", "lang", "source"]].to_csv(out, index=False)

    summary = pd.DataFrame(Yl, columns=LABELS).groupby(df["fold"]).sum().astype(int)
    summary.insert(0, "studies", df.groupby("fold").size())
    summary["human"] = df.groupby("fold")["source"].apply(lambda s: (s == "human").sum())
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\npositives (>= 0.5) per fold:")
        print(summary.to_string())
        print("\nlanguages per fold:")
        print(pd.crosstab(df["fold"], df["lang"]).to_string())
    if pids is not None:
        leak = df.groupby("PatientID")["fold"].nunique()
        leak = leak[(leak > 1) & (leak.index != "")]
        print(f"\npatients split across folds: {len(leak)} (must be 0)")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
