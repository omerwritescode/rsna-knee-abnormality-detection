"""Out-of-fold scoring for the 12 labels.

Two views, reported side by side:
  auc_llm   : LLM-labelled studies; per label the 'uncertain' (0.5) rows are left out
              and the rest are 1/0. Main signal (thousands of studies).
  auc_human : the 58 human-labelled studies (true labels). Small and noisy, but the
              only check against the organisers' own definitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def score(pred: np.ndarray, target: np.ndarray, source: np.ndarray) -> pd.DataFrame:
    """pred, target: [n, 12]; source: 'llm' / 'human' per row."""
    rows = []
    llm = source == "llm"
    hum = source == "human"
    for j, lab in enumerate(LABELS):
        t, p = target[:, j], pred[:, j]
        certain = llm & (t != 0.5)
        rows.append(dict(
            label=lab,
            auc_llm=safe_auc(t[certain] >= 0.5, p[certain]),
            pos_llm=int((t[certain] >= 0.5).sum()),
            auc_human=safe_auc(t[hum] >= 0.5, p[hum]),
            pos_human=int((t[hum] >= 0.5).sum()),
        ))
    df = pd.DataFrame(rows).set_index("label")
    df.loc["MACRO"] = [df["auc_llm"].mean(), df["pos_llm"].sum(),
                       df["auc_human"].mean(), df["pos_human"].sum()]
    return df
