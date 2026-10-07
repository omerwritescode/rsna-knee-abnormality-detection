"""Deterministic fixes applied on top of the raw LLM scores.

Applied when reading a JSONL (evaluate.py, build_labels.py), never written back,
so the raw model output stays untouched and rules can be changed without rerunning
the model.

Rule 1 (v3 dev): Synovitis from effusion size.
  In the dev 20, Synovitis=1 tracks a large/moderate effusion or hemarthrosis even
  when the report never mentions the synovium. The prompt asks the model to apply
  this, but v3 ignored it. So: if Synovitis is absent and the model's Effusion quote
  describes a moderate+ effusion or a hemarthrosis, set Synovitis to 0.5.
"""
from __future__ import annotations

import re

from .run import normalise

POST_VERSION = "p1"

# moderate-or-larger effusion, or blood in the joint, in the report languages
LARGE_EFFUSION = re.compile(
    r"moderate|large|massive|marked|significant|considerable|extensive|"
    r"h(a)?emarthros|h(a)?emorrhagic|lipoh(a)?emarthros|"
    r"moderad|abundante|importante|severo|hemartros|hemorr[aá]gic|"          # es
    r"orta|belirgin|yayg[ıi]n|fazla|hemartroz|"                                # tr
    r"m[äa]ßig|mäss?ig|ausgepr[äa]gt|deutlich|gro(ß|ss)|h[äa]marthros|"       # de
    r"умерен|голям|значителен|изразен|хеморагич|хемартроз|"                   # bg
    r"ικαν|μεγάλ|σημαντικ|αιμάρθρ|"                                            # el
    r"umjeren|opsež|obiln|znača|hemartroz|"                                    # hr
    r"matig|groot|fors|"                                                       # nl
    r"mod[ée]r[ée]|abondant|important|h[ée]marthros",                          # fr
    re.IGNORECASE,
)


NEGATED = re.compile(r"\b(no|not|without|sin|yok|kein|keine|geen|bez)\b|без|δεν|pas de",
                     re.IGNORECASE)


def apply(scores: dict[str, float], findings) -> tuple[dict[str, float], list[str]]:
    """Return (new_scores, list of rules that fired)."""
    out = dict(scores)
    fired = []
    f = normalise(findings)
    eff_q = str((f.get("Effusion") or {}).get("q", ""))
    if out.get("Synovitis", 0) == 0 and out.get("Effusion", 0) > 0 \
            and LARGE_EFFUSION.search(eff_q) and not NEGATED.search(eff_q):
        out["Synovitis"] = 0.5
        fired.append("synovitis_from_effusion")
    return out, fired
