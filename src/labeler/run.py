"""Label knee MRI reports with a local Ollama model. Resumable.

Usage (from the project root, venv active, Ollama running):
    python -m src.labeler.run --subset dev                 # iterate on the prompt
    python -m src.labeler.run --subset heldout             # score once definitions are frozen
    python -m src.labeler.run --subset all                 # overnight full run (4,407)

Each subset writes to its own file: data/labels/raw_<version>_<subset>.jsonl
Re-running the same command skips studies already in the file. Use --redo to
start that file again. Progress is logged to data/labels/run_<version>_<subset>.log
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .prompt import LABELS, PROMPT_VERSION, SCHEMA, SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:9b"
OPTIONS = {
    "temperature": 0,
    "num_ctx": 8192,       # default 4096 is too small for prompt + long report
    "num_predict": 600,    # hard cap; a normal answer is 50-250 tokens
    "seed": 0,
}


def to_scores(findings: list[dict]) -> tuple[dict[str, float], list[str]]:
    """Findings list -> 12 soft labels (present 1, uncertain 0.5, missing 0)."""
    scores = {lab: 0.0 for lab in LABELS}
    problems = []
    for f in findings:
        lab, st = f.get("label"), f.get("status")
        if lab not in scores:
            problems.append(f"unknown label {lab!r}")
            continue
        val = {"present": 1.0, "uncertain": 0.5}.get(st)
        if val is None:
            problems.append(f"unknown status {st!r} for {lab}")
            continue
        scores[lab] = max(scores[lab], val)
    return scores, problems


def call_ollama(session: requests.Session, model: str, report: str,
                timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"REPORT:\n{report.strip()}"},
        ],
        "format": SCHEMA,
        "think": False,        # qwen3.5 starts with thinking on otherwise
        "stream": False,
        "keep_alive": "2h",
        "options": OPTIONS,
    }
    r = session.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def select(df: pd.DataFrame, split_path: Path, subset: str) -> pd.DataFrame:
    labeled = df[LABELS].notna().all(axis=1)
    if subset == "all":
        return df
    if subset == "unlabeled":
        return df[~labeled]
    if subset == "labeled":
        return df[labeled]
    sp = pd.read_csv(split_path)
    ids = set(sp.loc[sp["split"] == subset, "StudyInstanceUID"])
    if not ids:
        raise SystemExit(f"no studies with split={subset!r} in {split_path}")
    return df[df["StudyInstanceUID"].isin(ids)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--subset", default="dev",
                    choices=["dev", "heldout", "labeled", "unlabeled", "all"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="stop after N new studies")
    ap.add_argument("--redo", action="store_true", help="overwrite the output file")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--tag", default="", help="extra suffix for the output name")
    args = ap.parse_args()

    labels_dir = args.root / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{PROMPT_VERSION}_{args.subset}{('_' + args.tag) if args.tag else ''}"
    out_path = labels_dir / f"raw_{stem}.jsonl"
    log_path = labels_dir / f"run_{stem}.log"

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()])
    log = logging.getLogger("labeler")

    df = pd.read_csv(args.root / "data" / "raw" / "train.csv")
    todo = select(df, labels_dir / "split.csv", args.subset)

    if args.redo and out_path.exists():
        out_path.unlink()
    done = set()
    if out_path.exists() and out_path.stat().st_size:
        # A crash mid-write leaves a last line with no newline; drop it so the
        # next record doesn't get glued onto it.
        data = out_path.read_bytes()
        if not data.endswith(b"\n"):
            out_path.write_bytes(data[: data.rfind(b"\n") + 1])
            log.warning("dropped a half-written last line in %s", out_path)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line after a crash
                if rec.get("ok"):
                    done.add(rec["StudyInstanceUID"])
    todo = todo[~todo["StudyInstanceUID"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("prompt=%s model=%s subset=%s: %d already done, %d to do -> %s",
             PROMPT_VERSION, args.model, args.subset, len(done), len(todo), out_path)

    session = requests.Session()
    t_start, n_ok, n_fail = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            uid, report = row.StudyInstanceUID, row.Report
            rec = {"StudyInstanceUID": uid, "prompt": PROMPT_VERSION,
                   "model": args.model, "ok": False}
            for attempt in range(3):
                t0 = time.time()
                try:
                    resp = call_ollama(session, args.model, report, args.timeout)
                    raw = resp["message"]["content"]
                    findings = json.loads(raw)["findings"]
                    scores, problems = to_scores(findings)
                    rec.update(ok=True, findings=findings, scores=scores,
                               problems=problems,
                               eval_tokens=resp.get("eval_count"),
                               prompt_tokens=resp.get("prompt_eval_count"),
                               seconds=round(time.time() - t0, 2),
                               truncated=resp.get("done_reason") == "length")
                    if problems:
                        log.warning("%s: %s", uid, "; ".join(problems))
                    break
                except (requests.RequestException, KeyError,
                        json.JSONDecodeError, TypeError) as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    log.warning("%s attempt %d failed: %s", uid, attempt + 1, rec["error"])
                    time.sleep(5 * (attempt + 1))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += rec["ok"]
            n_fail += not rec["ok"]
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t_start
                eta = el / i * (len(todo) - i)
                log.info("%d/%d  ok=%d fail=%d  %.1f s/report  ETA %.1f h",
                         i, len(todo), n_ok, n_fail, el / i, eta / 3600)
    log.info("finished: ok=%d fail=%d (failed studies are retried on the next run)",
             n_ok, n_fail)


if __name__ == "__main__":
    main()
