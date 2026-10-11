"""The submission file of a split, and the accuracy of one when the question file has the answers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def write_submission(manifest: Mapping[str, Any], log_dir: Path, out_path: Path) -> None:
    """The submission (OSL JSON: the question file with a `prediction` option id, or null, on every
    question) from the per-game logs in `log_dir` written so far, to `out_path`; then the number of
    questions answered and, if the question file has the answers (test), the accuracy of each
    question subcategory and their average. The official scorer is codes/eval_predictions.py in
    the challenge repository."""
    answered = {(r["sample"], r["id"]): r for path in sorted(log_dir.glob("*.jsonl"))
                for r in map(json.loads, open(path))}
    submission = {k: v for k, v in manifest.items() if k != "data"}
    submission["data"] = [
        {"id": s["id"], "inputs": s["inputs"], "metadata": s["metadata"],
         "streaming_vqa": [{"id": q["id"], "position_ms": q["position_ms"], "question": q["question"],
                            "options": q["options"], "metadata": q["metadata"],
                            "prediction": answered.get((s["id"], q["id"]), {}).get("prediction")}
                           for q in s["streaming_vqa"]]}
        for s in manifest["data"]]
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=1, ensure_ascii=False)
    questions = [q for s in manifest["data"] for q in s["streaming_vqa"]]
    predictions = [p for s in submission["data"] for p in s["streaming_vqa"]]
    print(f"{out_path}: {sum(p['prediction'] is not None for p in predictions)} of {len(predictions)} questions answered")

    if all(q.get("correct_option_id") for q in questions):
        by = defaultdict(list)
        for q, p in zip(questions, predictions):
            by[q["metadata"].get("subcategory")].append(p["prediction"] == q["correct_option_id"])
        accs = {c: sum(v) / len(v) for c, v in sorted(by.items())}
        for c, a in accs.items():
            print(f"  {c:22s} {a:6.1%}  ({len(by[c])} questions)")
        print(f"  {'average':22s} {sum(accs.values()) / len(accs):6.1%}")
