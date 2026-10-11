"""Step 3: answer every question of a split from what the system knew when it was asked.

    python scripts/run_qa.py --split test                    # every game -> outputs/submission_test.json
    python scripts/run_qa.py --split test --game game1       # one game (the submission marks the rest null)
    python scripts/run_qa.py --split test --run qwen --config configs/qwen.yaml
                                                             # the memory of build_memory.py --name qwen

Each question is asked at its time in the half (`position_ms`) and answered from what was known
at that moment: the match memory (scripts/build_memory.py) as known by then, perception and video
up to then. The model sees the pre-match facts, the question and its options, and nothing else from
the question file.

Writes one line per question to outputs/qa/<split>/<run>/<game>.jsonl, then the submission
(OSL JSON: the question file with a `prediction` option id, or null, on every question) from
every game answered so far: outputs/submission_<split>.json for the run `full`, else
outputs/submission_<split>_<run>.json. If the question file has the answers (test), the accuracy
of each question subcategory and their average are printed; the official scorer is
codes/eval_predictions.py in the challenge repository.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from matchlive.app.qa import QAAgent, question_prompt
from matchlive.app.submission import write_submission
from matchlive.common.api import APIClient, NoAPIKey
from matchlive.common.config import load_config
from matchlive.common.match import Match, games, load_manifest
from matchlive.runtime.qa_tools import QATools
from matchlive.runtime.runner import load_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("test", "challenge"))
    ap.add_argument("--game", action="append", help="a game of the split, e.g. game1 (repeatable; default: all)")
    ap.add_argument("--run", default="full", help="match memory run name (scripts/build_memory.py --name)")
    ap.add_argument("--workers", type=int, default=8, help="questions answered at once")
    ap.add_argument("--output", help="submission file (default: outputs/submission_<split>.json, or "
                                     "outputs/submission_<split>_<run>.json for a run other than full)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    outputs = config["paths"]["outputs"]
    manifest = load_manifest(config, args.split)
    # Every requested game is checked before any question is asked.
    matches = [Match(config, args.split, game, manifest) for game in args.game or games(manifest)]
    client = APIClient(config["api"], config["paths"]["cache"])
    agent = QAAgent(config["qa"], client.generate)
    log_dir = outputs / "qa" / args.split / args.run
    log_dir.mkdir(parents=True, exist_ok=True)

    for match in matches:
        tools = QATools(config, match, *load_run(config, match, args.run))
        questions = [(s, q) for s in match.samples for q in s["streaming_vqa"]]

        def ask(item):
            sample, q = item
            now = (sample["metadata"]["half"], q["position_ms"] / 1000)
            # Options are shown as letters, in the file's order; the answer maps back to the option id.
            text, letters = question_prompt(match.prematch, q["question"], [o["text"] for o in q["options"]], now)
            try:
                a = agent.answer(text, letters, lambda tool, targs: tools.call(now, tool, targs))
                out = {"letter": a.choice, "calls": a.calls, "latency": a.latency, "tools": a.tools}
            except NoAPIKey:
                raise
            except Exception as e:  # noqa: BLE001 — out of API retries: the question goes unanswered
                out = {"letter": None, "error": f"{type(e).__name__}: {e}"}
            prediction = q["options"][letters.index(out["letter"])]["id"] if out["letter"] else None
            return {"sample": sample["id"], "id": q["id"], "half": now[0], "position_ms": q["position_ms"],
                    "subcategory": q["metadata"].get("subcategory"), "question": q["question"],
                    "prediction": prediction, "correct": q.get("correct_option_id"), **out}

        print(f"== {match.key}: {len(questions)} questions", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            results = []
            for n, r in enumerate(pool.map(ask, questions), 1):
                results.append(r)
                if n % 50 == 0 or n == len(questions):
                    print(f"{n}/{len(questions)} answered, {time.time() - t0:.0f} s", flush=True)
        with open(log_dir / f"{match.game}.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    name = args.split if args.run == "full" else f"{args.split}_{args.run}"
    write_submission(manifest, log_dir, args.output or outputs / f"submission_{name}.json")


if __name__ == "__main__":
    main()
