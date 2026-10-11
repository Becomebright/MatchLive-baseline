"""Step 2: build the match memory of each game of a split, playing the game as a live stream.

    python scripts/build_memory.py --split test                  # every game
    python scripts/build_memory.py --split test --game game1     # one game (run games in parallel processes)

Needs the perception cache of both halves (scripts/run_perception.py) and, for any call not in
the API cache, an endpoint and key in configs/api.local.yaml. The system is given only the
pre-match facts (data/prematch/<split>.json) and the video up to the current moment: identity
voting, the event analyzer on every 30-s window, the consistency check and the arbiter all run on
one match clock, each model answer taking effect at its issue time + its measured latency.

Writes outputs/memory/<split>/<game>/<name>/: ops.jsonl (the memory log, everything the QA reads),
windows.jsonl (each analyzer call and its event candidates), cases.json (each arbitrated
conflict), perceived_half{1,2}.pkl (the frames as the analyzer was told about them) and run.json.
The model calls are sequential on the match clock, so a game takes 1-3 h of wall time; a re-run
with the same inputs replays from the API cache in minutes.
"""

from __future__ import annotations

import argparse
import json
import time

from matchlive.common.api import APIClient
from matchlive.common.cache import half_cache
from matchlive.common.config import load_config
from matchlive.common.match import Match, games, load_manifest
from matchlive.runtime.runner import build_memory, memory_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("test", "challenge"))
    ap.add_argument("--game", action="append", help="a game of the split, e.g. game1 (repeatable; default: all)")
    ap.add_argument("--name", default="full", help="run name (default: full)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(config, args.split)
    # Every requested game and its perception cache are checked before any model call.
    matches = [Match(config, args.split, game, manifest) for game in args.game or games(manifest)]
    for match in matches:
        for half in (1, 2):
            half_cache(config, match, half)
    client = APIClient(config["api"], config["paths"]["cache"])
    for match in matches:
        out_dir = memory_dir(config, match, args.name)
        print(f"== {match.key}: {match.prematch['teams']['home']} v {match.prematch['teams']['away']} -> {out_dir}",
              flush=True)
        t0 = time.time()
        mem = build_memory(config, match, out_dir, client)
        wall = time.time() - t0
        (out_dir / "run.json").write_text(json.dumps({"model": config["api"]["model"], "wall_s": wall}, indent=1))
        score = mem.at((3, 0.0)).score
        n_windows = sum(1 for _ in open(out_dir / "windows.jsonl"))
        n_cases = len(json.loads((out_dir / "cases.json").read_text()))
        print(f"{match.key}: {wall:.0f} s, {n_windows} analyzer calls, {n_cases} arbitrations, "
              f"{len(mem.entries)} memory entries; score in memory at the final whistle "
              f"{score['home']}-{score['away']}", flush=True)


if __name__ == "__main__":
    main()
