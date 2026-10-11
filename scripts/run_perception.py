"""Step 1: run perception over the half videos of a split and write the raw perception cache.

    python scripts/run_perception.py --split test                       # every game, both halves
    python scripts/run_perception.py --split test --game game1 --half 1
    python scripts/run_perception.py --split test --game game1 --half 1 --start 600 --seconds 120   # a check

One pass over each half video on the GPU: detection, shot cuts, tracking, jersey numbers and kit
colours, written to cache/perception/<split>/<game>/. A half whose cache exists is skipped.
TensorRT engines are built on first use (minutes) and cached under cache/trt/.
"""

from __future__ import annotations

import argparse

from matchlive.common.cache import load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match, games, load_manifest
from matchlive.runtime.runner import MatchRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("test", "challenge"))
    ap.add_argument("--game", action="append", help="a game of the split, e.g. game1 (repeatable; default: all)")
    ap.add_argument("--half", type=int, choices=(1, 2), help="one half only (default: both)")
    ap.add_argument("--start", type=float, default=0.0, help="start second within the half (a check)")
    ap.add_argument("--seconds", type=float, help="segment length (default: to the end of the half)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    config = load_config(args.config)
    manifest = load_manifest(config, args.split)
    matches = [Match(config, args.split, game, manifest) for game in args.game or games(manifest)]
    runner = MatchRunner(config)
    print(f"detector on {runner.detector.providers[0]}, jersey reader on {runner.jersey.providers[0]}", flush=True)
    for match in matches:
        for half in [args.half] if args.half else [1, 2]:
            # The videos are 25 fps; the runner reads the real rate from the video.
            start_frame = int(round(args.start * 25))
            n_frames = None if args.seconds is None else int(round(args.seconds * 25))
            path = runner.run_perception(match, half, start_frame, n_frames)

            arrays, meta = load_perception(path)
            print(f"{match.key} half {half}: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
            print(f"  {meta['n_frames']} frames, {arrays['shot_id'][-1] - arrays['shot_id'][0]} cuts, "
                  f"{len(arrays['det'])} detections, {len(arrays['track'])} track boxes")
            print("  ms/frame: " + "  ".join(f"{k} {v:.1f}" for k, v in meta["ms_per_frame"].items())
                  + f"  | total {sum(meta['ms_per_frame'].values()):.1f}", flush=True)


if __name__ == "__main__":
    main()
