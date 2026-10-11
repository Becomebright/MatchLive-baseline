"""Raw perception cache: one compressed .npz per (match, half, frame range).

Everything that needs the video pixels or a model is computed once, in one pass over the
half video (scripts/run_perception.py); identity voting and the match memory recompute from it
instead of re-running the models. Arrays (row order = frame order):

  shot_id  int32   (n_frames,)       shot id of each frame
  det      float32 (N, 7)            frame, l, t, w, h, score, class_id — the detector's
                                     output (its confidence cut applied, referees dropped);
                                     the ball is selected from these at load time
  track    float32 (M, 8)            frame, shot_id, role (0 goalkeeper, 1 player),
                                     track_id, l, t, w, h
  jersey   float16 (M, 101)          jersey-number probabilities of each track row, read
                                     on every frame; NaN where the box was too small to crop
  color    float32 (M, 3)            mean LAB kit colour of each track row's crop
                                     (identity.kit_color); NaN where there was no crop

`frame` is the frame index within the half. The file name carries a hash of everything
the arrays depend on (video, model files, perception parameters, format version), so a
change to any of them writes a new file instead of reusing a stale one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from matchlive.common.types import Track

FORMAT_VERSION = 4
ROLE_CODE = {"goalkeeper": 0, "player": 1}
ROLE_NAME = {code: role for role, code in ROLE_CODE.items()}


def _file_id(path: Path) -> list:
    # Name, size and a hash of the first and last MiB identify a video or model file without
    # hashing hundreds of MB. Unlike a modification time they survive a download to another
    # machine, so a shared cache stays valid there; re-exporting a model changes its bytes.
    size = path.stat().st_size
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    return [path.name, size, h.hexdigest()[:16]]


def perception_key(config: dict[str, Any], sources: list[Path]) -> str:
    """Hash of everything the cached arrays depend on: the input files (`sources`, e.g.
    the video), both model files, the perception parameters and the format version."""
    perception = config["perception"]
    ckpt = config["paths"]["checkpoints"]
    depends_on = {
        "format": FORMAT_VERSION,
        "sources": [_file_id(p) for p in sources],
        "detector": _file_id(ckpt / perception["detection"]["model"]),
        "jersey": _file_id(ckpt / perception["jersey"]["model"]),
        "confidence": perception["detection"]["confidence"],
        "shots": perception["shots"],
        "tracking": perception["tracking"],
    }
    return hashlib.sha1(json.dumps(depends_on, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(config: dict[str, Any], video: Path, match_key: str, half: int,
               start_frame: int, n_frames: int | None) -> Path:
    """cache/perception/<split>/<game>/half<h>_<span>_<key>.npz; `match_key` is Match.key."""
    span = "all" if (start_frame, n_frames) == (0, None) else f"{start_frame}+{'end' if n_frames is None else n_frames}"
    key = perception_key(config, [video])
    return config["paths"]["cache"] / "perception" / match_key / f"half{half}_{span}_{key}.npz"


def half_cache(config: dict[str, Any], match, half: int) -> Path:
    """The whole-half cache of `match` (common.match.Match) that scripts/run_perception.py writes."""
    path = cache_path(config, match.video_path(half), match.key, half, 0, None)
    if not path.exists():
        raise FileNotFoundError(f"no perception cache for {match.key} half {half} ({path}): run "
                                f"scripts/run_perception.py --split {match.split} --game {match.game} first")
    return path


def save_perception(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run never leaves a file that
    # looks complete.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
    tmp.replace(path)


def load_perception(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta


def iter_frames(arrays: dict[str, np.ndarray]) -> Iterator[tuple[int, list[Track], np.ndarray, np.ndarray, np.ndarray]]:
    """Decode cached arrays frame by frame, in frame order: (frame_idx, tracks, jersey,
    color, det) — the frame's Tracks (score 1.0: not cached), their jersey-probability and
    colour rows, and its detection rows (l, t, w, h, score, class_id). Frames with neither
    tracks nor detections are skipped."""
    track, jersey, color, det = arrays["track"], arrays["jersey"], arrays["color"], arrays["det"]
    # Rows are written in frame order, so each frame's rows form one contiguous run.
    frames = np.union1d(track[:, 0], det[:, 0]).astype(int)
    t_start = np.searchsorted(track[:, 0], frames, side="left")
    t_end = np.searchsorted(track[:, 0], frames, side="right")
    d_start = np.searchsorted(det[:, 0], frames, side="left")
    d_end = np.searchsorted(det[:, 0], frames, side="right")
    for f, ts, te, ds, de in zip(frames, t_start, t_end, d_start, d_end):
        tracks = [Track(int(r[1]), ROLE_NAME[int(r[2])], int(r[3]), tuple(float(v) for v in r[4:8]), 1.0)
                  for r in track[ts:te]]
        yield f, tracks, jersey[ts:te].astype(np.float32), color[ts:te], det[ds:de, 1:]
