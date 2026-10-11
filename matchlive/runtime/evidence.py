"""Evidence tools the runtime lends the arbiter: frames, a track's pictures,
its jersey reads, and read-only memory queries.

Everything is bounded by what was known when the case opened (`t_bound`): no frame, track row
or memory entry from later is ever returned. Perception comes from the raw cache (no model
is loaded); frames come straight from the half videos.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np

from matchlive.arbiter.agent import Evidence, event_line
from matchlive.common.cache import half_cache
from matchlive.events.analyzer import clock
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Time, track_name
from matchlive.perception.jersey import clear_number
from matchlive.perception.visualization import draw_ids, frame_jpeg


def read_frames(video: Path, frames: Iterable[int]) -> Iterator[tuple[int, np.ndarray]]:
    """(frame index, BGR image) for each of `frames` (indices into the half video) that can be
    read, seeking to each in turn."""
    cap = cv2.VideoCapture(str(video))
    try:
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if ok:
                yield f, image
    finally:
        cap.release()


class EvidenceTools:
    def __init__(self, config: dict[str, Any], match: Match):
        self.config, self.cfg = config, config["arbiter"]
        self.min_prob = config["perception"]["identity"]["jersey_min_prob"]
        self.match = match
        self.fps: dict[int, float] = {}
        self.rows: dict[int, np.ndarray] = {}         # half -> track rows (frame, shot, role, track, l, t, w, h)
        self.jersey: dict[int, np.ndarray] = {}
        self.paths: dict[int, Any] = {}
        self._tracks: dict[tuple, np.ndarray] = {}    # (half, shot, track) -> indices of its rows, in frame order

    def _load(self, half: int) -> None:
        # A half's cache is loaded on first use.
        if half in self.rows:
            return
        # Only the track rows are read here (the jersey probabilities are 6x larger and only
        # jersey_reads needs them, so they load on first use there).
        self.paths[half] = half_cache(self.config, self.match, half)
        with np.load(self.paths[half]) as data:
            self.fps[half] = json.loads(str(data["meta"]))["fps"]
            self.rows[half] = data["track"]
        keys = self.rows[half][:, 1:4:2].astype(np.int64)            # (shot, track)
        order = np.lexsort((self.rows[half][:, 0], keys[:, 1], keys[:, 0]))
        cuts = np.flatnonzero(np.any(np.diff(keys[order], axis=0) != 0, axis=1)) + 1
        for idx in np.split(order, cuts):
            shot, track = keys[idx[0]]
            self._tracks[(half, int(shot), int(track))] = idx

    def _track(self, key: tuple) -> np.ndarray | None:
        if key[0] not in (1, 2):
            return None
        self._load(key[0])
        return self._tracks.get(tuple(key))

    # ---- perception checks for the verdict's guardrails ----

    def track_frames(self, key: tuple, t_bound: Time) -> np.ndarray:
        """Frame indices where track `key` was seen, up to t_bound."""
        idx = self._track(key)
        if idx is None:
            return np.zeros(0, np.int64)
        frames = self.rows[key[0]][idx, 0].astype(np.int64)
        return frames[self._before(key[0], frames, t_bound)]

    def cooccur(self, a: tuple, b: tuple, t_bound: Time) -> bool:
        return a[0] == b[0] and bool(np.intersect1d(self.track_frames(a, t_bound), self.track_frames(b, t_bound)).size)

    def _before(self, half: int, frames: np.ndarray, t_bound: Time) -> np.ndarray:
        if half < t_bound[0]:
            return np.ones(len(frames), bool)
        if half > t_bound[0]:
            return np.zeros(len(frames), bool)
        return frames / self.fps[half] <= t_bound[1]

    # ---- the tools ----

    def call(self, mem: MatchMemory, t_bound: Time, tool: str, args: Mapping[str, Any]) -> Evidence:
        try:
            if tool == "get_frames":
                return self._frames(t_bound, int(args["half"]), float(args["start"]), float(args["end"]))
            if tool == "get_track_crops":
                return self._crops(t_bound, tuple(int(x) for x in args["track"]))
            if tool == "jersey_reads":
                return self._jersey_reads(t_bound, tuple(int(x) for x in args["track"]))
            if tool == "list_events":
                return self._list_events(mem, t_bound, int(args["half"]), float(args["start"]), float(args["end"]))
            if tool == "why":
                return self._why(mem, t_bound, str(args["entry"]))
        except (KeyError, TypeError, ValueError, IndexError) as e:
            return Evidence(f"error: bad arguments for {tool} ({type(e).__name__}: {e})")
        return Evidence(f"error: unknown tool {tool!r}")

    def _frames(self, t_bound: Time, half: int, start: float, end: float) -> Evidence:
        if half > t_bound[0]:
            return Evidence("error: that half had not been played when the case opened")
        if half == t_bound[0]:
            end = min(end, t_bound[1])
        if end <= start:
            return Evidence("error: empty span (nothing after the time the case opened can be seen)")
        self._load(half)
        fps, rows = self.fps[half], self.rows[half]
        # Frames at least 0.5 s apart, so no two seconds share a frame.
        n = int(min(self.cfg["max_frames"], (end - start) * 2 + 1))
        seconds = {int(round(s * fps)): s for s in np.linspace(start, end, n)}
        images, clocks = [], []
        for f, frame in read_frames(self.match.video_path(half), seconds):
            lo, hi = np.searchsorted(rows[:, 0], f, "left"), np.searchsorted(rows[:, 0], f, "right")
            frame = draw_ids(frame, [(track_name((int(r[1]), int(r[3]))), tuple(r[4:8])) for r in rows[lo:hi]])
            cv2.putText(frame, clock(seconds[f]), (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            images.append(frame_jpeg(frame))
            clocks.append(clock(seconds[f]))
        return Evidence(f"{len(images)} frames of half {half}, {clocks[0] if clocks else ''}–{clocks[-1] if clocks else ''}; "
                        f"tracked players are labelled shot/track", images)

    def _crops(self, t_bound: Time, key: tuple) -> Evidence:
        idx = self._track(key)
        if idx is None:
            return Evidence(f"error: no track {track_name(key)}")
        half, rows = key[0], self.rows[key[0]]
        idx = idx[self._before(half, rows[idx, 0], t_bound)]
        if not len(idx):
            return Evidence("error: that track had not been seen when the case opened")
        pick = idx[np.linspace(0, len(idx) - 1, min(self.cfg["crops"], len(idx))).round().astype(int)]
        boxes = {int(r[0]): r[4:8] for r in rows[pick]}        # one row per frame of a track
        tiles = []
        for f, frame in read_frames(self.match.video_path(half), boxes):
            x, y, w, h = boxes[f]
            # The whole box with a margin: the shirt, the name on the back and the build all help.
            x0, y0 = max(0, int(x - 0.2 * w)), max(0, int(y - 0.1 * h))
            crop = frame[y0:int(y + 1.1 * h), x0:int(x + 1.2 * w)]
            if crop.size:
                tiles.append(cv2.resize(crop, (int(200 * crop.shape[1] / crop.shape[0]), 200)))
        if not tiles:
            return Evidence("error: no picture of that track")
        ok, jpg = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        s0, s1 = rows[pick[0], 0] / self.fps[half], rows[pick[-1], 0] / self.fps[half]
        return Evidence(f"{len(tiles)} pictures of track {track_name(key)} between {clock(s0)} and {clock(s1)}",
                        [jpg.tobytes()])

    def _jersey_reads(self, t_bound: Time, key: tuple) -> Evidence:
        idx = self._track(key)
        if idx is None:
            return Evidence(f"error: no track {track_name(key)}")
        half = key[0]
        if half not in self.jersey:
            with np.load(self.paths[half]) as data:
                self.jersey[half] = data["jersey"]
        idx = idx[self._before(half, self.rows[half][idx, 0], t_bound)]
        probs = self.jersey[half][idx].astype(np.float32)
        probs = probs[~np.isnan(probs[:, 0])]
        clear = [n for n in (clear_number(p, self.min_prob) for p in probs) if n is not None]
        # Most read first; ties in the order first read.
        reads = ", ".join(f"#{n} ×{c}" for n, c in Counter(clear).most_common()) or "none"
        return Evidence(f"track {track_name(key)}: seen in {len(idx)} frames, {len(probs)} readable crops; "
                        f"numbers read clearly (probability ≥ {self.min_prob}): {reads}")

    def _list_events(self, mem: MatchMemory, t_bound: Time, half: int, start: float, end: float) -> Evidence:
        view = mem.at(t_bound, known_by=t_bound)
        lines = [event_line(r) for r in view.events(since=(half, start), until=(half, end))]
        return Evidence(f"events in half {half} {clock(start)}–{clock(end)}:\n" + ("\n".join(lines) or "(none)"))

    def _why(self, mem: MatchMemory, t_bound: Time, entry: str) -> Evidence:
        if entry not in mem.entries:
            return Evidence(f"error: no entry {entry}")
        past = mem.known_by(t_bound)
        js = "; ".join(f"{j.obs_id} ({j.kind}, {'holds' if past.holds(j) else 'rejected'})"
                       for _, j in past.justifications[entry])
        return Evidence(f"{entry} [{past.status(entry)}] rests on: {js}; open cases: "
                        f"{', '.join(past.open_cases(entry)) or 'none'}")
