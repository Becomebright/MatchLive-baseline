"""Player tracking with BoT-SORT from the `trackers` library, plus per-frame ball selection.

One tracker for outfield players and goalkeepers together; each track's role is the
majority class of the detections it has matched (a per-role tracker would break a player in
two whenever his detected class flickers: on SoccerNet-GSR 18% of keeper detections come out
as "player"). All tracking restarts at every shot cut and tracks carry the shot id; the ball
is not tracked (see select_ball). Synchronous: feed one frame's detections, get back that
frame's tracks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

from matchlive.common.types import Detection, Track
from matchlive.perception.detection import CLASS_ID_TO_ROLE


def _to_supervision(detections: list[Detection]):
    # The trackers consume supervision's Detections (xyxy + confidence + class_id),
    # so adapt our ltwh Detection list into that format.
    if not detections:
        return sv.Detections.empty()
    xyxy = np.asarray(
        [(x, y, x + w, y + h) for x, y, w, h in (d.bbox_ltwh for d in detections)],
        dtype=np.float32,
    )
    confidence = np.asarray([d.score for d in detections], dtype=np.float32)
    class_id = np.asarray([d.class_id for d in detections], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


class PlayerTracker:
    def __init__(self, config: dict[str, Any], frame_rate: float):
        # `config` is the perception.tracking section of the YAML config: BoTSORTTracker's
        # keyword arguments.
        self.tracker = BoTSORTTracker(frame_rate=float(frame_rate), **config)
        # Per track id (within the current shot): the roles of the detections it matched.
        self.role_votes: dict[int, Counter] = defaultdict(Counter)
        self.shot_id: int | None = None

    def update(self, detections: list[Detection], frame: np.ndarray, shot_id: int) -> list[Track]:
        # A new shot id means the camera cut: positions in the new shot have nothing to do
        # with the old one, so tracking starts over instead of matching across the cut.
        if shot_id != self.shot_id:
            self.tracker.reset()
            self.role_votes.clear()
            self.shot_id = shot_id

        people = _to_supervision([d for d in detections if d.role in ("player", "goalkeeper")])
        # `frame` feeds BoT-SORT's camera-motion compensation.
        tracked = self.tracker.update(people, frame)
        tracks: list[Track] = []
        for bbox, score, class_id, track_id in zip(tracked.xyxy, tracked.confidence,
                                                   tracked.class_id, tracked.tracker_id):
            # -1 marks a detection not yet promoted to a confirmed track
            # (still within minimum_consecutive_frames).
            if int(track_id) == -1:
                continue
            votes = self.role_votes[int(track_id)]
            votes[CLASS_ID_TO_ROLE[int(class_id)]] += 1
            x1, y1, x2, y2 = (float(v) for v in bbox.tolist())
            tracks.append(Track(
                shot_id=shot_id,
                role=votes.most_common(1)[0][0],
                track_id=int(track_id),
                bbox_ltwh=(x1, y1, x2 - x1, y2 - y1),
                score=float(score),
            ))
        return tracks


def select_ball(detections: list[Detection], min_score: float) -> Detection | None:
    """The frame's ball: its highest-scoring ball detection above `min_score`, or None.

    There is only one ball, so there is no identity to keep across frames and nothing
    for a tracker to associate. Measured on SoccerNet-GSR (one ball per frame, hit =
    centre within 15 px): top-1 at score > 0.3 gives precision 0.84 / recall 0.76, while
    tracking the ball drops recall — a small fast ball rarely clears a tracker's
    track-start confidence and IoU gates.
    """
    balls = [d for d in detections if d.role == "ball" and d.score > min_score]
    return max(balls, key=lambda d: d.score, default=None)
