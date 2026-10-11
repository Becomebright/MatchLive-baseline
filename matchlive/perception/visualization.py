"""What the models are shown of a frame: ID labels over the tracked players, and who is
nearest the ball."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from matchlive.common.types import Detection


def draw_ids(frame: np.ndarray, boxes: Sequence[tuple[str, tuple[float, float, float, float]]]) -> np.ndarray:
    """Only an ID label ("P3") above each (label, ltwh box), so the picture stays uncluttered
    and the identities travel as text instead of being drawn."""
    out = frame.copy()
    for label, (x, y, w, h) in boxes:
        _centered_text(out, label, (int(round(x + w / 2)), max(10, int(round(y)) - 10)), 0.6)
    return out


def frame_jpeg(image: np.ndarray) -> bytes:
    """A frame as the models are shown it: 960x540 keeps the ID labels legible, JPEG quality 80
    keeps the request small."""
    ok, jpg = cv2.imencode(".jpg", cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes()


def nearest_to_ball(players: Sequence[Detection], ball_point: tuple[int, int] | None) -> Detection | None:
    # Player whose feet (box bottom-center) are closest to the ball point.
    # Squared distance is enough for an argmin; no need for the sqrt.
    if ball_point is None or not players:
        return None
    bx, by = ball_point
    return min(
        players,
        key=lambda t: (t.bbox_ltwh[0] + t.bbox_ltwh[2] / 2 - bx) ** 2
        + (t.bbox_ltwh[1] + t.bbox_ltwh[3] - by) ** 2,
    )


def _centered_text(img, text: str, center: tuple[int, int], font_scale: float):
    # Draw white text centered on `center`. The 8 offset passes paint a black
    # outline first so the label stays legible against any background.
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    x, y = center[0] - tw // 2, center[1] + th // 2
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, (0, 0, 0), 2)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), 2)
