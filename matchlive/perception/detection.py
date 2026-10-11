"""RF-DETR ONNX detector for ball, goalkeepers and players.

Runs on ONNX Runtime with TensorRT first (FP16, engine cached on disk), then CUDA, then CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from matchlive.common.types import Detection

# RF-DETR class ids for this soccer model (checkpoint class_names are
# ['ball', 'goalkeeper', 'referee', 'player']). Referees (id 2) are dropped;
# the values are the only role names used in the package.
CLASS_ID_TO_ROLE = {0: "ball", 1: "goalkeeper", 3: "player"}

# ImageNet normalization used at training time, folded into one multiply-subtract per
# channel on 0-255 input: (x / 255 - mean) / std == x * scale - offset.
_INPUT_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_INPUT_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SCALE = 1.0 / (255.0 * _INPUT_STD)
_INPUT_OFFSET = _INPUT_MEAN / _INPUT_STD


def normalize_into(bgr: np.ndarray, out: np.ndarray) -> None:
    """Write a uint8 BGR image into `out` (3, H, W) as normalized RGB, one fused pass per
    channel. The straightforward astype/transpose/(x-mean)/std chain made three full-size
    copies of a strided array and cost ~8 ms per detector frame; this costs ~1.3 ms with the
    same values."""
    for c in range(3):
        np.multiply(bgr[:, :, 2 - c], _INPUT_SCALE[c], out=out[c])
        out[c] -= _INPUT_OFFSET[c]


def to_detections(boxes_ltwh: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                  confidence: float) -> list[Detection]:
    """The detector's output filter: keep candidates above `confidence` whose class has a role.

    Shared by RFDETRDetector.predict and by anything replaying raw candidates from a cache,
    so the confidence cut and the referee drop live in one place.
    """
    records: list[Detection] = []
    for box, score, class_id in zip(boxes_ltwh, scores, class_ids):
        role = CLASS_ID_TO_ROLE.get(int(class_id))
        if role is not None and score > confidence:
            records.append(Detection(role, tuple(float(v) for v in box), float(score), int(class_id)))
    return records


def session_providers(config: dict[str, Any], cache_dir: Path, trt_options: dict | None = None) -> list:
    """Build ORT's provider list from the config's preference order, skipping unavailable ones.

    `config` needs `providers`, `fp16` and `trt_cache` (engine dir under `cache_dir`);
    `trt_options` adds TensorRT EP options, e.g. the shape profile of a dynamic batch.
    """
    # CUDA/cuDNN come from the pip nvidia-* wheels; ORT only finds them after this call.
    ort.preload_dlls()
    available = set(ort.get_available_providers())
    providers: list = []
    for name in config["providers"]:
        if name == "tensorrt" and "TensorrtExecutionProvider" in available:
            # ORT's TensorRT EP dlopens libnvinfer* by soname, but the pip tensorrt wheel
            # keeps them in its own package dir, off the loader path. Importing the wheel
            # loads them all, so ORT's later dlopen calls find the loaded copies.
            try:
                import tensorrt_libs  # noqa: F401
            except ImportError:
                continue
            engine_dir = cache_dir / config["trt_cache"]
            engine_dir.mkdir(parents=True, exist_ok=True)
            # Building an engine takes minutes, so both the engine and TensorRT's kernel
            # timing cache (shared by all engines, one level up) are persisted; later
            # sessions with the same model and options load the engine straight from disk.
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": bool(config["fp16"]),
                    "trt_max_workspace_size": 2 << 30,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(engine_dir),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(engine_dir.parent),
                    **(trt_options or {}),
                },
            ))
        elif name == "cuda" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        elif name == "cpu":
            providers.append("CPUExecutionProvider")
    return providers


class RFDETRDetector:
    def __init__(self, config: dict[str, Any], checkpoints_dir: Path, cache_dir: Path):
        # `config` is the perception.detection section of the YAML config.
        self.confidence = float(config["confidence"])
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(checkpoints_dir / config["model"]),
            sess_options=options,
            providers=session_providers(config, cache_dir),
        )
        # ORT silently falls back when a provider fails to initialise, so record what
        # is actually running.
        self.providers = self.session.get_providers()

        self.input_name = self.session.get_inputs()[0].name
        # Read the fixed input resolution off the model so _preprocess can resize to it.
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape):
            raise ValueError(f"Expected static NCHW ONNX input shape, got {shape}")
        _, _, self.input_height, self.input_width = (int(v) for v in shape)
        # Reused input buffer: preprocessing writes straight into it every frame.
        self.input = np.empty((1, 3, self.input_height, self.input_width), dtype=np.float32)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        boxes, logits = self.session.run(["dets", "labels"], {self.input_name: self._preprocess(frame)})
        # [0] drops the batch dimension since we always run a single frame.
        boxes, logits = boxes[0], logits[0]

        # logits has shape (num_queries, num_classes) and the head is sigmoid, so
        # a query can score high on more than one class at once. Keep only each
        # query's best-scoring class: a query slot describes one object, and our
        # roles are mutually exclusive — a box is a ball, a player, or a keeper,
        # never two of them.
        #
        # Scoring every (query, class) pair independently (the usual DETR-style
        # postprocess) instead emits one physical box twice. Measured on our videos: a player
        # box at 0.72 also surfaced as a goalkeeper box at 0.23, on the same query,
        # in 2 of 20 sampled frames. That phantom is too weak to start a track of its
        # own, but the tracker's low-confidence second association stage can still
        # match it to the keeper's track and drag that track onto an outfield player.
        scores_per_class = 1.0 / (1.0 + np.exp(-logits))
        class_ids = scores_per_class.argmax(axis=1)
        scores = scores_per_class[np.arange(len(class_ids)), class_ids]

        # RF-DETR boxes are (center_x, center_y, width, height) normalized to [0, 1];
        # convert to ltwh in pixels of this frame.
        height, width = frame.shape[:2]
        wh = np.clip(boxes[:, 2:4], 0.0, None)
        ltwh = np.concatenate([boxes[:, 0:2] - wh / 2, wh], axis=1) * np.asarray(
            [width, height, width, height], dtype=np.float32)
        return to_detections(ltwh, scores, class_ids, self.confidence)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HWC BGR frame with 3 channels, got shape {frame.shape}")
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        normalize_into(resized, self.input[0])
        return self.input
