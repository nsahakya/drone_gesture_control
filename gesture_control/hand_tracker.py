from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional
import urllib.request

import cv2
import mediapipe as mp

from .types import FingerState, HandObservation, Handedness


@dataclass
class HandTrackerConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_path: str = "models/hand_landmarker.task"
    auto_download_model: bool = True


class HandTracker:
    def __init__(self, config: HandTrackerConfig | None = None) -> None:
        self._config = config or HandTrackerConfig()

        # Newer MediaPipe Python builds (e.g., for Python 3.13) are "tasks-only" and
        # don't expose mp.solutions. Use Tasks API (HandLandmarker) instead.
        model_path = Path(self._config.model_path)
        _ensure_hand_landmarker_model(model_path, auto_download=self._config.auto_download_model)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._config.max_num_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        self._drawer = mp.tasks.vision.drawing_utils
        self._styles = mp.tasks.vision.drawing_styles
        self._connections = mp.tasks.vision.HandLandmarksConnections

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr, draw: bool = True) -> tuple[Optional[HandObservation], Any]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        obs: Optional[HandObservation] = None

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            handedness: Handedness = "Unknown"
            if result.handedness:
                label = result.handedness[0][0].category_name
                if label in ("Left", "Right"):
                    handedness = label

            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            fingers, thumb_vertical = _finger_state_from_landmarks(landmarks, handedness)
            obs = HandObservation(
                handedness=handedness,
                landmarks=landmarks,
                fingers=fingers,
                thumb_vertical=thumb_vertical,
            )

            if draw:
                self._drawer.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    self._connections.HAND_CONNECTIONS,
                    self._styles.get_default_hand_landmarks_style(),
                    self._styles.get_default_hand_connections_style(),
                )

        return obs, frame_bgr


_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_hand_landmarker_model(model_path: Path, *, auto_download: bool) -> None:
    if model_path.exists():
        return
    if not auto_download:
        raise RuntimeError(
            f"HandLandmarker model not found: {model_path}. "
            "Download the model bundle and place it there. "
            f"Suggested URL: {_DEFAULT_MODEL_URL}"
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_DEFAULT_MODEL_URL, model_path)  # noqa: S310
    except Exception as e:
        raise RuntimeError(
            "Failed to download MediaPipe HandLandmarker model bundle. "
            "Check internet access or download manually. "
            f"URL: {_DEFAULT_MODEL_URL}"
        ) from e


def _finger_state_from_landmarks(
    lms: list[tuple[float, float, float]],
    handedness: Handedness,
) -> tuple[FingerState, int]:
    # MediaPipe indices:
    # 0 wrist
    # thumb: 1-4, index: 5-8, middle: 9-12, ring: 13-16, pinky: 17-20
    def lm(i: int) -> tuple[float, float, float]:
        return lms[i]

    # For fingers (except thumb): tip.y < pip.y means "up" in image coords.
    index_up = lm(8)[1] < lm(6)[1]
    middle_up = lm(12)[1] < lm(10)[1]
    ring_up = lm(16)[1] < lm(14)[1]
    pinky_up = lm(20)[1] < lm(18)[1]

    # Thumb: compare x depending on handedness.
    thumb_tip_x = lm(4)[0]
    thumb_ip_x = lm(3)[0]
    if handedness == "Right":
        thumb_up = thumb_tip_x > thumb_ip_x
    elif handedness == "Left":
        thumb_up = thumb_tip_x < thumb_ip_x
    else:
        # Fallback: distance in x
        thumb_up = abs(thumb_tip_x - thumb_ip_x) > 0.02

    # Thumb vertical direction (only meaningful when thumb is extended)
    thumb_vertical = 0
    if thumb_up:
        thumb_mcp = lm(2)
        thumb_tip = lm(4)
        dx = thumb_tip[0] - thumb_mcp[0]
        dy = thumb_tip[1] - thumb_mcp[1]
        if abs(dy) > abs(dx) * 1.2:  # mostly vertical
            thumb_vertical = -1 if dy < 0 else 1

    return FingerState(
        thumb=thumb_up,
        index=index_up,
        middle=middle_up,
        ring=ring_up,
        pinky=pinky_up,
    ), thumb_vertical
