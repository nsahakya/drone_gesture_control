from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Sequence


class Gesture(str, Enum):
    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    FIST = "FIST"
    POINT = "POINT"  # index finger
    TWO = "TWO"  # index + middle
    THREE = "THREE"  # index + middle + ring
    FOUR = "FOUR"  # index + middle + ring + pinky
    THUMB_UP = "THUMB_UP"
    THUMB_DOWN = "THUMB_DOWN"
    PINKY = "PINKY"
    ROCK = "ROCK"  # index + pinky


Handedness = Literal["Left", "Right", "Unknown"]


@dataclass(frozen=True)
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def count_up(self) -> int:
        return sum(self.as_tuple())


@dataclass(frozen=True)
class HandObservation:
    handedness: Handedness
    landmarks: Sequence[tuple[float, float, float]]  # normalized (x,y,z)
    fingers: FingerState
    thumb_vertical: int  # -1 up, +1 down, 0 unknown/sideways
