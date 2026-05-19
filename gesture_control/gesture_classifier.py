from __future__ import annotations

from .types import Gesture, HandObservation


def classify_gesture(obs: HandObservation | None) -> Gesture:
    if obs is None:
        return Gesture.NONE

    f = obs.fingers
    thumb, index, middle, ring, pinky = f.as_tuple()

    # Strict patterns first
    if not any((thumb, index, middle, ring, pinky)):
        return Gesture.FIST

    if all((thumb, index, middle, ring, pinky)):
        return Gesture.OPEN_PALM

    if index and not any((thumb, middle, ring, pinky)):
        return Gesture.POINT

    if index and middle and not any((thumb, ring, pinky)):
        return Gesture.TWO

    if index and middle and ring and not any((thumb, pinky)):
        return Gesture.THREE

    if index and middle and ring and pinky and not thumb:
        return Gesture.FOUR

    if pinky and not any((thumb, index, middle, ring)):
        return Gesture.PINKY

    if index and pinky and not any((thumb, middle, ring)):
        return Gesture.ROCK

    # Thumb-only with direction
    if thumb and not any((index, middle, ring, pinky)):
        if obs.thumb_vertical == -1:
            return Gesture.THUMB_UP
        if obs.thumb_vertical == 1:
            return Gesture.THUMB_DOWN

    return Gesture.NONE
