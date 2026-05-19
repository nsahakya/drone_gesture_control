from __future__ import annotations

import time

from .gesture_classifier import classify_gesture
from .types import Gesture, HandObservation
from .drone.base import DroneBase, RcCommand


class GestureDroneController:
    def __init__(
        self,
        drone: DroneBase,
        *,
        speed: int = 50,
        hold_to_toggle_arm_s: float = 1.2,
        hold_to_takeoff_s: float = 1.0,
        hold_to_land_s: float = 1.0,
    ) -> None:
        self._drone = drone
        self._speed = int(max(10, min(100, speed)))
        self._hold_to_toggle_arm_s = hold_to_toggle_arm_s
        self._hold_to_takeoff_s = hold_to_takeoff_s
        self._hold_to_land_s = hold_to_land_s

        self.armed = False
        self.flying = False

        self._last_gesture = Gesture.NONE
        self._gesture_since = time.monotonic()
        self._action_consumed = False

    def update(self, obs: HandObservation | None) -> tuple[Gesture, RcCommand]:
        gesture = classify_gesture(obs)
        now = time.monotonic()

        if gesture != self._last_gesture:
            self._last_gesture = gesture
            self._gesture_since = now
            self._action_consumed = False

        held_s = now - self._gesture_since

        # Discrete actions (require holding). Only one action per continuous hold.
        if (
            not self._action_consumed
            and gesture == Gesture.OPEN_PALM
            and held_s >= self._hold_to_toggle_arm_s
        ):
            self.armed = not self.armed
            self._action_consumed = True

        if (
            not self._action_consumed
            and self.armed
            and not self.flying
            and gesture == Gesture.ROCK
            and held_s >= self._hold_to_takeoff_s
        ):
            self._drone.takeoff()
            self.flying = True
            self._action_consumed = True

        if (
            not self._action_consumed
            and self.armed
            and self.flying
            and gesture == Gesture.FIST
            and held_s >= self._hold_to_land_s
        ):
            self._drone.land()
            self.flying = False
            self._action_consumed = True

        cmd = self._continuous_command(gesture)
        self._drone.send_rc_control(cmd)
        return gesture, cmd

    def _continuous_command(self, gesture: Gesture) -> RcCommand:
        if not self.armed or not self.flying:
            return RcCommand(0, 0, 0, 0)

        s = self._speed

        if gesture == Gesture.POINT:
            return RcCommand(0, s, 0, 0)
        if gesture == Gesture.TWO:
            return RcCommand(0, -s, 0, 0)
        if gesture == Gesture.THREE:
            return RcCommand(-s, 0, 0, 0)
        if gesture == Gesture.FOUR:
            return RcCommand(s, 0, 0, 0)
        if gesture == Gesture.THUMB_UP:
            return RcCommand(0, 0, s, 0)
        if gesture == Gesture.THUMB_DOWN:
            return RcCommand(0, 0, -s, 0)
        if gesture == Gesture.PINKY:
            return RcCommand(0, 0, 0, -s)

        # OPEN_PALM / NONE / anything else => hover
        return RcCommand(0, 0, 0, 0)

    def safe_shutdown(self) -> None:
        try:
            self._drone.send_rc_control(RcCommand(0, 0, 0, 0))
        except Exception:
            pass
        if self.flying:
            try:
                self._drone.land()
            except Exception:
                pass
            self.flying = False
