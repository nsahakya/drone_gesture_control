from __future__ import annotations

import time

from .base import DroneBase, RcCommand


class MockDrone(DroneBase):
    def __init__(self) -> None:
        self._connected = False
        self._flying = False
        self._last_print = 0.0

    def connect(self) -> None:
        self._connected = True
        print("[MOCK] connect()")

    def takeoff(self) -> None:
        if not self._connected:
            raise RuntimeError("MockDrone not connected")
        self._flying = True
        print("[MOCK] takeoff()")

    def land(self) -> None:
        self._flying = False
        print("[MOCK] land()")

    def send_rc_control(self, cmd: RcCommand) -> None:
        now = time.monotonic()
        # Rate-limit console spam
        if now - self._last_print >= 0.2:
            self._last_print = now
            print(f"[MOCK] rc lr={cmd.lr} fb={cmd.fb} ud={cmd.ud} yaw={cmd.yaw}")

    def end(self) -> None:
        print("[MOCK] end()")
