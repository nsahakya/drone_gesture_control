from __future__ import annotations

from .base import DroneBase, RcCommand


class TelloDrone(DroneBase):
    def __init__(self, *, enable_video: bool = False) -> None:
        try:
            from djitellopy import Tello
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "djitellopy is not installed or failed to import. "
                "Install it with: pip install djitellopy"
            ) from e

        self._tello = Tello()
        self._enable_video = enable_video

    def connect(self) -> None:
        self._tello.connect()
        # Useful to confirm connection
        _ = self._tello.get_battery()
        if self._enable_video:
            self._tello.streamon()

    def takeoff(self) -> None:
        self._tello.takeoff()

    def land(self) -> None:
        self._tello.land()

    def send_rc_control(self, cmd: RcCommand) -> None:
        self._tello.send_rc_control(cmd.lr, cmd.fb, cmd.ud, cmd.yaw)

    def end(self) -> None:
        try:
            if self._enable_video:
                self._tello.streamoff()
        finally:
            self._tello.end()
