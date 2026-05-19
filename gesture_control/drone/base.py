from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RcCommand:
    lr: int  # left/right
    fb: int  # forward/back
    ud: int  # up/down
    yaw: int  # yaw left/right


class DroneBase:
    def connect(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def takeoff(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def land(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def send_rc_control(self, cmd: RcCommand) -> None:  # pragma: no cover
        raise NotImplementedError

    def stop(self) -> None:
        self.send_rc_control(RcCommand(0, 0, 0, 0))

    def end(self) -> None:  # pragma: no cover
        raise NotImplementedError
