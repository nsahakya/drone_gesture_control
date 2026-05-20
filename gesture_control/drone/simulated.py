from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .base import DroneBase, RcCommand


@dataclass
class DroneState:
    x: float = 0.0      # метры (влево/вправо)
    y: float = 0.0      # высота
    z: float = 0.0      # метры (вперёд/назад)
    yaw: float = 0.0    # градусы
    flying: bool = False


class SimulatedDrone(DroneBase):
    def __init__(self, update_hz: float = 30.0, max_speed: float = 1.5):
        self._state = DroneState()
        self._cmd = RcCommand(0, 0, 0, 0)
        self._speed = 50.0  # масштаб скорости из RcCommand (0..100 -> м/с)
        self._max_speed = max_speed
        self._update_interval = 1.0 / update_hz
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        print("[SIM] Дрон сконфигурирован (симуляция)")

    def takeoff(self) -> None:
        with self._lock:
            if not self._state.flying:
                self._state.flying = True
                self._state.y = 1.0   # взлетаем на 1 метр
                print(f"[SIM] Взлёт -> позиция ({self._state.x:.2f}, {self._state.y:.2f}, {self._state.z:.2f})")

    def land(self) -> None:
        with self._lock:
            if self._state.flying:
                self._state.flying = False
                self._state.y = 0.0
                self._cmd = RcCommand(0, 0, 0, 0)
                print(f"[SIM] Посадка -> позиция ({self._state.x:.2f}, 0, {self._state.z:.2f})")

    def send_rc_control(self, cmd: RcCommand) -> None:
        with self._lock:
            self._cmd = cmd

    def stop(self) -> None:
        self.send_rc_control(RcCommand(0, 0, 0, 0))

    def _physics_update(self):
        """Обновляет позицию по текущей команде (вызывается в отдельном потоке)."""
        while self._running:
            time.sleep(self._update_interval)
            with self._lock:
                if not self._state.flying:
                    continue
                # Преобразуем значения RC (0..100) в скорость (м/с)
                # lr = left/right, fb = forward/back, ud = up/down, yaw = угловая скорость (град/с)
                lr = self._cmd.lr / 100.0 * self._max_speed
                fb = self._cmd.fb / 100.0 * self._max_speed
                ud = self._cmd.ud / 100.0 * self._max_speed
                yaw_rate = self._cmd.yaw / 100.0 * 90.0   # 90 град/с при полной шкале

                dt = self._update_interval
                # Обновление позиции с учётом текущего рысканья
                import math
                yaw_rad = math.radians(self._state.yaw)
                # Вперёд/назад по оси Z (в локальной системе дрона)
                dz = fb * math.cos(yaw_rad) - lr * math.sin(yaw_rad)
                dx = fb * math.sin(yaw_rad) + lr * math.cos(yaw_rad)
                self._state.x += dx * dt
                self._state.z += dz * dt
                self._state.y += ud * dt
                self._state.yaw += yaw_rate * dt

                # Простая граница (куб ±5 метров)
                self._state.x = max(-5.0, min(5.0, self._state.x))
                self._state.z = max(-5.0, min(5.0, self._state.z))
                self._state.y = max(0.0, min(5.0, self._state.y))

    def start_physics(self):
        """Запускает фоновый поток для физики."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._physics_update, daemon=True)
        self._thread.start()

    def end(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[SIM] Симуляция завершена")

    @property
    def state(self) -> DroneState:
        with self._lock:
            # возвращаем копию
            return DroneState(
                x=self._state.x, y=self._state.y, z=self._state.z,
                yaw=self._state.yaw, flying=self._state.flying
            )
