from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from .simulated import SimulatedDrone


class DroneVisualizer3D:
    def __init__(self, drone: SimulatedDrone):
        self.drone = drone

        # Включаем интерактивный режим
        plt.ion()
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 5)
        self.ax.set_xlabel('X (влево/вправо)')
        self.ax.set_ylabel('Z (вперёд/назад)')
        self.ax.set_zlabel('Y (высота)')
        self.ax.set_title('3D симуляция дрона')

        # Графические объекты
        self.drone_point, = self.ax.plot([0], [0], [0], 'ro', markersize=8)
        self.arrow = None
        self.trail, = self.ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1)
        self.trail_points = []   # список (x, y, z)

        plt.show(block=False)

    def update(self):
        """Вызывается из главного цикла для обновления визуализации."""
        state = self.drone.state
        x, y, z = state.x, state.y, state.z
        yaw_rad = np.radians(state.yaw)

        # Обновляем точку дрона
        self.drone_point.set_data([x], [z])        # X и Y на графике
        self.drone_point.set_3d_properties([y])

        # Удаляем старую стрелку и рисуем новую
        if self.arrow:
            self.arrow.remove()
        dx = 0.5 * np.cos(yaw_rad)
        dz = 0.5 * np.sin(yaw_rad)
        self.arrow = self.ax.quiver(x, z, y, dx, dz, 0, color='r', length=0.5, normalize=False)

        # След
        self.trail_points.append((x, y, z))
        if len(self.trail_points) > 50:
            self.trail_points.pop(0)
        xs = [p[0] for p in self.trail_points]
        zs = [p[2] for p in self.trail_points]   # ось Z графика
        ys = [p[1] for p in self.trail_points]
        self.trail.set_data(xs, zs)
        self.trail.set_3d_properties(ys)

        # Заголовок
        self.ax.set_title(f'Дрон | x={x:.2f} y={y:.2f} z={z:.2f} yaw={state.yaw:.0f}° | flying={state.flying}')

        # Принудительная перерисовка и небольшая пауза для обработки событий
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)
