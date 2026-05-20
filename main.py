from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _warn_if_not_using_venv() -> None:
    project_root = Path(__file__).resolve().parent
    venv_dir = project_root / ".venv"
    if venv_dir.exists() and venv_dir.is_dir() and str(venv_dir) not in sys.prefix:
        print("[WARN] You are not using the project venv (.venv).")
        print("       Recommended:")
        print("         source .venv/bin/activate")
        print("         python -m pip install -r requirements.txt")


def _print_camera_help(camera_index: int) -> None:
    print(f"[ERROR] Could not open camera index {camera_index}")
    print("       Tips:")
    print("         - Try another index: --camera 1 (or 2, 3...)")
    print("         - Check devices: ls /dev/video*")
    print("         - If running in WSL/container/remote session, camera may be unavailable")


def _print_tello_help() -> None:
    print("[ERROR] Tello did not respond to 'command'.")
    print("       Checklist:")
    print("         1) Connect your PC to the Tello Wi‑Fi network")
    print("         2) Disable VPN / proxy that may block local UDP")
    print("         3) Firewall: allow UDP to 192.168.10.1:8889")
    print("         4) Make sure only one program is controlling Tello")
    print("       Quick test:")
    print("         python -c \"from djitellopy import Tello; t=Tello(); t.connect(); print(t.get_battery())\"")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Control a drone with hand gestures (MediaPipe Hands).")
    p.add_argument("--mode", choices=["mock", "tello", "simulated"], default="mock", help="Drone backend")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--speed", type=int, default=50, help="RC speed (10..100)")
    p.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window")
    p.add_argument("--hold-arm", type=float, default=1.2, help="Hold OPEN_PALM to toggle arm")
    p.add_argument("--hold-takeoff", type=float, default=1.0, help="Hold ROCK to takeoff")
    p.add_argument("--hold-land", type=float, default=1.0, help="Hold FIST to land")
    return p


def main() -> int:
    _warn_if_not_using_venv()
    args = build_arg_parser().parse_args()

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        print("[ERROR] Missing dependency: opencv (cv2)")
        print("       Install with: python -m pip install -r requirements.txt")
        return 1

    try:
        from gesture_control.gesture_drone_controller import GestureDroneController
        from gesture_control.hand_tracker import HandTracker
        from gesture_control.drone.mock import MockDrone
        from gesture_control.drone.tello import TelloDrone
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or "(unknown)"
        print(f"[ERROR] Missing dependency: {missing}")
        print("       Install with: python -m pip install -r requirements.txt")
        return 1

    # --- Выбор режима дрона (с поддержкой симуляции) ---
    visualizer = None
    if args.mode == "simulated":
        from gesture_control.drone.simulated import SimulatedDrone
        from gesture_control.drone.visualizer_3d import DroneVisualizer3D
        drone = SimulatedDrone()
        drone.connect()
        drone.start_physics()
        visualizer = DroneVisualizer3D(drone)
        print("[INFO] Запущена 3D симуляция дрона")
    elif args.mode == "tello":
        drone = TelloDrone(enable_video=False)
    else:  # mock
        drone = MockDrone()

    print(f"[INFO] Connecting drone backend: {args.mode}")
    try:
        # Для simulated connect уже вызван, но повторный вызов безвреден
        drone.connect()
    except Exception as e:
        if args.mode == "tello":
            _print_tello_help()
        print(f"[DETAILS] {type(e).__name__}: {e}")
        return 3

    tracker = HandTracker()
    controller = GestureDroneController(
        drone,
        speed=args.speed,
        hold_to_toggle_arm_s=args.hold_arm,
        hold_to_takeoff_s=args.hold_takeoff,
        hold_to_land_s=args.hold_land,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        _print_camera_help(args.camera)
        return 2

    print("[INFO] Controls:")
    print("  - Hold OPEN_PALM to toggle ARM")
    print("  - When ARMED: hold ROCK to TAKEOFF")
    print("  - When flying: hold FIST to LAND")
    print("  - Press 'q' to quit (lands if flying)")

    last_overlay = time.monotonic()
    last_gesture_str = "NONE"
    last_cmd_str = "lr=0 fb=0 ud=0 yaw=0"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera frame grab failed")
                break

            obs, frame = tracker.process(frame, draw=not args.no_window)
            gesture, cmd = controller.update(obs)

            # Обновляем 3D визуализатор (если запущена симуляция)
            if visualizer is not None:
                visualizer.update()

            now = time.monotonic()
            if now - last_overlay >= 0.05:
                last_overlay = now
                last_gesture_str = gesture.value
                last_cmd_str = f"lr={cmd.lr} fb={cmd.fb} ud={cmd.ud} yaw={cmd.yaw}"

            if not args.no_window:
                cv2.putText(
                    frame,
                    f"Gesture: {last_gesture_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"ARM={controller.armed} FLY={controller.flying} {last_cmd_str}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("gesture_control", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        controller.safe_shutdown()
        tracker.close()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        drone.end()
        if visualizer is not None:
            visualizer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
