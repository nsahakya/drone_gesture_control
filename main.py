from __future__ import annotations

import argparse
import sys
import time

import cv2

from gesture_control.gesture_drone_controller import GestureDroneController
from gesture_control.hand_tracker import HandTracker
from gesture_control.drone.mock import MockDrone
from gesture_control.drone.tello import TelloDrone


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Control a drone with hand gestures (MediaPipe Hands).")
    p.add_argument("--mode", choices=["mock", "tello"], default="mock", help="Drone backend")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--speed", type=int, default=50, help="RC speed (10..100)")
    p.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window")
    p.add_argument("--hold-arm", type=float, default=1.2, help="Hold OPEN_PALM to toggle arm")
    p.add_argument("--hold-takeoff", type=float, default=1.0, help="Hold ROCK to takeoff")
    p.add_argument("--hold-land", type=float, default=1.0, help="Hold FIST to land")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.mode == "tello":
        drone = TelloDrone(enable_video=False)
    else:
        drone = MockDrone()

    print(f"[INFO] Connecting drone backend: {args.mode}")
    drone.connect()

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
        print(f"[ERROR] Could not open camera index {args.camera}")
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
                # If there is no window, keep loop responsive
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
