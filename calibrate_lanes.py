"""
Calibrate 3 lane polygons against your printed road paper.

Usage:
  python3 calibrate_lanes.py

Steps:
  1. Position the printed paper in the camera view.
  2. Press SPACE to freeze the frame.
  3. Click 4 corners for Lane 1 (left), then 4 for Lane 2 (middle), then 4 for Lane 3 (right).
     Click each lane's corners in order: top-left, top-right, bottom-right, bottom-left.
  4. Press 's' to save to lanes.json, 'r' to restart clicking, 'q' to quit without saving.
"""

import cv2
import json

CAM_INDEX = 0
NUM_LANES = 3
POINTS_PER_LANE = 4

points = []
frozen_frame = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frozen_frame is not None:
        if len(points) < NUM_LANES * POINTS_PER_LANE:
            points.append((x, y))


def draw_overlay(base):
    img = base.copy()
    for i, p in enumerate(points):
        cv2.circle(img, p, 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for lane in range(NUM_LANES):
        lane_pts = points[lane * POINTS_PER_LANE:(lane + 1) * POINTS_PER_LANE]
        if len(lane_pts) == POINTS_PER_LANE:
            for i in range(4):
                cv2.line(img, lane_pts[i], lane_pts[(i + 1) % 4], (0, 200, 255), 2)
    total_needed = NUM_LANES * POINTS_PER_LANE
    lane_in_progress = len(points) // POINTS_PER_LANE + 1
    status = f"Points {len(points)}/{total_needed}  (lane {min(lane_in_progress, NUM_LANES)})"
    cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "s=save  r=reset  q=quit", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main():
    global frozen_frame, points

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", on_mouse)

    print("Press SPACE to freeze the frame, then click lane corners.")

    while True:
        if frozen_frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            display = frame.copy()
            cv2.putText(display, "Press SPACE to freeze", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Calibrate", display)
        else:
            cv2.imshow("Calibrate", draw_overlay(frozen_frame))

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and frozen_frame is None:
            ok, frame = cap.read()
            if ok:
                frozen_frame = frame
        elif key == ord('r'):
            points = []
        elif key == ord('q'):
            break
        elif key == ord('s'):
            if len(points) != NUM_LANES * POINTS_PER_LANE:
                print(f"Need {NUM_LANES * POINTS_PER_LANE} points, have {len(points)}.")
                continue
            lanes = [
                points[i * POINTS_PER_LANE:(i + 1) * POINTS_PER_LANE]
                for i in range(NUM_LANES)
            ]
            with open("lanes.json", "w") as f:
                json.dump({"lanes": lanes}, f, indent=2)
            print("Saved lanes.json")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
