"""
Live lane monitor over a printed 3-lane road, using YOLO tracking.

Requires lanes.json produced by calibrate_lanes.py.

Rules:
  - A moving vehicle  -> "Running on lane X" (or "Running on lane X-Y" if it spans lanes)
  - A stationary vehicle -> "STOPPED at lane X" (or X-Y if it spans lanes)
  - Any non-vehicle object overlapping a lane -> "Lane X blocked"

Press 'q' to quit.
"""

import json
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

CAM_INDEX = 0
MODEL_NAME = "yolo11n.pt"

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

LANE_OVERLAP_THRESHOLD = 0.15   # fraction of box area inside a lane to count as occupying it
HISTORY_LEN = 15                # frames of centroid history for speed estimation
STOP_SPEED_THRESHOLD = 2.0      # avg px/frame displacement below this = STOPPED


def load_lanes(path="lanes.json"):
    with open(path) as f:
        data = json.load(f)
    return [np.array(lane, dtype=np.int32) for lane in data["lanes"]]


def build_lane_masks(lanes, frame_shape):
    h, w = frame_shape[:2]
    masks = []
    for lane in lanes:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [lane], 1)
        masks.append(mask)
    return masks


def lanes_for_box(box, lane_masks):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    box_area = max((x2 - x1) * (y2 - y1), 1)
    occupied = []
    for idx, mask in enumerate(lane_masks):
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            continue
        overlap = int(region.sum())
        if overlap / box_area >= LANE_OVERLAP_THRESHOLD:
            occupied.append(idx)
    return occupied


def format_lane_range(lane_indices):
    lanes_1based = sorted(i + 1 for i in lane_indices)
    if not lanes_1based:
        return ""
    if lanes_1based == list(range(lanes_1based[0], lanes_1based[-1] + 1)):
        if len(lanes_1based) == 1:
            return str(lanes_1based[0])
        return f"{lanes_1based[0]}-{lanes_1based[-1]}"
    return ",".join(str(n) for n in lanes_1based)


def main():
    lanes = load_lanes()
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    lane_masks = None
    track_history = {}  # id -> deque of centroids

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if lane_masks is None:
            lane_masks = build_lane_masks(lanes, frame.shape)

        results = model.track(frame, persist=True, verbose=False)[0]

        messages = []
        blocked_lanes = set()

        overlay = frame.copy()
        for idx, lane in enumerate(lanes):
            cv2.polylines(overlay, [lane], True, (0, 200, 255), 2)
            cv2.putText(overlay, f"Lane {idx + 1}", tuple(lane[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                class_name = model.names[cls_id]
                occupied = lanes_for_box(box, lane_masks)
                if not occupied:
                    continue

                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                color = (0, 255, 0)

                if class_name in VEHICLE_CLASSES:
                    hist = track_history.setdefault(track_id, deque(maxlen=HISTORY_LEN))
                    hist.append((cx, cy))

                    if len(hist) >= 2:
                        dists = [
                            np.hypot(hist[i][0] - hist[i - 1][0], hist[i][1] - hist[i - 1][1])
                            for i in range(1, len(hist))
                        ]
                        avg_speed = sum(dists) / len(dists)
                    else:
                        avg_speed = None

                    lane_str = format_lane_range(occupied)
                    if avg_speed is None:
                        label = f"lane {lane_str}..."
                    elif avg_speed < STOP_SPEED_THRESHOLD:
                        label = f"STOPPED at lane {lane_str}"
                        color = (0, 0, 255)
                    else:
                        label = f"Running on lane {lane_str}"
                    messages.append(label)
                else:
                    lane_str = format_lane_range(occupied)
                    for i in occupied:
                        blocked_lanes.add(i)
                    label = f"{class_name} blocking lane {lane_str}"
                    color = (0, 0, 255)
                    messages.append(f"Lane {lane_str} blocked ({class_name})")

                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(overlay, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        y = 80
        for msg in dict.fromkeys(messages):
            cv2.putText(overlay, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25

        cv2.imshow("Lane Monitor", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
