#!/usr/bin/env python3
"""
three_lane_yolo_monitor.py

YOLO-based 3-lane road monitor for a fixed camera.

Behavior:
- Vehicle moving in a lane:
      RUNNING - Lane 1
- Vehicle stopped:
      STOPPED - Lane 2
- Vehicle overlaps multiple lanes:
      RUNNING - Lane 1-2
      STOPPED - Lane 2-3
- A detected non-vehicle object on the road:
      BLOCKED - Lane 1
      BLOCKED - Lane 1-2

Designed for a fixed overhead / angled camera looking at a printed 3-lane road.

Install:
    pip install ultralytics opencv-python numpy

Examples:
    python three_lane_yolo_monitor.py --source 0
    python three_lane_yolo_monitor.py --source road_test.mp4
    python three_lane_yolo_monitor.py --source 0 --model yolov8n.pt

For toy cars / unusual objects, a custom-trained YOLO model may give much better
results than a standard COCO model. Pass the custom model with --model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Lane polygons are NORMALIZED coordinates: (x_ratio, y_ratio), each in [0, 1].
#
# Default layout assumes 3 roughly vertical lanes filling most of the image.
# Adjust these coordinates to match your printed road/camera.
#
# Lane 1 = left
# Lane 2 = center
# Lane 3 = right
#
# Example:
#   (0.05, 0.05) means x=5% of image width, y=5% of image height.
#
LANE_POLYGONS_NORMALIZED = {
    1: [(0.03, 0.05), (0.34, 0.05), (0.34, 0.95), (0.03, 0.95)],
    2: [(0.34, 0.05), (0.66, 0.05), (0.66, 0.95), (0.34, 0.95)],
    3: [(0.66, 0.05), (0.97, 0.05), (0.97, 0.95), (0.66, 0.95)],
}

# Standard COCO vehicle class names.
# If using a custom model, update this set if your custom vehicle class names differ.
VEHICLE_CLASS_NAMES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
}

# Minimum fraction of a detection bounding-box area that must overlap a lane
# for that detection to be considered inside that lane.
LANE_OVERLAP_THRESHOLD = 0.12

# Track center history length.
MOTION_HISTORY = 12

# Minimum center displacement, measured relative to the object's own size,
# to call it moving.
#
# Example:
#   0.10 means the center must move at least 10% of the object's diagonal
#   during the motion-history window.
MOTION_THRESHOLD_RATIO = 0.10

# A track must have at least this many positions before motion state is trusted.
MIN_HISTORY_FOR_STATE = 5

# If True, detections outside all lane polygons are ignored.
IGNORE_OBJECTS_OUTSIDE_ROAD = True


# ---------------------------------------------------------------------------
# DATA TYPES
# ---------------------------------------------------------------------------

@dataclass
class DetectionState:
    track_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]
    lanes: List[int]
    status: str


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalized_polygon_to_pixels(
    polygon: List[Tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    points = [
        (int(x * width), int(y * height))
        for x, y in polygon
    ]
    return np.array(points, dtype=np.int32)


def build_lane_polygons(width: int, height: int) -> Dict[int, np.ndarray]:
    return {
        lane_id: normalized_polygon_to_pixels(points, width, height)
        for lane_id, points in LANE_POLYGONS_NORMALIZED.items()
    }


def intersection_area_box_polygon(
    box: Tuple[int, int, int, int],
    polygon: np.ndarray,
    frame_shape: Tuple[int, int, int],
) -> int:
    """
    Calculate approximate pixel intersection area between a bounding box
    and a lane polygon using binary masks.

    For a small proof-of-concept this is simple and reliable.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0

    # Work only inside the bounding-box ROI rather than creating a full-frame mask.
    roi_w = x2 - x1
    roi_h = y2 - y1

    shifted_polygon = polygon.copy()
    shifted_polygon[:, 0] -= x1
    shifted_polygon[:, 1] -= y1

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_polygon], 255)

    return int(cv2.countNonZero(mask))


def lanes_for_box(
    box: Tuple[int, int, int, int],
    lane_polygons: Dict[int, np.ndarray],
    frame_shape: Tuple[int, int, int],
) -> List[int]:
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))

    overlaps = []

    for lane_id, polygon in lane_polygons.items():
        intersection = intersection_area_box_polygon(
            box,
            polygon,
            frame_shape,
        )
        ratio = intersection / box_area

        if ratio >= LANE_OVERLAP_THRESHOLD:
            overlaps.append((lane_id, ratio))

    if overlaps:
        return [lane_id for lane_id, _ in sorted(overlaps)]

    # Fallback:
    # If overlap is below threshold, assign by object center if center lies in a lane.
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    for lane_id, polygon in lane_polygons.items():
        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
            return [lane_id]

    return []


def lane_text(lanes: List[int]) -> str:
    if not lanes:
        return "Outside road"

    if len(lanes) == 1:
        return f"Lane {lanes[0]}"

    # Gives "Lane 1-2", "Lane 2-3", etc.
    return f"Lane {min(lanes)}-{max(lanes)}"


def center_of_box(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def object_diagonal(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, float(np.hypot(x2 - x1, y2 - y1)))


def motion_status(
    track_id: int,
    box: Tuple[int, int, int, int],
    histories: Dict[int, deque],
) -> str:
    """
    Determine RUNNING vs STOPPED based on movement across several frames.

    Using displacement relative to object size makes the threshold less dependent
    on camera resolution.
    """
    center = center_of_box(box)
    histories[track_id].append(center)

    history = histories[track_id]

    if len(history) < MIN_HISTORY_FOR_STATE:
        return "TRACKING"

    # Compare recent center to an older center.
    start = np.array(history[0], dtype=np.float32)
    end = np.array(history[-1], dtype=np.float32)

    displacement = float(np.linalg.norm(end - start))
    threshold = object_diagonal(box) * MOTION_THRESHOLD_RATIO

    return "RUNNING" if displacement >= threshold else "STOPPED"


def draw_lanes(frame: np.ndarray, lane_polygons: Dict[int, np.ndarray]) -> None:
    overlay = frame.copy()

    for lane_id, polygon in lane_polygons.items():
        cv2.polylines(
            overlay,
            [polygon],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
        )

        center = polygon.mean(axis=0).astype(int)
        cv2.putText(
            overlay,
            f"LANE {lane_id}",
            (center[0] - 45, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)


def draw_detection(frame: np.ndarray, state: DetectionState) -> None:
    x1, y1, x2, y2 = state.box

    if state.status == "STOPPED":
        color = (0, 0, 255)
    elif state.status == "BLOCKED":
        color = (0, 165, 255)
    elif state.status == "RUNNING":
        color = (0, 255, 0)
    else:
        color = (255, 200, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = (
        f"{state.status} - {lane_text(state.lanes)}"
        f" | {state.class_name} #{state.track_id}"
    )

    # Text background
    (text_w, text_h), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )

    text_y = max(text_h + 8, y1 - 7)

    cv2.rectangle(
        frame,
        (x1, text_y - text_h - 8),
        (x1 + text_w + 6, text_y + 3),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_lane_summary(
    frame: np.ndarray,
    states: List[DetectionState],
) -> None:
    """
    Show one summary status per lane.

    Priority:
        BLOCKED > STOPPED > RUNNING > CLEAR
    """
    priority = {
        "CLEAR": 0,
        "TRACKING": 1,
        "RUNNING": 2,
        "STOPPED": 3,
        "BLOCKED": 4,
    }

    lane_state = {
        1: "CLEAR",
        2: "CLEAR",
        3: "CLEAR",
    }

    for state in states:
        for lane in state.lanes:
            if lane not in lane_state:
                continue

            if priority[state.status] > priority[lane_state[lane]]:
                lane_state[lane] = state.status

    y = 30

    for lane_id in sorted(lane_state):
        status = lane_state[lane_id]

        if status == "BLOCKED":
            color = (0, 165, 255)
        elif status == "STOPPED":
            color = (0, 0, 255)
        elif status == "RUNNING":
            color = (0, 255, 0)
        elif status == "TRACKING":
            color = (255, 200, 0)
        else:
            color = (220, 220, 220)

        text = f"Lane {lane_id}: {status}"

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO 3-lane moving/stopped/blockage detector"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index such as 0, or path to video file",
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path/name, e.g. yolov8n.pt or best.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO detection confidence",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config, e.g. bytetrack.yaml",
    )

    parser.add_argument(
        "--show-all-detections",
        action="store_true",
        help="Show detections even when outside the road polygons",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # One deque per YOLO track ID
    histories: Dict[int, deque] = defaultdict(
        lambda: deque(maxlen=MOTION_HISTORY)
    )

    print("Starting detection.")
    print("Press Q or ESC to quit.")

    lane_polygons = None

    # persist=True is essential so track IDs remain consistent between frames.
    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.confidence,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img.copy()

        if lane_polygons is None:
            h, w = frame.shape[:2]
            lane_polygons = build_lane_polygons(w, h)

        states: List[DetectionState] = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                # This can happen briefly before the tracker has assigned IDs.
                track_ids = [-1] * len(boxes)

            for box_array, class_id, confidence, track_id in zip(
                boxes,
                classes,
                confidences,
                track_ids,
            ):
                box = tuple(int(v) for v in box_array)
                class_name = str(model.names[class_id]).lower()

                lanes = lanes_for_box(
                    box,
                    lane_polygons,
                    frame.shape,
                )

                if (
                    not lanes
                    and IGNORE_OBJECTS_OUTSIDE_ROAD
                    and not args.show_all_detections
                ):
                    continue

                is_vehicle = class_name in VEHICLE_CLASS_NAMES

                if is_vehicle:
                    if track_id >= 0:
                        status = motion_status(
                            track_id,
                            box,
                            histories,
                        )
                    else:
                        status = "TRACKING"
                else:
                    # Any YOLO-detected non-vehicle object located on the road
                    # is treated as a blockage.
                    status = "BLOCKED"

                state = DetectionState(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    box=box,
                    lanes=lanes,
                    status=status,
                )

                states.append(state)
                draw_detection(frame, state)

        draw_lanes(frame, lane_polygons)
        draw_lane_summary(frame, states)

        cv2.putText(
            frame,
            "Q / ESC = quit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("3-Lane YOLO Road Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
