"""
lane_monitor_v6.py

Fixes in V6:
- Moving vehicles are classified much sooner and more reliably.
- Uses RECENT motion over a short window instead of waiting on a long history.
- Once a track is recognized as a vehicle, that track stays vehicle-like even
  if YOLO briefly changes the class label.
- Stopped / too-slow vehicle -> Obstacle.
- Forward moving vehicle -> Vehicle running on Lane X / Lane X-Y.
- Backward vehicle -> Vehicle moving WRONG WAY on Lane X / Lane X-Y.
- Static printed road lines remain permanently ignored for UNKNOWN obstacles.
"""

import json
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CAM_INDEX = 0
MODEL_NAME = "yolo11m.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
}

CONF_THRESHOLD = 0.18

# YOLO road/lane geometry
MIN_YOLO_ROAD_OVERLAP = 0.15
LANE_OVERLAP_THRESHOLD = 0.06

# ============================================================
# VEHICLE MOTION SETTINGS
# ============================================================

# Keep a reasonably long buffer...
HISTORY_LEN = 16

# ...but classify using only recent frames.
RECENT_MOTION_FRAMES = 5

# Need only a few samples before showing a state.
MIN_MOTION_SAMPLES = 3

# If center displacement over recent frames is below this fraction
# of object diagonal, treat as stopped / too slow.
STOPPED_DISPLACEMENT_RATIO = 0.035

# Direction projected onto road forward vector.
# Smaller than older version so a moving toy vehicle is not stuck in TRACKING.
DIRECTION_MIN_RATIO = 0.015

# Pixel fallback for tiny detections.
MIN_RUNNING_PIXELS = 4.0

# Wrong-way needs a little stronger evidence than normal forward motion.
WRONG_WAY_MIN_PIXELS = 5.0

# Reduce flicker but do not delay state too much.
STATE_CONFIRM_FRAMES = 2

# If a track has ever been recognized as a vehicle, remember that fact.
VEHICLE_MEMORY_SECONDS = 3.0

# ============================================================
# BASELINE / UNKNOWN OBJECT SETTINGS
# ============================================================

BASELINE_FRAMES = 20

GRAY_DIFF_THRESHOLD = 14
LAB_DIFF_THRESHOLD = 16.0

MIN_UNKNOWN_AREA = 80
MIN_LANE_CHANGED_PIXELS = 35
UNKNOWN_LANE_PIXEL_FRACTION = 0.08
MAX_UNKNOWN_ROAD_FRACTION = 0.20

PERSISTENCE_FRAMES = 4
PERSISTENCE_REQUIRED = 2

OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)

BLACK_MARKING_THRESHOLD = 190
STATIC_BLACK_IGNORE_DILATION = 13
CALIBRATION_EDGE_IGNORE_WIDTH = 12
ROAD_INNER_MARGIN_PX = 5


# ============================================================
# LANE / ROAD MASKS
# ============================================================

def load_lanes(path="lanes.json"):
    with open(path, "r") as f:
        data = json.load(f)

    lanes = [np.array(lane, dtype=np.int32) for lane in data["lanes"]]

    if len(lanes) != 3:
        raise ValueError(f"Expected 3 lanes in {path}; found {len(lanes)}")

    return lanes


def build_lane_masks(lanes, frame_shape):
    h, w = frame_shape[:2]
    masks = []

    for lane in lanes:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [lane], 255)
        masks.append(mask)

    return masks


def build_road_mask(lane_masks):
    road = np.zeros_like(lane_masks[0])

    for mask in lane_masks:
        road = cv2.bitwise_or(road, mask)

    if ROAD_INNER_MARGIN_PX > 0:
        k = ROAD_INNER_MARGIN_PX * 2 + 1
        road = cv2.erode(
            road,
            np.ones((k, k), np.uint8),
            iterations=1,
        )

    return road


def build_calibration_edge_mask(lanes, frame_shape, road_mask):
    h, w = frame_shape[:2]

    edge_mask = np.zeros((h, w), dtype=np.uint8)

    for lane in lanes:
        cv2.polylines(
            edge_mask,
            [lane],
            True,
            255,
            thickness=CALIBRATION_EDGE_IGNORE_WIDTH,
            lineType=cv2.LINE_AA,
        )

    road_expanded = cv2.dilate(
        road_mask,
        np.ones((11, 11), np.uint8),
        iterations=1,
    )

    return cv2.bitwise_and(edge_mask, road_expanded)


def format_lane_range(indices):
    nums = sorted(set(i + 1 for i in indices))

    if not nums:
        return ""

    if len(nums) == 1:
        return str(nums[0])

    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"{nums[0]}-{nums[-1]}"

    return ",".join(str(n) for n in nums)


def box_road_overlap_ratio(box, road_mask):
    h, w = road_mask.shape

    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    area = max(1.0, float((x2 - x1) * (y2 - y1)))

    return np.count_nonzero(
        road_mask[y1:y2, x1:x2]
    ) / area


def lanes_for_box(box, lane_masks):
    h, w = lane_masks[0].shape

    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return []

    box_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    occupied = []

    for idx, lane_mask in enumerate(lane_masks):
        overlap = np.count_nonzero(
            lane_mask[y1:y2, x1:x2]
        )

        if overlap / box_area >= LANE_OVERLAP_THRESHOLD:
            occupied.append(idx)

    return occupied


def lanes_for_component(component, lane_masks):
    total = np.count_nonzero(component)

    if total == 0:
        return []

    occupied = []

    for idx, lane_mask in enumerate(lane_masks):
        inside = cv2.bitwise_and(component, lane_mask)
        count = np.count_nonzero(inside)

        if count < MIN_LANE_CHANGED_PIXELS:
            continue

        if count / float(total) >= UNKNOWN_LANE_PIXEL_FRACTION:
            occupied.append(idx)

    return occupied


# ============================================================
# VEHICLE MOTION
# ============================================================

def compute_forward_vector(lanes):
    vectors = []

    for lane in lanes:
        far_mid = (
            lane[0].astype(np.float32)
            + lane[1].astype(np.float32)
        ) / 2.0

        near_mid = (
            lane[3].astype(np.float32)
            + lane[2].astype(np.float32)
        ) / 2.0

        v = far_mid - near_mid
        norm = np.linalg.norm(v)

        if norm > 0:
            vectors.append(v / norm)

    if not vectors:
        return np.array([0.0, -1.0], dtype=np.float32)

    v = np.mean(vectors, axis=0)
    norm = np.linalg.norm(v)

    return v / norm if norm > 0 else np.array([0.0, -1.0], dtype=np.float32)


def box_center(box):
    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float32,
    )


def box_diagonal(box):
    x1, y1, x2, y2 = box

    return max(
        1.0,
        float(np.hypot(x2 - x1, y2 - y1)),
    )


def classify_vehicle_motion(history, box, forward_vec):
    """
    Use SHORT recent motion.

    This works better for toy-car demonstrations because:
    - YOLO track IDs may not survive for a long time.
    - Waiting 12-15 frames can delay RUNNING classification.
    - Recent net motion is enough to distinguish stopped vs moving.
    """

    if len(history) < MIN_MOTION_SAMPLES:
        return "TRACKING", 0.0, 0.0, 0.0

    sample_count = min(
        len(history),
        RECENT_MOTION_FRAMES,
    )

    recent = list(history)[-sample_count:]

    first_center = recent[0][0]
    last_center = recent[-1][0]

    displacement = last_center - first_center

    total_pixels = float(
        np.linalg.norm(displacement)
    )

    diag = box_diagonal(box)

    move_ratio = total_pixels / diag

    forward_pixels = float(
        np.dot(displacement, forward_vec)
    )

    direction_ratio = forward_pixels / diag

    # --------------------------------------------------------
    # STOPPED / TOO SLOW
    # --------------------------------------------------------

    if (
        move_ratio < STOPPED_DISPLACEMENT_RATIO
        and total_pixels < MIN_RUNNING_PIXELS
    ):
        return (
            "OBSTACLE",
            move_ratio,
            direction_ratio,
            total_pixels,
        )

    # --------------------------------------------------------
    # WRONG WAY
    # --------------------------------------------------------

    if (
        direction_ratio < -DIRECTION_MIN_RATIO
        and forward_pixels < -WRONG_WAY_MIN_PIXELS
    ):
        return (
            "WRONG_WAY",
            move_ratio,
            direction_ratio,
            total_pixels,
        )

    # --------------------------------------------------------
    # MOVING
    # --------------------------------------------------------

    if (
        move_ratio >= STOPPED_DISPLACEMENT_RATIO
        or total_pixels >= MIN_RUNNING_PIXELS
    ):
        return (
            "RUNNING",
            move_ratio,
            direction_ratio,
            total_pixels,
        )

    return (
        "OBSTACLE",
        move_ratio,
        direction_ratio,
        total_pixels,
    )


def stabilize_state(track_id, raw_state, memory):
    """
    Fast two-frame confirmation.
    """

    state = memory.get(
        track_id,
        {
            "display": "TRACKING",
            "candidate": None,
            "count": 0,
        },
    )

    if raw_state == "TRACKING":
        return state["display"]

    if raw_state == state["display"]:
        state["candidate"] = None
        state["count"] = 0

    elif raw_state == state["candidate"]:
        state["count"] += 1

        if state["count"] >= STATE_CONFIRM_FRAMES:
            state["display"] = raw_state
            state["candidate"] = None
            state["count"] = 0

    else:
        state["candidate"] = raw_state
        state["count"] = 1

    memory[track_id] = state

    return state["display"]


def is_vehicle_track(
    track_id,
    class_name,
    vehicle_memory,
):
    """
    Once YOLO calls this track a vehicle, remember it for a few seconds.
    This prevents class-label flicker on toy cars.
    """

    now = time.monotonic()

    if class_name in VEHICLE_CLASSES:
        if track_id >= 0:
            vehicle_memory[track_id] = now
        return True

    if track_id >= 0:
        last_seen_as_vehicle = vehicle_memory.get(track_id)

        if (
            last_seen_as_vehicle is not None
            and now - last_seen_as_vehicle <= VEHICLE_MEMORY_SECONDS
        ):
            return True

    return False


# ============================================================
# BASELINE / STATIC ROAD LINE MASK
# ============================================================

def preprocess_gray(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.GaussianBlur(
        gray,
        (9, 9),
        0,
    )


def preprocess_lab(frame):
    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB,
    )

    return cv2.GaussianBlur(
        lab,
        (7, 7),
        0,
    )


def capture_baseline(cap):
    frames = []

    print(
        f"Capturing {BASELINE_FRAMES} EMPTY-road frames..."
    )

    for _ in range(BASELINE_FRAMES):
        ok, frame = cap.read()

        if ok:
            frames.append(frame.copy())

        cv2.waitKey(20)

    if not frames:
        return None

    median = np.median(
        np.stack(frames, axis=0),
        axis=0,
    ).astype(np.uint8)

    print("Baseline captured.")

    return {
        "bgr": median,
        "gray": preprocess_gray(median),
        "lab": preprocess_lab(median),
    }


def build_black_marking_mask(
    baseline_bgr,
    road_mask,
):
    gray = cv2.cvtColor(
        baseline_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    dark = np.zeros_like(
        gray,
        dtype=np.uint8,
    )

    dark[
        gray < BLACK_MARKING_THRESHOLD
    ] = 255

    dark = cv2.bitwise_and(
        dark,
        road_mask,
    )

    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    size = max(
        3,
        STATIC_BLACK_IGNORE_DILATION,
    )

    if size % 2 == 0:
        size += 1

    dark = cv2.dilate(
        dark,
        np.ones((size, size), np.uint8),
        iterations=1,
    )

    return dark


def build_static_ignore_mask(
    baseline_bgr,
    lanes,
    road_mask,
):
    black_mask = build_black_marking_mask(
        baseline_bgr,
        road_mask,
    )

    edge_mask = build_calibration_edge_mask(
        lanes,
        baseline_bgr.shape,
        road_mask,
    )

    ignore = cv2.bitwise_or(
        black_mask,
        edge_mask,
    )

    return ignore, black_mask, edge_mask


# ============================================================
# UNKNOWN OBJECT DETECTION
# ============================================================

def build_raw_change(
    frame,
    baseline,
    road_mask,
):
    current_gray = preprocess_gray(frame)
    current_lab = preprocess_lab(frame)

    gray_diff = cv2.absdiff(
        current_gray,
        baseline["gray"],
    )

    delta = (
        current_lab.astype(np.float32)
        - baseline["lab"].astype(np.float32)
    )

    lab_distance = np.sqrt(
        np.sum(delta * delta, axis=2)
    )

    changed = (
        (gray_diff >= GRAY_DIFF_THRESHOLD)
        | (lab_distance >= LAB_DIFF_THRESHOLD)
    )

    mask = changed.astype(np.uint8) * 255

    return cv2.bitwise_and(
        mask,
        road_mask,
    )


def suppress_static_lines(
    mask,
    static_ignore_mask,
    road_mask,
):
    result = mask.copy()

    result[
        static_ignore_mask > 0
    ] = 0

    return cv2.bitwise_and(
        result,
        road_mask,
    )


def clean_change_mask(
    mask,
    road_mask,
):
    result = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        OPEN_KERNEL,
    )

    result = cv2.bitwise_and(
        result,
        road_mask,
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        CLOSE_KERNEL,
    )

    return cv2.bitwise_and(
        result,
        road_mask,
    )


def update_persistence(
    mask,
    history,
    road_mask,
):
    history.append(
        (mask > 0).astype(np.uint8)
    )

    if len(history) < PERSISTENCE_REQUIRED:
        return np.zeros_like(mask)

    count = np.sum(
        np.stack(history, axis=0),
        axis=0,
    )

    persistent = (
        count >= PERSISTENCE_REQUIRED
    ).astype(np.uint8) * 255

    return cv2.bitwise_and(
        persistent,
        road_mask,
    )


def remove_yolo_regions(
    mask,
    boxes,
    road_mask,
    pad=7,
):
    result = mask.copy()
    h, w = result.shape

    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        result[y1:y2, x1:x2] = 0

    return cv2.bitwise_and(
        result,
        road_mask,
    )


def detect_unknown_obstacles(
    frame,
    baseline,
    road_mask,
    lane_masks,
    static_ignore_mask,
    yolo_boxes,
    persistence_history,
):
    raw = build_raw_change(
        frame,
        baseline,
        road_mask,
    )

    no_lines = suppress_static_lines(
        raw,
        static_ignore_mask,
        road_mask,
    )

    cleaned = clean_change_mask(
        no_lines,
        road_mask,
    )

    persistent = update_persistence(
        cleaned,
        persistence_history,
        road_mask,
    )

    candidate = remove_yolo_regions(
        persistent,
        yolo_boxes,
        road_mask,
    )

    contours, _ = cv2.findContours(
        candidate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    road_area = max(
        1,
        np.count_nonzero(road_mask),
    )

    obstacles = []
    accepted = np.zeros_like(candidate)

    for contour in contours:
        component = np.zeros_like(
            candidate
        )

        cv2.drawContours(
            component,
            [contour],
            -1,
            255,
            -1,
        )

        component = cv2.bitwise_and(
            component,
            road_mask,
        )

        component[
            static_ignore_mask > 0
        ] = 0

        pixels = np.count_nonzero(
            component
        )

        if pixels < MIN_UNKNOWN_AREA:
            continue

        if (
            pixels / float(road_area)
            > MAX_UNKNOWN_ROAD_FRACTION
        ):
            continue

        occupied = lanes_for_component(
            component,
            lane_masks,
        )

        if not occupied:
            continue

        contours2, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours2:
            continue

        pts = np.vstack(contours2)
        x, y, w, h = cv2.boundingRect(pts)

        obstacles.append(
            (
                (x, y, x + w, y + h),
                occupied,
            )
        )

        accepted = cv2.bitwise_or(
            accepted,
            component,
        )

    return obstacles, raw, no_lines, persistent, accepted


# ============================================================
# DRAWING
# ============================================================

def state_color(state):
    if state == "RUNNING":
        return (0, 255, 0)

    if state == "WRONG_WAY":
        return (255, 0, 255)

    if state == "OBSTACLE":
        return (0, 0, 255)

    return (0, 255, 255)


def vehicle_label(
    state,
    occupied,
):
    lane = format_lane_range(
        occupied
    )

    if state == "RUNNING":
        return f"Vehicle running on Lane {lane}"

    if state == "WRONG_WAY":
        return f"Vehicle moving WRONG WAY on Lane {lane}"

    if state == "OBSTACLE":
        return f"Obstacle on Lane {lane}"

    return f"Tracking vehicle on Lane {lane}"


def draw_box(
    frame,
    box,
    label,
    color,
):
    x1, y1, x2, y2 = [
        int(v)
        for v in box
    ]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(20, y1 - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_lanes(
    frame,
    lanes,
):
    for idx, lane in enumerate(lanes):
        cv2.polylines(
            frame,
            [lane],
            True,
            (0, 200, 255),
            2,
        )

        top_mid = (
            (
                lane[0].astype(float)
                + lane[1].astype(float)
            )
            / 2
        ).astype(int)

        cv2.putText(
            frame,
            f"Lane {idx + 1}",
            (
                int(top_mid[0]) - 30,
                int(top_mid[1]) + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    lanes = load_lanes()

    forward_vec = compute_forward_vector(
        lanes
    )

    print(
        "Forward vector:",
        np.round(
            forward_vec,
            3,
        ),
    )

    print("Device:", DEVICE)

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(
        CAM_INDEX
    )

    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    lane_masks = None
    road_mask = None

    baseline = None

    static_ignore_mask = None
    black_marking_mask = None
    calibration_edge_mask = None

    track_history = defaultdict(
        lambda: deque(maxlen=HISTORY_LEN)
    )

    state_memory = {}

    # Remember tracks previously recognized as vehicles.
    vehicle_memory = {}

    persistence_history = deque(
        maxlen=PERSISTENCE_FRAMES
    )

    show_debug = False

    print("")
    print(
        "V6 motion mode: fast recent-motion vehicle classification"
    )
    print(
        "b = baseline | d = debug | q = quit"
    )
    print("")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if lane_masks is None:
            original_lane_masks = build_lane_masks(
                lanes,
                frame.shape,
            )

            road_mask = build_road_mask(
                original_lane_masks
            )

            lane_masks = [
                cv2.bitwise_and(
                    m,
                    road_mask,
                )
                for m in original_lane_masks
            ]

        display = frame.copy()

        draw_lanes(
            display,
            lanes,
        )

        # ====================================================
        # YOLO
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            device=DEVICE,
            conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml",
        )[0]

        messages = []
        yolo_boxes = []

        if (
            results.boxes is not None
            and len(results.boxes) > 0
        ):
            boxes = (
                results.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            classes = (
                results.boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                results.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            if results.boxes.id is not None:
                ids = (
                    results.boxes.id
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )
            else:
                ids = np.full(
                    len(boxes),
                    -1,
                    dtype=int,
                )

            for (
                box,
                track_id,
                cls_id,
                confidence,
            ) in zip(
                boxes,
                ids,
                classes,
                confidences,
            ):
                if (
                    box_road_overlap_ratio(
                        box,
                        road_mask,
                    )
                    < MIN_YOLO_ROAD_OVERLAP
                ):
                    continue

                occupied = lanes_for_box(
                    box,
                    lane_masks,
                )

                if not occupied:
                    continue

                x1, y1, x2, y2 = [
                    int(v)
                    for v in box
                ]

                yolo_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

                class_name = str(
                    model.names[cls_id]
                ).lower()

                treat_as_vehicle = is_vehicle_track(
                    track_id,
                    class_name,
                    vehicle_memory,
                )

                # ------------------------------------------------
                # VEHICLE
                # ------------------------------------------------

                if treat_as_vehicle:
                    center = box_center(
                        box
                    )

                    if track_id >= 0:
                        history = track_history[
                            track_id
                        ]

                        history.append(
                            (
                                center,
                                time.monotonic(),
                            )
                        )

                        (
                            raw_state,
                            move_ratio,
                            dir_ratio,
                            move_pixels,
                        ) = classify_vehicle_motion(
                            history,
                            box,
                            forward_vec,
                        )

                        state = stabilize_state(
                            track_id,
                            raw_state,
                            state_memory,
                        )

                    else:
                        state = "TRACKING"
                        move_ratio = 0.0
                        dir_ratio = 0.0
                        move_pixels = 0.0

                    label = vehicle_label(
                        state,
                        occupied,
                    )

                    draw_box(
                        display,
                        box,
                        label,
                        state_color(state),
                    )

                    messages.append(
                        label
                    )

                    # Motion debug under box.
                    cv2.putText(
                        display,
                        (
                            f"{class_name} {confidence:.2f} "
                            f"px={move_pixels:.1f} "
                            f"move={move_ratio:.3f} "
                            f"dir={dir_ratio:.3f}"
                        ),
                        (
                            x1,
                            min(
                                frame.shape[0] - 5,
                                y2 + 18,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (220, 220, 220),
                        1,
                    )

                # ------------------------------------------------
                # YOLO NON-VEHICLE
                # ------------------------------------------------

                else:
                    lane = format_lane_range(
                        occupied
                    )

                    label = (
                        f"Obstacle on Lane {lane}"
                    )

                    draw_box(
                        display,
                        box,
                        f"{label} ({class_name})",
                        (0, 0, 255),
                    )

                    messages.append(
                        label
                    )

        # ====================================================
        # UNKNOWN OBJECTS
        # ====================================================

        raw_debug = None
        no_lines_debug = None
        persistent_debug = None
        accepted_debug = None

        if (
            baseline is not None
            and static_ignore_mask is not None
        ):
            (
                unknowns,
                raw_debug,
                no_lines_debug,
                persistent_debug,
                accepted_debug,
            ) = detect_unknown_obstacles(
                frame,
                baseline,
                road_mask,
                lane_masks,
                static_ignore_mask,
                yolo_boxes,
                persistence_history,
            )

            for box, occupied in unknowns:
                lane = format_lane_range(
                    occupied
                )

                label = (
                    f"Obstacle on Lane {lane}"
                )

                draw_box(
                    display,
                    box,
                    f"{label} (unknown)",
                    (0, 0, 255),
                )

                messages.append(
                    label
                )

        # ====================================================
        # STATUS
        # ====================================================

        if baseline is None:
            status = (
                "Baseline NOT SET - clear road and press B"
            )
            status_color = (
                0,
                0,
                255,
            )
        else:
            status = (
                "Baseline SET"
            )
            status_color = (
                0,
                255,
                0,
            )

        cv2.putText(
            display,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
        )

        y = 55

        for msg in dict.fromkeys(
            messages
        ):
            if "WRONG WAY" in msg:
                color = (
                    255,
                    0,
                    255,
                )

            elif "Obstacle" in msg:
                color = (
                    0,
                    0,
                    255,
                )

            else:
                color = (
                    0,
                    255,
                    0,
                )

            cv2.putText(
                display,
                msg,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

            y += 25

        cv2.imshow(
            "Lane Monitor",
            display,
        )

        if show_debug:
            cv2.imshow(
                "DEBUG - ROAD MASK",
                road_mask,
            )

            if static_ignore_mask is not None:
                cv2.imshow(
                    "DEBUG - STATIC IGNORE",
                    static_ignore_mask,
                )

            if raw_debug is not None:
                cv2.imshow(
                    "DEBUG - RAW CHANGES",
                    raw_debug,
                )

            if no_lines_debug is not None:
                cv2.imshow(
                    "DEBUG - AFTER LINE REMOVAL",
                    no_lines_debug,
                )

            if accepted_debug is not None:
                cv2.imshow(
                    "DEBUG - ACCEPTED OBSTACLES",
                    accepted_debug,
                )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("b"):
            baseline = capture_baseline(
                cap
            )

            if baseline is not None:
                persistence_history.clear()

                (
                    static_ignore_mask,
                    black_marking_mask,
                    calibration_edge_mask,
                ) = build_static_ignore_mask(
                    baseline["bgr"],
                    lanes,
                    road_mask,
                )

        elif key == ord("d"):
            show_debug = not show_debug

            if not show_debug:
                for name in [
                    "DEBUG - ROAD MASK",
                    "DEBUG - STATIC IGNORE",
                    "DEBUG - RAW CHANGES",
                    "DEBUG - AFTER LINE REMOVAL",
                    "DEBUG - ACCEPTED OBSTACLES",
                ]:
                    try:
                        cv2.destroyWindow(name)
                    except cv2.error:
                        pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
