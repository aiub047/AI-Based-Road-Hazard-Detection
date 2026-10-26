"""
Calibrate lane polygons against a frame from a video file.

Usage:
  python3 v2_calibrate_lanes.py path/to/video.mp4
  python3 v2_calibrate_lanes.py -Y "https://www.youtube.com/watch?v=..."

Steps:
  1. Press 'n' and 'p' to move through video frames.
  2. Press SPACE to freeze the current frame.
  3. Click the lane points in order for each lane.
  4. Press 's' to save to lanes.json, 'r' to restart clicking, 'u' to return to frame selection, 'q' to quit.
"""

import argparse
import cv2
import json
from pathlib import Path
from urllib.parse import urlparse
import shutil
import subprocess
import tempfile

NUM_LANES = 4
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
    cv2.putText(img, "s=save  r=reset  u=unfreeze  q=quit", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate lane polygons from a video file."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-Y",
        "--youtube",
        help="YouTube video URL to download and calibrate from",
    )
    args = parser.parse_args()

    if not args.source and not args.youtube:
        parser.error("Provide either a local video path or -Y/--youtube URL.")

    if args.source and args.youtube:
        parser.error("Use either a local video path or -Y/--youtube, not both.")

    return args


def normalize_youtube_url(url):
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    return f"https://{url}"


def download_youtube_video(url):
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is required for -Y/--youtube. Install it first, e.g. 'brew install yt-dlp'."
        )

    normalized_url = normalize_youtube_url(url)
    temp_dir = tempfile.TemporaryDirectory(prefix="lane-calibration-")
    output_template = str(Path(temp_dir.name) / "video.%(ext)s")

    format_attempts = [
        [
            "--format",
            "best[ext=mp4][vcodec!=none][acodec!=none]/best[vcodec!=none][acodec!=none]/best",
        ],
        [
            "--format",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
            "--merge-output-format",
            "mp4",
        ],
    ]

    errors = []

    for format_args in format_attempts:
        for path in Path(temp_dir.name).iterdir():
            if path.is_file():
                path.unlink()

        command = [
            yt_dlp,
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            *format_args,
            "--output",
            output_template,
            normalized_url,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            break

        error_text = result.stderr.strip() or result.stdout.strip() or "yt-dlp download failed."
        errors.append(error_text)
    else:
        temp_dir.cleanup()
        raise RuntimeError(errors[-1])

    downloaded_files = sorted(Path(temp_dir.name).iterdir())
    if not downloaded_files:
        temp_dir.cleanup()
        raise RuntimeError("yt-dlp did not produce a downloadable video file.")

    return str(downloaded_files[0]), temp_dir


def resolve_video_source(args):
    if args.youtube:
        print("Downloading YouTube video...")
        video_path, temp_dir = download_youtube_video(args.youtube)
        return video_path, temp_dir

    return args.source, None


def read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def draw_video_frame(base, frame_index, total_frames):
    img = base.copy()
    frame_label = f"Frame {frame_index + 1}"
    if total_frames > 0:
        frame_label += f"/{total_frames}"
    cv2.putText(img, frame_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "n=next  p=prev  SPACE=freeze  q=quit", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main():
    global frozen_frame, points

    args = parse_args()
    temp_dir = None

    try:
        video_source, temp_dir = resolve_video_source(args)
    except RuntimeError as exc:
        print(exc)
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video file: {video_source}")
        if temp_dir is not None:
            temp_dir.cleanup()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    current_frame = read_frame(cap, frame_index)
    if current_frame is None:
        print(f"Could not read frames from video file: {video_source}")
        cap.release()
        if temp_dir is not None:
            temp_dir.cleanup()
        return

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", on_mouse)

    print("Use 'n'/'p' to choose a frame, then press SPACE to freeze it.")

    while True:
        if frozen_frame is None:
            cv2.imshow("Calibrate", draw_video_frame(current_frame, frame_index, total_frames))
        else:
            cv2.imshow("Calibrate", draw_overlay(frozen_frame))

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' ') and frozen_frame is None:
            frozen_frame = current_frame.copy()
        elif key == ord('r'):
            points = []
        elif key == ord('u'):
            points = []
            frozen_frame = None
        elif key == ord('q'):
            break
        elif key == ord('n') and frozen_frame is None:
            next_index = frame_index + 1
            if total_frames > 0:
                next_index = min(next_index, total_frames - 1)
            frame = read_frame(cap, next_index)
            if frame is not None:
                frame_index = next_index
                current_frame = frame
        elif key == ord('p') and frozen_frame is None:
            prev_index = max(frame_index - 1, 0)
            frame = read_frame(cap, prev_index)
            if frame is not None:
                frame_index = prev_index
                current_frame = frame
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
    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
