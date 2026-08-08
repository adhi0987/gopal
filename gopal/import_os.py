import argparse
import os
import sys

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

# Simple green-marker alignment script.
# This version keeps the core workflow short and easy to edit.

LOWER_GREEN = np.array([30, 70, 30])
UPPER_GREEN = np.array([90, 255, 255])
AREA_MIN = 500
AREA_MAX = 200000
EXPECTED_MARKERS = 8
KNOWN_STICKER_DIAMETER_MM = 15.0
ALIGNMENT_TOLERANCE_MM = 0.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

CALIBRATED_CHUCK = {
    "center": None,
    "radius_px": None,
    "pixel_to_mm": None,
    "markers": None,
}


def detect_markers(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        _, radius = cv2.minEnclosingCircle(contour)
        markers.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "contour": contour,
            "diameter_px": 2.0 * radius,
        })

    return markers, mask


def fit_circle(points):
    points = np.array(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("Need at least 3 points")

    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    radius = np.sqrt(max(c + cx**2 + cy**2, 0.0))
    return cx, cy, radius


def compute_pixel_to_mm_ratio(markers):
    if not markers:
        return None
    diameters = np.array([m["diameter_px"] for m in markers], dtype=np.float64)
    avg_diameter = np.mean(diameters)
    if avg_diameter <= 0:
        return None
    return KNOWN_STICKER_DIAMETER_MM / avg_diameter


def calibrate_chuck(markers):
    if len(markers) < 3:
        return None

    try:
        cx, cy, radius_px = fit_circle([(m["cx"], m["cy"]) for m in markers])
    except Exception:
        return None

    pixel_to_mm = compute_pixel_to_mm_ratio(markers)
    if pixel_to_mm is None:
        return None

    return {
        "center": (cx, cy),
        "radius_px": radius_px,
        "pixel_to_mm": pixel_to_mm,
        "markers": markers,
    }


def compute_offset(chuck_center, component_center, pixel_to_mm):
    dx_px = component_center[0] - chuck_center[0]
    dy_px = component_center[1] - chuck_center[1]
    dx_mm = dx_px * pixel_to_mm
    dy_mm = dy_px * pixel_to_mm
    distance_mm = np.hypot(dx_mm, dy_mm)

    if abs(dx_mm) <= ALIGNMENT_TOLERANCE_MM:
        x_dir = "aligned"
    elif dx_mm > 0:
        x_dir = "move left"
    else:
        x_dir = "move right"

    if abs(dy_mm) <= ALIGNMENT_TOLERANCE_MM:
        y_dir = "aligned"
    elif dy_mm > 0:
        y_dir = "move up"
    else:
        y_dir = "move down"

    aligned = distance_mm <= ALIGNMENT_TOLERANCE_MM
    return {
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "distance_mm": distance_mm,
        "x_direction": x_dir,
        "y_direction": y_dir,
        "aligned": aligned,
    }


def annotate_frame(img, markers, mask):
    img_display = img.copy()

    if EXPECTED_MARKERS is not None and len(markers) != EXPECTED_MARKERS:
        cv2.putText(
            img_display,
            f"Markers found: {len(markers)} / expected {EXPECTED_MARKERS}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if CALIBRATED_CHUCK["center"] is not None:
        cx, cy = CALIBRATED_CHUCK["center"]
        radius_px = CALIBRATED_CHUCK["radius_px"]
        cv2.circle(img_display, (int(cx), int(cy)), int(radius_px), (0, 0, 255), 2)
        cv2.circle(img_display, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(
            img_display,
            f"Chuck center: ({cx:.1f},{cy:.1f})",
            (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    for marker in markers:
        cv2.circle(img_display, (int(marker["cx"]), int(marker["cy"])), 4, (0, 255, 0), -1)
        cv2.drawContours(img_display, [marker["contour"]], -1, (0, 255, 255), 1)

    return img_display


def show_image(img, title=""):
    if img is None:
        print(f"show_image: no image to display for {title}")
        return

    window_name = title or "Image"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
    except cv2.error as exc:
        print(f"Could not display window {window_name}: {exc}")
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def browse_for_media_file():
    if filedialog is None:
        print("File dialog is unavailable in this environment.")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select an image or video file",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
    except Exception as exc:
        print(f"Could not open file dialog: {exc}")
        path = None
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return path or None


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    markers, mask = detect_markers(img)
    if CALIBRATED_CHUCK["center"] is None:
        calibration = calibrate_chuck(markers)
        if calibration is not None:
            CALIBRATED_CHUCK.update(calibration)

    print(f"Markers found: {len(markers)}")
    if CALIBRATED_CHUCK["center"] is not None:
        print(f"Calibrated chuck center: {CALIBRATED_CHUCK['center']}")
        print(f"Pixel to mm ratio: {CALIBRATED_CHUCK['pixel_to_mm']:.4f} mm/px")

    annotated = annotate_frame(img, markers, mask)
    show_image(annotated, "Annotated Image")


def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not read video: {video_path}")
        return

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_annotated.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Could not open video writer.")
        cap.release()
        return

    print("Processing video...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        markers, mask = detect_markers(frame)
        if CALIBRATED_CHUCK["center"] is None:
            calibration = calibrate_chuck(markers)
            if calibration is not None:
                CALIBRATED_CHUCK.update(calibration)
        writer.write(annotate_frame(frame, markers, mask))

    cap.release()
    writer.release()
    print(f"Saved annotated video to: {output_path}")


def process_camera(camera_index=0, output_path=None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}")
        return

    window_name = "Live Alignment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    print("Press 'c' to calibrate and 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        markers, mask = detect_markers(frame)
        if CALIBRATED_CHUCK["center"] is None:
            calibration = calibrate_chuck(markers)
            if calibration is not None:
                CALIBRATED_CHUCK.update(calibration)

        cv2.imshow(window_name, annotate_frame(frame, markers, mask))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            calibration = calibrate_chuck(markers)
            if calibration is not None:
                CALIBRATED_CHUCK.update(calibration)
                print("Chuck center recalibrated.")

    cap.release()
    cv2.destroyAllWindows()


def process_media(input_path=None, output_path=None, browse_file=False):
    if input_path is None and not browse_file:
        choice = input("Choose input type: [f]ile path or [1] for live feed: ").strip().lower()
        if choice in {"", "f", "file", "filepath", "path"}:
            input_path = input("Enter image/video file path: ").strip().strip('"')
        elif choice in {"1", "live", "livefeed", "camera"}:
            input_path = "1"
        else:
            input_path = input("Enter image/video file path: ").strip().strip('"')

    if browse_file:
        input_path = browse_for_media_file()
        if input_path is None:
            print("No file selected. Exiting.")
            return

    if input_path is None:
        print("No input provided. Exiting.")
        return

    if isinstance(input_path, str):
        input_path = input_path.strip().strip('"')

    if str(input_path).isdigit():
        process_camera(int(input_path) if input_path != "1" else 0, output_path)
        return

    ext = os.path.splitext(str(input_path))[1].lower()
    if ext in VIDEO_EXTENSIONS:
        process_video(str(input_path), output_path)
    else:
        process_image(str(input_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple chuck alignment tracker")
    parser.add_argument("-i", "--input", help="Image file, video file, or camera index")
    parser.add_argument("-o", "--output", help="Output video path for video/live feed")
    parser.add_argument("--browse-file", action="store_true", help="Open file picker")
    parser.add_argument("--sticker-diameter-mm", type=float, default=KNOWN_STICKER_DIAMETER_MM)
    args = parser.parse_args()

    KNOWN_STICKER_DIAMETER_MM = args.sticker_diameter_mm
    process_media(args.input, args.output, browse_file=args.browse_file)
