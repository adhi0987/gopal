import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Install once in your environment if needed:
# pip install opencv-python numpy matplotlib

print("Libraries imported successfully.")

# Green HSV range - tune with a color picker on your actual tape color.
LOWER_GREEN = np.array([30, 70, 30])
UPPER_GREEN = np.array([90, 255, 255])

# Expected blob area in pixels.
AREA_MIN = 500
AREA_MAX = 200000

# Expect exactly 8 markers; set to None to accept any number that passes filters.
EXPECTED_MARKERS = 8

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

print("Parameters initialized.")

# Optional: supply a real-world chuck diameter (in mm) via environment variable
# Example: export REAL_WORLD_CHUCK_DIAMETER_MM=250.0
REAL_WORLD_CHUCK_DIAMETER_MM = None
_env_val = os.getenv("REAL_WORLD_CHUCK_DIAMETER_MM")
if _env_val:
    try:
        REAL_WORLD_CHUCK_DIAMETER_MM = float(_env_val)
    except Exception:
        REAL_WORLD_CHUCK_DIAMETER_MM = None

# Optional max frames guard (can be set via env MAX_FRAMES or CLI --max-frames=N)
MAX_FRAMES = None
_mf_env = os.getenv("MAX_FRAMES")
if _mf_env:
    try:
        MAX_FRAMES = int(_mf_env)
    except Exception:
        MAX_FRAMES = None

# parse CLI-style --max-frames=NN if provided
for arg in sys.argv[1:]:
    if arg.startswith("--max-frames="):
        try:
            MAX_FRAMES = int(arg.split("=", 1)[1])
        except Exception:
            pass

def detect_markers(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Clean up noise / fill small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < AREA_MIN or area > AREA_MAX:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroids.append((cx, cy, area, c))

    return centroids, mask


def fit_circle_least_squares(points):
    """
    Least-squares circle fit (Kasa method) through a set of (x, y) points.
    Returns (cx, cy, radius).
    """
    pts = np.array(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(c + cx**2 + cy**2)
    return cx, cy, r


def detect_component_circle(img_bgr, reference_center=None):
    """Detect a prominent circular component (inner ring) using Hough Circles.
    If reference_center is provided, prefer circles near that center.
    Returns (cx, cy, r) or None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough parameters tuned conservatively; may be adjusted per-image
    rows = gray.shape[0]
    minrad = 8
    maxrad = int(min(gray.shape[:2]) * 0.45)
    try:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=rows / 8,
            param1=100,
            param2=30,
            minradadius=minrad,
            maxradadius=maxrad,
        )
    except Exception:
        circles = None

    if circles is None:
        return None

    circles = np.uint16(np.around(circles[0]))

    if reference_center is not None:
        rcx, rcy = reference_center
        # choose circle closest to reference center
        dists = [np.hypot(c[0] - rcx, c[1] - rcy) for c in circles]
        idx = int(np.argmin(dists))
        c = circles[idx]
        return float(c[0]), float(c[1]), float(c[2])

    # fallback: choose circle with largest radius (likely the visible ring)
    idx = int(np.argmax(circles[:, 2]))
    c = circles[idx]
    return float(c[0]), float(c[1]), float(c[2])


def show_image(img, title=""):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    # Non-interactive environments: do not call plt.show() by default.
    return


def annotate_frame(img, detections, mask):
    img_display = img.copy()

    if EXPECTED_MARKERS is not None and len(detections) != EXPECTED_MARKERS:
        print(f"WARNING: Expected {EXPECTED_MARKERS} markers, but found {len(detections)}.")

    marker_coords = [(d[0], d[1]) for d in detections]

    # Method A: least-squares circle fit on marker centroids
    pixel_to_mm_ratio = None
    if len(marker_coords) >= 3:
        try:
            cx_b, cy_b, radius_b = fit_circle_least_squares(marker_coords)
            deviations = [abs(np.sqrt((mx - cx_b) ** 2 + (my - cy_b) ** 2) - radius_b) for mx, my in marker_coords]
            avg_deviation = np.mean(deviations) if deviations else 0

            if REAL_WORLD_CHUCK_DIAMETER_MM is not None:
                pixel_to_mm_ratio = REAL_WORLD_CHUCK_DIAMETER_MM / (2 * radius_b)
                chuck_radius_mm = radius_b * pixel_to_mm_ratio
            else:
                chuck_radius_mm = None
                print("WARNING: REAL_WORLD_CHUCK_DIAMETER_MM not defined. Cannot calculate real-world measurements.")

            cv2.circle(img_display, (int(cx_b), int(cy_b)), int(radius_b), (0, 0, 255), 2)
            cv2.circle(img_display, (int(cx_b), int(cy_b)), 5, (0, 0, 255), -1)
            cv2.putText(img_display, f'B: ({cx_b:.2f}, {cy_b:.2f}) R={radius_b:.2f}px', (int(cx_b) + 10, int(cy_b) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(f"--- Method A (Least-Squares Circle Fit) ---")
            print(f"Center (x, y): ({cx_b:.2f}, {cy_b:.2f}) pixels")
            print(f"Radius: {radius_b:.2f} pixels")
            print(f"Average Deviation from fitted circle: {avg_deviation:.2f} pixels")

            if pixel_to_mm_ratio is not None:
                print(f"Pixel to MM ratio: {pixel_to_mm_ratio:.4f} mm/pixel")
                print(f"Chuck Radius: {chuck_radius_mm:.2f} mm")

            # Detect inner component circle (Hough) and annotate it
            comp = detect_component_circle(img, reference_center=(cx_b, cy_b))
            if comp is not None:
                comp_cx, comp_cy, comp_r = comp
                cv2.circle(img_display, (int(comp_cx), int(comp_cy)), int(comp_r), (0, 165, 255), 2)
                cv2.circle(img_display, (int(comp_cx), int(comp_cy)), 4, (0, 165, 255), -1)
                cv2.putText(img_display, f'Comp: ({comp_cx:.2f}, {comp_cy:.2f}) R={comp_r:.2f}px', (int(comp_cx) + 10, int(comp_cy) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                print(f"Detected component circle: Center=({comp_cx:.2f}, {comp_cy:.2f}) Radius={comp_r:.2f} px")

        except Exception as e:
            print(f"Could not perform Method A: {e}")

    # Method B: concentric circles from all marker contour points
    all_contour_points = []
    for d in detections:
        contour = d[3]
        for point in contour:
            all_contour_points.append(point[0])

    if all_contour_points:
        all_contour_points_np = np.array(all_contour_points, dtype=np.float32)
        c_x_c = float(np.mean(all_contour_points_np[:, 0]))
        c_y_c = float(np.mean(all_contour_points_np[:, 1]))
        distances = np.linalg.norm(all_contour_points_np - np.array([c_x_c, c_y_c]), axis=1)

        if distances.size:
            r_inner_c = float(np.min(distances))
            r_outer_c = float(np.max(distances))

            if pixel_to_mm_ratio is not None:
                r_inner_c_mm = r_inner_c * pixel_to_mm_ratio
                r_outer_c_mm = r_outer_c * pixel_to_mm_ratio
            else:
                r_inner_c_mm = None
                r_outer_c_mm = None

            cv2.circle(img_display, (int(c_x_c), int(c_y_c)), int(r_inner_c), (255, 0, 0), 1)
            cv2.circle(img_display, (int(c_x_c), int(c_y_c)), int(r_outer_c), (255, 0, 0), 1)
            cv2.circle(img_display, (int(c_x_c), int(c_y_c)), 5, (255, 0, 0), -1)
            cv2.putText(img_display, f'C: ({c_x_c:.2f}, {c_y_c:.2f}) R_in={r_inner_c:.2f}px R_out={r_outer_c:.2f}px', (int(c_x_c) + 10, int(c_y_c) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            print(f"\n--- Method B (Concentric Circles from Marker Edges) ---")
            print(f"Center (x, y): ({c_x_c:.2f}, {c_y_c:.2f}) pixels")
            print(f"Inner Radius: {r_inner_c:.2f} pixels")
            print(f"Outer Radius: {r_outer_c:.2f} pixels")
            if r_inner_c_mm is not None:
                print(f"Inner Radius: {r_inner_c_mm:.2f} mm")
                print(f"Outer Radius: {r_outer_c_mm:.2f} mm")
        else:
            print("No contour points found for Method B.")

    # Draw detected marker centroids and contours
    for cx, cy, area, contour in detections:
        cv2.circle(img_display, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        cv2.drawContours(img_display, [contour], -1, (0, 255, 255), 1)

    return img_display


def process_video_and_save(video_path, output_path=None):
    if video_path is None:
        video_path = input("Enter the video path: ").strip().strip('"')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not read video from path: {video_path}")
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
        print("Could not create output video writer. Try a different codec or file extension.")
        cap.release()
        return

    frame_count = 0
    print(f"Starting video processing: {video_path} (max frames: {MAX_FRAMES})", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        if MAX_FRAMES is not None and frame_count > MAX_FRAMES:
            print(f"Reached MAX_FRAMES={MAX_FRAMES}, stopping.", flush=True)
            break
        detections, mask = detect_markers(frame)
        annotated_frame = annotate_frame(frame, detections, mask)
        writer.write(annotated_frame)

        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")

    cap.release()
    writer.release()
    print(f"Saved annotated video to: {output_path}")


def process_image_and_display(image_path=None):
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = input("Enter the image path: ").strip().strip('"')

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image from path: {image_path}")
        return

    print(f"Processing image: {image_path}")

    detections, mask = detect_markers(img)
    print(f"Found {len(detections)} markers.")

    if EXPECTED_MARKERS is not None and len(detections) != EXPECTED_MARKERS:
        print(f"WARNING: Expected {EXPECTED_MARKERS} markers, but found {len(detections)}.")

    marker_coords = [(d[0], d[1]) for d in detections]
    if len(marker_coords) < 3:
        print("Not enough markers detected (less than 3) to fit a circle.")
        # still save original and mask
        base, ext = os.path.splitext(image_path)
        out_annot = f"{base}_annotated{ext}"
        out_mask = f"{base}_mask.png"
        cv2.imwrite(out_annot, img)
        cv2.imwrite(out_mask, mask)
        print(f"Saved: {out_annot}")
        print(f"Saved: {out_mask}")
        return

    annotated = annotate_frame(img, detections, mask)

    # Save annotated image and mask next to input file
    base, ext = os.path.splitext(image_path)
    out_annot = f"{base}_annotated{ext}"
    out_mask = f"{base}_mask.png"

    # Ensure annotated is BGR image (it is) and mask is single-channel
    cv2.imwrite(out_annot, annotated)
    cv2.imwrite(out_mask, mask)

    print("\n--- Visual Results Saved ---")
    print(f"Annotated image: {out_annot}")
    print(f"Mask image: {out_mask}")


def process_media(input_path=None):
    if input_path is None:
        input_path = sys.argv[1] if len(sys.argv) > 1 else None

    if input_path is None:
        input_path = input("Enter the image or video path: ").strip().strip('"')

    # Allow user to provide real-world chuck diameter once (env var preferred)
    global REAL_WORLD_CHUCK_DIAMETER_MM
    if REAL_WORLD_CHUCK_DIAMETER_MM is None:
        try:
            val = input("Enter real-world chuck diameter in mm (or press Enter to skip): ").strip()
            if val:
                REAL_WORLD_CHUCK_DIAMETER_MM = float(val)
        except Exception:
            print("Invalid input for REAL_WORLD_CHUCK_DIAMETER_MM; continuing without scaling.")

    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        process_video_and_save(input_path)
    else:
        process_image_and_display(input_path)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    process_media(input_path)