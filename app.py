import argparse
import csv
import math
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5
def cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )
def vector3(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])
def flatten_landmarks(hand_landmarks):
    return [
        coord
        for lm in hand_landmarks
        for coord in (lm.x, lm.y, lm.z)
    ]
def normalize_landmark_vector(vector):
    if len(vector) != 63:
        raise ValueError('Landmark vector must contain 63 values')
    points = [
        (vector[i], vector[i + 21], vector[i + 42])
        for i in range(21)
    ]
    wrist = points[0]
    deltas = [
        (x - wrist[0], y - wrist[1], z - wrist[2])
        for x, y, z in points
    ]
    scale = max(
        math.sqrt(dx * dx + dy * dy + dz * dz)
        for dx, dy, dz in deltas[1:]
    )
    scale = max(scale, 1e-6)
    return [coord / scale for delta in deltas for coord in delta]
def resolve_dataset_path(csv_path):
    path = Path(csv_path)
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parent / csv_path
    if fallback.exists():
        return fallback
    return path
def load_landmark_dataset(csv_path):
    dataset = []
    path = resolve_dataset_path(csv_path)
    if not path.exists():
        return dataset
    with path.open('r', encoding='utf-8', newline='') as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if not header or len(header) < 65:
            return dataset
        for row in reader:
            if len(row) < 65:
                continue
            try:
                label = row[0]
                coords = [float(value) for value in row[2:65]]
                dataset.append((label, normalize_landmark_vector(coords)))
            except ValueError:
                continue
    return dataset
def predict_label_from_dataset(dataset, hand_landmarks, k=5):
    if not dataset:
        return None
    live_vector = flatten_landmarks(hand_landmarks)
    normalized_live = normalize_landmark_vector(live_vector)
    distances = []
    for label, sample_vector in dataset:
        dist = math.sqrt(
            sum(
                (lv - sv) ** 2
                for lv, sv in zip(normalized_live, sample_vector)
            )
        )
        distances.append((dist, label))
    distances.sort(key=lambda item: item[0])
    nearest = distances[:k]
    if not nearest:
        return None
    votes = {}
    for _, label in nearest:
        votes[label] = votes.get(label, 0) + 1
    return max(votes.items(), key=lambda item: (item[1], -next(dist for dist, lab in nearest if lab == item[0])))[0]
def finger_raised(tip, pip):
    return tip[1] < pip[1]
def finger_direction(tip, pip):
    dx = tip[0] - pip[0]
    dy = tip[1] - pip[1]
    dz = tip[2] - pip[2]
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    if ax > ay and ax > az:
        return "Right" if dx > 0 else "Left"
    if ay > ax and ay > az:
        return "Up" if dy < 0 else "Down"
    if az > ax and az > ay:
        return "Forward" if dz < 0 else "Backward"
    return "Neutral"
def get_hand_properties(hand_landmarks):
    points2 = [(lm.x, lm.y) for lm in hand_landmarks]
    points3 = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    def is_extended(tip_idx, pip_idx, wrist_idx=0):
        d_tip = math.hypot(
            hand_landmarks[tip_idx].x - hand_landmarks[wrist_idx].x,
            hand_landmarks[tip_idx].y - hand_landmarks[wrist_idx].y
        )
        d_pip = math.hypot(
            hand_landmarks[pip_idx].x - hand_landmarks[wrist_idx].x,
            hand_landmarks[pip_idx].y - hand_landmarks[wrist_idx].y
        )
        return d_tip > d_pip * 1.15
    thumb_open = is_extended(4, 3)
    index_open = is_extended(8, 6)
    middle_open = is_extended(12, 10)
    ring_open = is_extended(16, 14)
    pinky_open = is_extended(20, 18)
    hand_vec = vector3(points3[0], points3[12])
    if abs(hand_vec[0]) > abs(hand_vec[1]):
        hand_direction = "RIGHT" if hand_vec[0] > 0 else "LEFT"
    else:
        hand_direction = "UP" if hand_vec[1] < 0 else "DOWN"
    palm_normal = cross3(
        vector3(points3[0], points3[5]),
        vector3(points3[0], points3[17])
    )
    palm_normal_length = math.sqrt(
        palm_normal[0]**2 + palm_normal[1]**2 + palm_normal[2]**2
    )
    palm_normal_ratio = (abs(palm_normal[2]) / palm_normal_length) if palm_normal_length > 0 else 1.0
    palm_sideways = palm_normal_length > 0 and palm_normal_ratio < 0.45
    palm_sideways_strict = palm_normal_length > 0 and palm_normal_ratio < 0.25
    palm_sideways_diagonal = palm_normal_length > 0 and 0.40 <= palm_normal_ratio < 0.65
    tips_z = np.mean([hand_landmarks[i].z for i in [4, 8, 12, 16, 20]])
    mcps_z = np.mean([hand_landmarks[i].z for i in [2, 5, 9, 13, 17]])
    hand_facing = "PALM" if tips_z < mcps_z else "BACK"
    finger_dirs = {
        "Thumb": finger_direction(points3[4], points3[3]),
        "Index": finger_direction(points3[8], points3[6]),
        "Middle": finger_direction(points3[12], points3[10]),
        "Ring": finger_direction(points3[16], points3[14]),
        "Pinky": finger_direction(points3[20], points3[18]),
    }
    palm_size = distance(points2[0], points2[9])
    return {
        "states": {
            "Thumb": thumb_open,
            "Index": index_open,
            "Middle": middle_open,
            "Ring": ring_open,
            "Pinky": pinky_open,
        },
        "hand_direction": hand_direction,
        "hand_facing": hand_facing,
        "finger_dirs": finger_dirs,
        "palm_size": palm_size,
        "palm_normal_ratio": palm_normal_ratio,
        "palm_sideways": palm_sideways,
        "palm_sideways_strict": palm_sideways_strict,
        "palm_sideways_diagonal": palm_sideways_diagonal,
    }
def classify_gesture(hand_props, hand_landmarks, dataset=None):
    states = hand_props["states"]
    hand_direction = hand_props["hand_direction"]
    hand_facing = hand_props["hand_facing"]
    finger_dirs = hand_props["finger_dirs"]
    palm_size = hand_props["palm_size"]
    palm_normal_ratio = hand_props.get(
        "palm_normal_ratio_smoothed",
        hand_props.get("palm_normal_ratio", 1.0)
    )
    palm_sideways = palm_normal_ratio < 0.45
    palm_sideways_strict = palm_normal_ratio < 0.25
    palm_sideways_diagonal = 0.40 <= palm_normal_ratio < 0.65
    points = [(lm.x, lm.y) for lm in hand_landmarks]
    thumb_index_distance = distance(points[4], points[8])
    fingers_open = [
        states[f]
        for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    ]
    open_count = sum(fingers_open)
    thumb_dy = points[4][1] - points[2][1]
    index_middle_distance = distance(points[8], points[12])
    index_middle_pip_distance = distance(points[6], points[10])
    index_middle_mcp_distance = distance(points[5], points[9])
    middle_ring_distance = distance(points[12], points[16])
    index_raised = finger_raised(points[8], points[6])
    middle_raised = finger_raised(points[12], points[10])
    middle_open = states["Middle"]
    ring_open = states["Ring"]
    pinky_open = states["Pinky"]
    thumb_index_close = thumb_index_distance < palm_size * 0.18
    all_tips_clustered = max(
        distance(points[i], points[j])
        for i, j in [(4, 8), (8, 12), (12, 16), (16, 20), (20, 4)]
    ) < palm_size * 0.18
    ring_tips_clustered = max(
        distance(points[i], points[j])
        for i, j in [(4, 12), (4, 16), (4, 20), (12, 16), (12, 20), (16, 20)]
    ) < palm_size * 0.18
    index_up = finger_dirs["Index"] == "Up"
    middle_up = finger_dirs["Middle"] == "Up"
    ring_up = finger_dirs["Ring"] == "Up"
    pinky_up = finger_dirs["Pinky"] == "Up"
    index_down = finger_dirs["Index"] == "Down"
    middle_down = finger_dirs["Middle"] == "Down"
    ring_down = finger_dirs["Ring"] == "Down"
    index_curve = index_raised and not index_up
    thumb_horizontal = finger_dirs["Thumb"] in ("Right", "Left")
    adjacent_fingers_close = (
        distance(points[8], points[12]) < palm_size * 0.18 and
        distance(points[12], points[16]) < palm_size * 0.18
    )
    ring_shape_loose = (
        max(
            distance(points[i], points[j])
            for i, j in [(4, 8), (8, 12), (12, 16), (16, 20), (20, 4)]
        ) < palm_size * 0.30
        and not all_tips_clustered
    )
    horizontal_fingers = all(
        finger_dirs[f] in ("Right", "Left")
        for f in ["Index", "Middle", "Ring", "Pinky"]
    )
    if (
        not states["Thumb"] and
        not states["Index"] and
        not states["Middle"] and
        states["Ring"] and
        states["Pinky"] and
        ring_up and
        pinky_up
    ):
        return "И"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        index_up and
        middle_up and
        hand_facing == "PALM"
    ):
        return "К"
    same_LP_state = (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"]
    )
    if same_LP_state:
        if (
            index_middle_distance < palm_size * 0.24 and
            index_middle_pip_distance < palm_size * 0.22
        ):
            return "П"
        if (
            index_middle_distance > palm_size * 0.30 and
            index_middle_pip_distance > palm_size * 0.26
        ):
            return "Л"      
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        states["Pinky"] and
        index_up and
        middle_up and
        pinky_up
    ):
        return "Н"
    if (
        thumb_index_close and
        middle_up and
        ring_up and
        pinky_up
    ):
        return "О"
    if (
        not states["Thumb"] and
        states["Index"] and
        not states["Middle"] and
        states["Ring"] and
        states["Pinky"] and
        index_up and
        ring_up and
        pinky_up
    ):
        return "Р"
    if (
        not states["Thumb"] and
        not states["Index"] and
        states["Middle"] and
        states["Ring"] and
        states["Pinky"] and
        thumb_index_close and
        ring_shape_loose and
        palm_sideways
    ):
        return "С"
    fingers_down_for_mt = (
        finger_dirs["Thumb"] == "Down" and
        index_down and
        middle_down and
        ring_down
    )
    spread_like_l = (
        index_middle_distance > palm_size * 0.22 and
        middle_ring_distance > palm_size * 0.22
    )
    closed_like_p = (
        index_middle_distance < palm_size * 0.18 and
        middle_ring_distance < palm_size * 0.18
    )
    if (
        states["Index"] and
        states["Middle"] and
        states["Ring"] and
        not states["Pinky"] and
        fingers_down_for_mt and
        spread_like_l
    ):
        return "М"
    if (
        states["Index"] and
        states["Middle"] and
        states["Ring"] and
        not states["Pinky"] and
        fingers_down_for_mt and
        closed_like_p
    ):
        return "Т"
    if (
        thumb_horizontal and
        pinky_up and
        not states["Index"] and
        not states["Middle"] and
        not states["Ring"]
    ):
        return "У"
    if (
        open_count == 5 and
        palm_sideways and
        any(
            finger_dirs[f] == "Down"
            for f in ["Index", "Middle", "Ring", "Pinky"]
        )
    ):
        return "Ф"
    if (
        not states["Thumb"] and
        states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        index_curve and
        palm_sideways
    ):
        return "Х"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        hand_facing == "BACK" and
        index_up and
        middle_up
    ):
        return "Ц"
    if (
        thumb_index_close and
        states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        palm_sideways
    ):
        return "Ч"
    if (
        thumb_index_close and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        not states["Thumb"]
    ):
        return "Э"
    if (
        not states["Thumb"] and
        not states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        states["Pinky"] and
        pinky_up and
        palm_sideways and
        thumb_index_close
    ):
        return "Ю"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        index_up and
        middle_up and
        index_middle_distance < palm_size * 0.12
    ):
        return "Я"
    if (
        open_count == 5 and
        palm_sideways and
        horizontal_fingers
    ):
        return "Ж"
    if (
        not states["Thumb"] and
        states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        index_raised and
        index_up and
        palm_sideways and
        ring_tips_clustered
    ):
        return "З"
    if (
        states["Thumb"] and
        states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        thumb_index_distance >= palm_size * 0.18 and
        not ring_tips_clustered
    ):
        return "Г"

    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        palm_sideways_diagonal and
        index_raised and
        middle_raised
    ):
        return "Б"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        palm_sideways_strict and
        index_raised and
        middle_raised
    ):
        return "Д"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        not palm_sideways and
        not palm_sideways_diagonal and
        index_raised and
        middle_raised
    ):
        return "НЕИЗВЕСТНО"
    if open_count == 5 and hand_facing == "PALM":
        return "В"
    if open_count == 0 and hand_direction in ("RIGHT", "LEFT"):
        return "А"
    if (
        states["Thumb"] and
        not states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        not states["Pinky"] and
        thumb_index_distance > palm_size * 0.35
    ):
        if thumb_dy < -0.03:
            return "ЛАЙК"
        if thumb_dy > 0.03:
            return "ДИЗЛАЙК"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        not states["Ring"] and
        not states["Pinky"]
    ):
        if index_raised and middle_raised:
            return "НЕИЗВЕСТНО"
        predicted = None
        if dataset:
            predicted = predict_label_from_dataset(dataset, hand_landmarks)
        return predicted or "НЕИЗВЕСТНО"
    if (
        not states["Thumb"] and
        states["Index"] and
        not states["Middle"] and
        not states["Ring"] and
        states["Pinky"]
    ):
        return "Ы"
    if (
        all_tips_clustered and
        thumb_index_close and
        not (open_count == 0)
    ):
        return "Е"
    if (
        not states["Thumb"] and
        states["Index"] and
        states["Middle"] and
        states["Ring"] and
        not states["Pinky"] and
        adjacent_fingers_close and
        index_up and
        middle_up and
        ring_up
    ):
        return "Ш"
    if dataset:
        predicted = predict_label_from_dataset(dataset, hand_landmarks)
        if predicted:
            return predicted
    return "НЕИЗВЕСТНО"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI"
    )
    parser.add_argument(
        "--dataset",
        default="landmarks.csv",
        help="CSV file with labeled 21-point landmark records"
    )
    return parser.parse_args()
def main():
    args = get_args()
    model_path = "model/hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    detector = vision.HandLandmarker.create_from_options(options)
    dataset_path = resolve_dataset_path(args.dataset)
    dataset = load_landmark_dataset(dataset_path)
    if dataset:
        print(f"Loaded {len(dataset)} labeled landmark records from {dataset_path}")
    else:
        print(f"No dataset loaded from {dataset_path}. Using rule-based detection only.")
    def find_available_camera():
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera {i} is available")
                cap.release()
                return i
        return None
    camera_index = find_available_camera()
    if camera_index is None:
        print("No camera found!")
        return
    print(f"Using camera {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Failed to open camera!")
        return
    print("Camera opened successfully!")
    if not args.headless:
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    label_root = None
    label_var = None
    if not args.headless:
        label_root = tk.Tk()
        label_root.title("Gesture Label")
        label_root.geometry("320x100")
        label_root.attributes("-topmost", True)
        label_var = tk.StringVar(value="Жест: ")
        label = tk.Label(
            label_root,
            textvariable=label_var,
            font=("Segoe UI", 32),
            bg="white",
            fg="black"
        )
        label.pack(fill="both", expand=True, padx=10, pady=10)
    def draw(frame, result, gesture_text="", hand_props=None):
        h, w, _ = frame.shape
        if not result.hand_landmarks:
            return
        for hand in result.hand_landmarks:
            points = []
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
                cv2.circle(frame, (x, y), 5,
                           (0, 255, 255), -1)
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]
            for c in connections:
                cv2.line(
                    frame,
                    points[c[0]],
                    points[c[1]],
                    (255, 0, 0),
                    2
                )
        y = 30
        if hand_props:
            cv2.putText(
                frame,
                f"Hand: {hand_props['hand_facing']} facing, {hand_props['hand_direction']}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )
            y += 25
            for name, direction in hand_props["finger_dirs"].items():
                cv2.putText(
                    frame,
                    f"{name}: {direction}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                y += 25
    print("Starting detection loop...")
    frame_count = 0
    palm_ratio_history = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame!")
                break
            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = detector.detect(mp_image)
            gesture_text = ""
            hand_props = None
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    hand_props = get_hand_properties(
                        hand_landmarks
                    )
                    palm_ratio_history.append(hand_props.get("palm_normal_ratio", 1.0))
                    if len(palm_ratio_history) > 5:
                        palm_ratio_history.pop(0)
                    hand_props["palm_normal_ratio_smoothed"] = (
                        sum(palm_ratio_history) / len(palm_ratio_history)
                    )
                    gesture_text = classify_gesture(
                        hand_props,
                        hand_landmarks,
                        dataset=dataset
                    )
                    break
            draw(
                frame,
                result,
                gesture_text,
                hand_props
            )
            if label_root is not None and label_var is not None:
                label_var.set(
                    f"Жест: {gesture_text}" if gesture_text else "Жест: —"
                )
                if not label_root.winfo_exists():
                    break
                try:
                    label_root.update_idletasks()
                    label_root.update()
                except tk.TclError:
                    break
            if args.headless:
                cv2.imwrite("test_camera.jpg", frame)
                print(
                    f"Frame {frame_count} "
                    f"saved to test_camera.jpg"
                )
                break
            else:
                cv2.imshow(
                    "Hand Tracking",
                    frame
                )
                if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
    finally:
        print(f"Total frames processed: {frame_count}")
        cap.release()
        if label_root is not None:
            try:
                label_root.destroy()
            except tk.TclError:
                pass
        cv2.destroyAllWindows()
        print("Done!")
if __name__ == '__main__':
    main()