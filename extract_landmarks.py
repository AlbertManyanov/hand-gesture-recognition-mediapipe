import argparse
import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_image(path: Path):
    data = path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def create_detector(model_path: str, num_hands: int = 1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands
    )
    return vision.HandLandmarker.create_from_options(options)


def collect_image_paths(data_root: Path):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    image_list = []
    for label_dir in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not label_dir.is_dir():
            continue
        for image_path in sorted(label_dir.iterdir(), key=lambda p: p.name):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_list.append((label_dir.name, image_path))
    return image_list


def extract_landmarks(detector, image_path: Path):
    frame = read_image(image_path)
    if frame is None:
        return None, 'read_failed'

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, 'no_hand'

    landmarks = []
    hand_landmarks = result.hand_landmarks[0]
    for lm in hand_landmarks:
        landmarks.extend([lm.x, lm.y, lm.z])

    return landmarks, None


def write_csv(output_path: Path, rows, label_name='label'):
    header = [label_name, 'filename']
    header += [f'x{i}' for i in range(21)]
    header += [f'y{i}' for i in range(21)]
    header += [f'z{i}' for i in range(21)]

    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def write_failures(output_path: Path, failures):
    if not failures:
        return

    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'filename', 'error'])
        for failure in failures:
            writer.writerow(failure)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract 21 hand landmark points from labeled gesture images.'
    )
    parser.add_argument(
        '--data-root',
        default='data',
        help='Root folder containing letter directories with gesture images.'
    )
    parser.add_argument(
        '--model',
        default='model/hand_landmarker.task',
        help='Path to MediaPipe hand landmarker model.'
    )
    parser.add_argument(
        '--output',
        default='landmarks.csv',
        help='CSV file where landmark vectors will be saved.'
    )
    parser.add_argument(
        '--failures',
        default='landmark_failures.csv',
        help='CSV file where failed image results are saved.'
    )
    parser.add_argument(
        '--num-hands',
        type=int,
        default=1,
        help='Maximum number of hands to detect per image.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    failures_path = Path(args.failures)

    print(f'Loading model from: {args.model}')
    detector = create_detector(args.model, args.num_hands)

    image_list = collect_image_paths(data_root)
    print(f'Found {len(image_list)} images in {data_root}')

    rows = []
    failures = []
    counts = {}

    for label, image_path in image_list:
        landmarks, error = extract_landmarks(detector, image_path)
        counts.setdefault(label, {'ok': 0, 'failed': 0})

        if landmarks is not None:
            rel_path = os.path.relpath(image_path, data_root)
            rows.append([label, rel_path] + landmarks)
            counts[label]['ok'] += 1
        else:
            failures.append([label, os.path.relpath(image_path, data_root), error])
            counts[label]['failed'] += 1

    write_csv(output_path, rows)
    write_failures(failures_path, failures)

    print('--- Summary ---')
    for label in sorted(counts):
        ok = counts[label]['ok']
        failed = counts[label]['failed']
        print(f'{label}: {ok} extracted, {failed} failed')
    print(f'Total extracted: {len(rows)}')
    print(f'Landmark CSV: {output_path}')
    if failures:
        print(f'Failures CSV: {failures_path}')
    else:
        print('No failures.')


if __name__ == '__main__':
    main()
