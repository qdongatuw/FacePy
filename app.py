from __future__ import annotations

import argparse
import math
import time
import wave
from pathlib import Path

import cv2
import mediapipe as mp
import pygame


ROOT = Path(__file__).resolve().parent
DEFAULT_SOUND_PATH = ROOT / "assets" / "mouth_open.wav"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


def ensure_default_sound(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44_100
    duration = 0.18
    frequency = 880.0
    volume = 0.35
    total_samples = int(sample_rate * duration)

    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for index in range(total_samples):
            fade = min(index / 800, (total_samples - index) / 1200, 1.0)
            value = int(32767 * volume * fade * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))

        wav_file.writeframes(bytes(frames))


def landmark_point(landmarks, index: int, width: int, height: int) -> tuple[int, int]:
    landmark = landmarks[index]
    return int(landmark.x * width), int(landmark.y * height)


def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mouth_open_ratio(landmarks, width: int, height: int) -> float:
    upper = landmark_point(landmarks, UPPER_LIP, width, height)
    lower = landmark_point(landmarks, LOWER_LIP, width, height)
    left = landmark_point(landmarks, LEFT_MOUTH_CORNER, width, height)
    right = landmark_point(landmarks, RIGHT_MOUTH_CORNER, width, height)

    mouth_width = max(distance(left, right), 1.0)
    return distance(upper, lower) / mouth_width


def draw_polyline(frame, landmarks, indices: list[int], color: tuple[int, int, int], closed: bool = True) -> None:
    height, width = frame.shape[:2]
    points = [landmark_point(landmarks, index, width, height) for index in indices]
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
    if closed and len(points) > 2:
        cv2.line(frame, points[-1], points[0], color, 2, cv2.LINE_AA)


def init_sound(sound_path: Path) -> pygame.mixer.Sound | None:
    try:
        pygame.mixer.init()
        return pygame.mixer.Sound(str(sound_path))
    except pygame.error as exc:
        print(f"Warning: sound disabled because pygame mixer failed to initialize: {exc}")
        return None


def run_app(camera_index: int, sound_path: Path, threshold: float, cooldown: float) -> int:
    ensure_default_sound(sound_path)
    sound = init_sound(sound_path)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Error: could not open camera index {camera_index}.")
        return 1

    last_played_at = 0.0
    window_name = "FacePy - press Q or Esc to quit"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: failed to read from camera.")
                return 1

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status = "No face"
            status_color = (180, 180, 180)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ratio = mouth_open_ratio(landmarks, width, height)
                is_open = ratio >= threshold

                eye_color = (80, 220, 255)
                mouth_color = (0, 80, 255) if is_open else (60, 255, 120)

                draw_polyline(frame, landmarks, LEFT_EYE, eye_color)
                draw_polyline(frame, landmarks, RIGHT_EYE, eye_color)
                draw_polyline(frame, landmarks, OUTER_MOUTH, mouth_color)
                draw_polyline(frame, landmarks, INNER_MOUTH, mouth_color)

                status = f"Mouth ratio: {ratio:.2f} / threshold: {threshold:.2f}"
                status_color = mouth_color

                now = time.monotonic()
                if is_open and sound and now - last_played_at >= cooldown:
                    sound.play()
                    last_played_at = now

            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Press Q or Esc to quit", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    finally:
        capture.release()
        face_mesh.close()
        pygame.mixer.quit()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track eyes and mouth with a webcam, and play a sound when the mouth opens.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open. Default: 0")
    parser.add_argument("--sound", type=Path, default=DEFAULT_SOUND_PATH, help="WAV sound to play when the mouth opens.")
    parser.add_argument("--threshold", type=float, default=0.34, help="Mouth-open ratio threshold. Default: 0.34")
    parser.add_argument("--cooldown", type=float, default=0.8, help="Minimum seconds between sound plays. Default: 0.8")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_app(args.camera, args.sound, args.threshold, args.cooldown))
