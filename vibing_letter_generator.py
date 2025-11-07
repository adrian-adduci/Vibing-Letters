"""Vibing Animation Generator
This script generates a vibing animation from a set of images by morphing between shapes and adding effects.    
It takes  a static image and creates an animated GIF by morphing between the static shape and the shapes extracted from input images.\
"""


import os
import cv2
import numpy as np
from PIL import Image

# === SETTINGS ===
input_dir = "input"
output_dir = "output"
static_shape_file = "_Static_O.png"
background_file = "clean_background.png"
frame_duration_ms = 20
blank_pause_duration_ms = 60
overshoot_ts = [0.0, 1.1, 1.0]
reverse_ts = [1.0, 0.0]
vibration_cycles = 3
jitter_strength = 1.5
n_points = 120

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load common assets ===
base_shape = cv2.imread(static_shape_file)
background = cv2.imread(background_file)

if base_shape is None or background is None:
    raise FileNotFoundError("Static shape or background not found. Check file paths.")

def extract_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :]

def resample_contour(contour, n_points=100):
    perimeter = cv2.arcLength(contour.reshape((-1,1,2)), True)
    resampled = []
    distances = [0]
    for i in range(1, len(contour)):
        d = np.linalg.norm(contour[i] - contour[i-1])
        distances.append(d)
    distances = np.cumsum(distances)
    for i in np.linspace(0, distances[-1], n_points, endpoint=False):
        idx = np.searchsorted(distances, i)
        idx = min(idx, len(contour)-2)
        p1 = contour[idx]
        p2 = contour[idx+1]
        segment_length = np.linalg.norm(p2 - p1)
        if segment_length == 0:
            resampled.append(p1)
        else:
            ratio = (i - distances[idx]) / segment_length
            new_point = (1 - ratio) * p1 + ratio * p2
            resampled.append(new_point)
    return np.array(resampled)

# === Extract base points from _Static_O.png ===
points1 = resample_contour(extract_contour(base_shape), n_points=n_points)

# === Process each input file ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    filepath = os.path.join(input_dir, filename)
    spiky_img = cv2.imread(filepath)
    if spiky_img is None:
        print(f"Skipping {filename} — unreadable.")
        continue

    points2 = resample_contour(extract_contour(spiky_img), n_points=n_points)

    frames = []
    durations = []

    # Frame 1: static smooth shape
    frame_static = background.copy()
    cv2.polylines(frame_static, [points1.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
    frames.append(cv2.cvtColor(frame_static, cv2.COLOR_BGR2RGB))
    durations.append(frame_duration_ms * 3)

    # Morph overshoot
    for t in overshoot_ts:
        t = np.clip(t, -0.2, 1.2)
        interp = (1 - t) * points1 + t * points2
        frame = background.copy()
        cv2.polylines(frame, [interp.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        durations.append(frame_duration_ms)

    # Vibration frames
    for _ in range(vibration_cycles):
        jittered = points2 + np.random.normal(0, jitter_strength, size=points2.shape)
        frame = background.copy()
        cv2.polylines(frame, [jittered.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        durations.append(frame_duration_ms)

    # Reverse morph
    for t in reverse_ts:
        t = np.clip(t, -0.2, 1.2)
        interp = (1 - t) * points1 + t * points2
        frame = background.copy()
        cv2.polylines(frame, [interp.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        durations.append(frame_duration_ms)

    # Final clean frame
    final_frame = background.copy()
    cv2.polylines(final_frame, [points1.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
    frames.append(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    durations.append(frame_duration_ms * 2)

    # Blank collapse
    frames.append(cv2.cvtColor(background.copy(), cv2.COLOR_BGR2RGB))
    durations.append(blank_pause_duration_ms)

    # Save
    pil_frames = [Image.fromarray(f) for f in frames]
    output_filename = os.path.splitext(filename)[0] + "_animated.gif"
    output_path = os.path.join(output_dir, output_filename)
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=durations)

    print(f"Saved animation: {output_filename}")
