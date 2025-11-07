#collapse_O.py
# This script creates an animation of a circle collapsing into a line and back to a circle.
import cv2
import numpy as np
from PIL import Image
import os

# SETTINGS
static_shape_file = "baseIMG.png"
background_file = "backgroundIMG.png"
output_gif_name = "_.gif"

frame_duration_ms = 20
morph_steps = 10
pause_duration_ms = 20
output_size = (1024, 1024)

# Load assets
img_circle = cv2.imread(static_shape_file)
background = cv2.imread(background_file)

if img_circle is None or background is None:
    raise FileNotFoundError("Missing background or static shape image.")

# Extract contour
def extract_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return contour[:, 0, :]

# Resample points
def resample_contour(contour, n_points=120):
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

# Extract base contour
contour_circle = resample_contour(extract_contour(img_circle), n_points=120)

# Generate "line" version by collapsing Y axis
center_y = np.mean(contour_circle[:, 1])
contour_line = contour_circle.copy()
contour_line[:, 1] = center_y  # flatten to a single Y value

# Build frames
frames = []
durations = []

# Morph: circle to line
for t in np.linspace(0, 1, morph_steps, endpoint=False):
    interp = (1 - t) * contour_circle + t * contour_line
    frame = background.copy()
    cv2.polylines(frame, [interp.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    durations.append(frame_duration_ms)

# Pause at full line
frame_line = background.copy()
cv2.polylines(frame_line, [contour_line.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
frames.append(cv2.cvtColor(frame_line, cv2.COLOR_BGR2RGB))
durations.append(pause_duration_ms)

# Morph: line to circle
for t in np.linspace(0, 1, morph_steps):
    interp = (1 - t) * contour_line + t * contour_circle
    frame = background.copy()
    cv2.polylines(frame, [interp.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    durations.append(frame_duration_ms)

# Final hold on circle
final_frame = background.copy()
cv2.polylines(final_frame, [contour_circle.reshape((-1,1,2)).astype(np.int32)], isClosed=True, color=(0,0,0), thickness=8)
frames.append(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
durations.append(pause_duration_ms)

# Save GIF
output_path = os.path.join("output", output_gif_name)
os.makedirs("output", exist_ok=True)
pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=durations)

print(f"Saved animation to {output_path}")
