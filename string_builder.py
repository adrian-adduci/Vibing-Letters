#String Builder
import sys
import os
from PIL import Image, ImageSequence

# === Validate CLI input ===
if len(sys.argv) < 2:
    print("Usage: python string_to_gif.py \"YOURTEXT\"")
    sys.exit(1)

raw_input = sys.argv[1]
clean_text = "".join(c for c in raw_input.upper() if c.isalpha() or c == '.')

# === Absolute folders ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "input")
output_folder = os.path.join(script_dir, "output")
os.makedirs(output_folder, exist_ok=True)

# === Build frame sequence ===
combined_frames = []
durations = []

for char in clean_text:
    gif_filename = f"{char}.gif"
    gif_path = os.path.join(input_folder, gif_filename)

    print(f"🔍 Looking for: {gif_path}")  # Debug line

    if not os.path.exists(gif_path):
        print(f"⚠️  Missing: {gif_filename} — skipped.")
        continue

    with Image.open(gif_path) as gif:
        frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]
        dur = gif.info.get("duration", 5)
        combined_frames.extend(frames)
        durations.extend([dur] * len(frames))

# === Output filename ===
output_filename = f"{clean_text.lower().replace('.', '')}.gif"
output_path = os.path.join(output_folder, output_filename)

if combined_frames:
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=durations,
        loop=0
    )
    print(f"✅ Saved to: {output_path}")
else:
    print("❌ No valid GIFs found. Nothing saved.")
