"""Make the README animation and the repository's social preview image.

    python docs/scripts/make_banner.py

Writes docs/assets/gridhunt.gif (four scenarios side by side) and docs/assets/social-preview.png
(1280 x 640, for the repository settings).
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from animation import frames, gif_bytes  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
BACKGROUND = (20, 20, 20)
TEXT = (235, 235, 235)
MUTED = (170, 170, 170)
SCENES = [
    ("GridHunt-ToxicStag-v0", "Toxic stag"),
    ("GridHunt-Specialists-v0", "Specialists"),
    ("GridHunt-Military-v0", "Military"),
    ("GridHunt-LandAndSea-v0", "Land and sea"),
]


def montage(tile=200, gap=12, label_height=28, steps=40):
    clips = [[im.resize((tile, tile), Image.LANCZOS) for im in frames(env_id, steps, crop=True)]
             for env_id, _ in SCENES]
    font = ImageFont.load_default(size=16)
    width = len(SCENES) * tile + (len(SCENES) + 1) * gap
    height = tile + 2 * gap + label_height
    out = []
    for step in range(steps):
        canvas = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        for index, ((_, title), clip) in enumerate(zip(SCENES, clips, strict=True)):
            x = gap + index * (tile + gap)
            canvas.paste(clip[step], (x, gap))
            text_width = draw.textlength(title, font=font)
            draw.text((x + (tile - text_width) / 2, gap + tile + 6), title, fill=MUTED, font=font)
        out.append(canvas)
    # One palette for every frame, without dithering, keeps the file small and the colours steady
    sample = Image.new("RGB", (width, height * 4))
    for row, step in enumerate(range(0, steps, max(1, steps // 4))[:4]):
        sample.paste(out[step], (0, row * height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in out]


def social_preview(tile=280, gap=30):
    canvas = Image.new("RGB", (1280, 640), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    left = (1280 - 3 * tile - 2 * gap) // 2
    draw.text((left, 40), "GridHunt", fill=TEXT, font=ImageFont.load_default(size=84))
    tagline = "Heterogeneous predator-prey environments\nfor multi-agent RL and human-agent teaming"
    draw.multiline_text((left + 4, 148), tagline, fill=MUTED, font=ImageFont.load_default(size=30), spacing=8)
    for index, (env_id, _) in enumerate(SCENES[:3]):
        image = frames(env_id, steps=20, crop=True)[-1].resize((tile, tile), Image.LANCZOS)
        canvas.paste(image, (left + index * (tile + gap), 640 - tile - 50))
    return canvas


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    gif = gif_bytes(montage())
    (ASSETS / "gridhunt.gif").write_bytes(gif)
    social_preview().save(ASSETS / "social-preview.png", optimize=True)
    print(f"gridhunt.gif {len(gif) / 1e3:.0f} KB, social-preview.png "
          f"{(ASSETS / 'social-preview.png').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
