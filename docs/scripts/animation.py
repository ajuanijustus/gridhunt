"""Animated frames of scenarios played by the scripted team, for the docs and the README."""

import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import gridhunt  # noqa: E402
from gridhunt.policies import GreedyPolicy  # noqa: E402


def frames(env_id, steps=48, seed=3, crop=False):
    """RGB frames of the scripted team playing a scenario, with the whole board visible.

    crop: trim the dark margin around the board.
    """
    env = gridhunt.make(env_id, render_mode="rgb_array", gui_config={"vision_mask": None})
    env.reset(seed=seed)
    policy = GreedyPolicy()
    images = []
    for _ in range(steps):
        images.append(env.render())
        env.step({agent: policy(None, agent, env) for agent in env.agents})
    env.close()
    if crop:
        background = images[0][0, 0]
        rows, cols = np.where((images[0] != background).any(axis=2))
        images = [image[rows.min():rows.max() + 1, cols.min():cols.max() + 1] for image in images]
    return [Image.fromarray(image) for image in images]


def shared_palette(images, colors=128):
    """Quantize frames to one palette without dithering: small files and steady colours."""
    samples = images[:: max(1, len(images) // 4)][:4]
    width, height = images[0].size
    sample = Image.new("RGB", (width, height * len(samples)))
    for row, image in enumerate(samples):
        sample.paste(image.convert("RGB"), (0, row * height))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return [image.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE) for image in images]


def gif_bytes(images, duration=180):
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0,
                   optimize=True)
    return buffer.getvalue()


def scenario_gif(env_id, steps=48, width=360):
    """An animation of one scenario, scaled smoothly to ``width`` pixels."""
    images = frames(env_id, steps)
    scaled = [im.resize((width, int(im.height * width / im.width)), Image.LANCZOS) for im in images]
    return gif_bytes(shared_palette(scaled))
