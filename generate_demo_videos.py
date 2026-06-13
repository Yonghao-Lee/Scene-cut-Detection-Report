"""Generate self-contained synthetic videos with known scene cuts.

Each clip splices two visually distinct scenes at a fixed frame. Within every
scene there is deliberate per-frame motion and noise: that motion shuffles
probability mass between neighbouring intensity bins, which makes the simple
(PDF) histogram distance jitter wildly while the cumulative (CDF) distance
stays flat -- exactly the behaviour the detector exploits.

Run ``python generate_demo_videos.py`` to (re)create the clips in ``videos/``.
The ground-truth cut frame is encoded in each filename.
"""

from __future__ import annotations

from pathlib import Path

import mediapy as media
import numpy as np

H, W = 180, 240
FPS = 24
RNG = np.random.default_rng(0)


def _finish(gray: np.ndarray, tint: tuple[float, float, float]) -> np.ndarray:
    """Add noise, clip, tint and convert a float scene to a uint8 RGB frame."""
    gray = gray + RNG.normal(0.0, 0.03, gray.shape)
    gray = np.clip(gray, 0.0, 1.0)
    rgb = np.stack([gray * c for c in tint], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def gradient_pan(n: int) -> list[np.ndarray]:
    """A smooth horizontal gradient that slowly pans -- a gentle PDF jitter."""
    x = np.linspace(0.0, 1.0, W)
    frames = []
    for t in range(n):
        shift = 0.30 * np.sin(2 * np.pi * t / n)
        base = 0.20 + 0.60 * np.clip(np.tile(x + shift, (H, 1)), 0.0, 1.0)
        frames.append(_finish(base, tint=(1.05, 1.0, 0.85)))  # warm
    return frames


def spotlight(n: int) -> list[np.ndarray]:
    """A dark scene with a moving bright disc -- mass concentrated near zero."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for t in range(n):
        cx = W * (0.30 + 0.40 * (t / max(n - 1, 1)))
        cy = H * (0.50 + 0.20 * np.sin(2 * np.pi * t / n))
        disc = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 30.0 ** 2))
        frames.append(_finish(0.06 + 0.9 * disc, tint=(0.85, 0.95, 1.1)))  # cool
    return frames


def checker(n: int) -> list[np.ndarray]:
    """A high-frequency checkerboard that scrolls -- huge PDF jitter, flat CDF."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for t in range(n):
        pattern = ((xx + t * 3) // 16 + (yy // 16)) % 2
        frames.append(_finish(0.20 + 0.60 * pattern, tint=(1.0, 1.0, 1.0)))
    return frames


def sky(n: int) -> list[np.ndarray]:
    """A bright vertical gradient -- mass concentrated near one."""
    y = np.linspace(1.0, 0.55, H)[:, None]
    base = np.tile(y, (1, W))
    frames = []
    for t in range(n):
        drift = 0.03 * np.sin(2 * np.pi * t / n)
        frames.append(_finish(base + drift, tint=(0.9, 0.97, 1.1)))  # cool/bright
    return frames


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = {
        # name template -> (scene A, scene B, frames before cut, frames after cut)
        "demo1_pan_to_spotlight__cut{cut}": (gradient_pan, spotlight, 60, 60),
        "demo2_checker_to_sky__cut{cut}": (checker, sky, 75, 75),
    }
    for template, (scene_a, scene_b, n_a, n_b) in clips.items():
        frames = scene_a(n_a) + scene_b(n_b)
        cut = n_a - 1  # last frame of scene A
        path = out_dir / f"{template.format(cut=cut)}.mp4"
        media.write_video(path, np.asarray(frames), fps=FPS)
        print(f"wrote {path}  ({len(frames)} frames, ground-truth cut at {cut})")


if __name__ == "__main__":
    build(Path(__file__).parent / "videos")
