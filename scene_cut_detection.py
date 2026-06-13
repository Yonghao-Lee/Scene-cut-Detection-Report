"""Scene-cut detection from frame-to-frame grayscale histogram distances.

Two distance measures are computed for every pair of consecutive frames:

* **PDF** -- L1 distance between the (normalised) per-bin histograms. Sensitive
  to small intensity shifts and per-frame noise, so it produces many spurious
  spikes inside a single scene.
* **CDF** -- L1 distance between the *cumulative* histograms (the 1-D
  Wasserstein distance up to a constant). Small shifts move probability mass
  between neighbouring bins, which barely changes the CDF, so it stays flat
  within a scene and spikes sharply at the true cut.

The detected cut is simply the frame pair with the largest CDF distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk without a display
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

# Rec. 601 luma weights used to collapse RGB to a single intensity channel.
GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert an (H, W, C) uint8 frame to a float grayscale image in [0, 1]."""
    return np.dot(frame.astype(np.float32)[..., :3] / 255.0, GRAYSCALE_WEIGHTS)


def histogram_distances(video_path: str, bins: int = 256):
    """Return ``(cdf_distances, pdf_distances)`` for consecutive frame pairs.

    Each returned array has length ``n_frames - 1`` (empty for < 2 frames).
    """
    pdfs, cdfs = [], []
    for frame in media.read_video(video_path):
        gray = to_grayscale(frame)
        hist, _ = np.histogram(gray, bins=bins, range=(0, 1))
        pdfs.append(hist / hist.sum())
        cdfs.append(np.cumsum(hist))

    pdfs, cdfs = np.asarray(pdfs), np.asarray(cdfs)
    if len(cdfs) < 2:
        empty = np.empty(0)
        return empty, empty

    cdf_distances = np.abs(np.diff(cdfs, axis=0)).sum(axis=1)
    pdf_distances = np.abs(np.diff(pdfs, axis=0)).sum(axis=1)
    return cdf_distances, pdf_distances


def detect_scene_cut(video_path: str, bins: int = 256) -> tuple[int, int]:
    """Detect the single hardest cut in a video.

    Returns ``(last_frame_of_scene_1, first_frame_of_scene_2)``, or ``(0, 0)``
    when the video has fewer than two frames.
    """
    cdf_distances, _ = histogram_distances(video_path, bins)
    if cdf_distances.size == 0:
        return 0, 0
    cut = int(np.argmax(cdf_distances))
    return cut, cut + 1


def plot_distance(cdf_distances: np.ndarray, name: str, out_dir: Path) -> Path:
    """Plot the CDF distance curve with the detected cut marked."""
    cut = int(np.argmax(cdf_distances))
    plt.figure(figsize=(12, 5))
    plt.plot(cdf_distances)
    plt.axvline(cut, color="r", linestyle="--", label=f"Detected cut at frame {cut}")
    plt.title(f"Frame-to-frame CDF distance - {name}", fontsize=15, fontweight="bold")
    plt.xlabel("Frame index")
    plt.ylabel("Cumulative-histogram L1 distance")
    plt.legend()
    plt.grid(True)
    return _save(out_dir / f"{name}_distance_plot.png")


def plot_comparison(cdf_distances, pdf_distances, name: str, out_dir: Path) -> Path:
    """Stack the CDF and PDF distance curves to contrast their stability."""
    fig, (ax_cdf, ax_pdf) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"CDF vs. PDF distance - {name}", fontsize=16, fontweight="bold")

    ax_cdf.plot(cdf_distances)
    ax_cdf.axvline(np.argmax(cdf_distances), color="g", linestyle="--", label="Detected cut")
    ax_cdf.set_title("Cumulative histogram (CDF) - flat baseline, sharp single peak")
    ax_cdf.set_ylabel("L1 distance")
    ax_cdf.legend()
    ax_cdf.grid(True)

    ax_pdf.plot(pdf_distances, color="tab:orange")
    ax_pdf.axvline(np.argmax(pdf_distances), color="r", linestyle="--", label="Largest peak")
    ax_pdf.set_title("Simple histogram (PDF) - noisy baseline, weak peak contrast")
    ax_pdf.set_xlabel("Frame index")
    ax_pdf.set_ylabel("L1 distance")
    ax_pdf.legend()
    ax_pdf.grid(True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save(out_dir / f"{name}_comparison_plot.png")


def _save(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", help="Path(s) to video file(s).")
    parser.add_argument("--bins", type=int, default=256, help="Histogram bins (default 256).")
    parser.add_argument("--plots", action="store_true", help="Also write distance/comparison plots.")
    parser.add_argument("--out", default="results", help="Directory for plots (default: results/).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    for video in args.videos:
        name = Path(video).stem
        last, first = detect_scene_cut(video, args.bins)
        print(f"{name}: scene cut between frame {last} and {first}")
        if args.plots:
            cdf_distances, pdf_distances = histogram_distances(video, args.bins)
            print("  wrote", plot_distance(cdf_distances, name, out_dir))
            print("  wrote", plot_comparison(cdf_distances, pdf_distances, name, out_dir))


if __name__ == "__main__":
    _cli()
