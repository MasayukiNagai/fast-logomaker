"""Render the forward/reverse example these axis options are for.

Top: + strand, coordinates increase to the right, letters face forward.
Bottom: reverse complement of that same locus, coordinates decrease to the
right, letters still face forward. Mirroring is applied first so that
inverting the bottom axis does not leave the letters backwards.

Run from anywhere:

    python examples/preview_axis_options.py

The figure is written to examples/axis_options_preview.png.
Pass --show to also open an interactive window.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fast_logomaker import FastLogo  # noqa: E402


def _onehot(sequence):
    index = {base: i for i, base in enumerate("ACGT")}
    values = np.zeros((len(sequence), 4))
    for position, base in enumerate(sequence):
        values[position, index[base]] = 1.0
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an interactive window after saving",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "axis_options_preview.png",
        help="where to write the preview image",
    )
    args = parser.parse_args()

    # + strand 5' ATGC 3' at genomic positions 1000..1003.
    # The minus strand, read 5' to 3', is GCAT at 1003..1000.
    forward = _onehot("ATGC")
    reverse = _onehot("GCAT")
    values = np.stack([forward, reverse])
    positions = np.stack([
        np.arange(1000, 1004),
        np.arange(1003, 999, -1),
    ])

    logo = FastLogo(
        values,
        positions=positions,
        mirror_glyphs=[False, True],
        show_progress=False,
        fade_below=0,
        shade_below=0,
    )
    logo.process_all()

    plt.rcParams["figure.dpi"] = 140
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), layout="constrained")
    logo.draw_single(0, ax=axes[0], apply_layout=False)
    logo.draw_single(1, ax=axes[1], apply_layout=False)
    axes[1].xaxis.set_inverted(True)

    axes[0].set_title("+ strand: letters face forward, coordinates increase")
    axes[1].set_title("− strand: letters face forward, coordinates decrease")
    axes[1].set_xlabel("genomic position")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
