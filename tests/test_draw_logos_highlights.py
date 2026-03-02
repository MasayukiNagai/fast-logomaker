import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fast_logomaker import BatchLogo


def _build_logo(n_logos=3, length=12):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((n_logos, length, 4))
    logo = BatchLogo(values, show_progress=False)
    logo.process_all()
    return logo


def _used_axes(axes, n):
    return list(axes.flatten()[:n])


def test_draw_logos_global_highlights_apply_to_all_axes():
    logo = _build_logo(n_logos=3, length=12)
    indices = [0, 1, 2]

    fig, axes = logo.draw_logos(
        indices=indices,
        rows=1,
        cols=3,
        highlight_ranges=[(2, 4), (7, 9)],
        highlight_colors=["lightcyan", "honeydew"],
        highlight_alpha=0.3,
    )

    for ax in _used_axes(axes, len(indices)):
        assert len(ax.patches) == 2
        assert all(patch.get_alpha() == pytest.approx(0.3) for patch in ax.patches)

    plt.close(fig)


def test_draw_logos_per_logo_nested_highlights():
    logo = _build_logo(n_logos=3, length=12)
    indices = [0, 1, 2]

    fig, axes = logo.draw_logos(
        indices=indices,
        rows=1,
        cols=3,
        highlight_ranges=[[(1, 3)], [(0, 2), (4, 6)], []],
        highlight_colors=[["mistyrose"], ["honeydew", "lavender"], []],
        highlight_alpha=[0.2, 0.4, 0.6],
    )

    expected_counts = [1, 2, 0]
    expected_alphas = [0.2, 0.4, 0.6]

    for i, ax in enumerate(_used_axes(axes, len(indices))):
        assert len(ax.patches) == expected_counts[i]
        if expected_counts[i] > 0:
            assert all(
                patch.get_alpha() == pytest.approx(expected_alphas[i])
                for patch in ax.patches
            )

    plt.close(fig)


def test_draw_logos_position_lists_remain_global():
    logo = _build_logo(n_logos=2, length=12)
    indices = [0, 1]

    fig, axes = logo.draw_logos(
        indices=indices,
        rows=1,
        cols=2,
        highlight_ranges=[[1, 2, 3], [8, 9]],
        highlight_alpha=0.7,
    )

    for ax in _used_axes(axes, len(indices)):
        assert len(ax.patches) == 2
        assert all(patch.get_alpha() == pytest.approx(0.7) for patch in ax.patches)

    plt.close(fig)


def test_draw_logos_per_logo_validation_errors():
    logo = _build_logo(n_logos=3, length=12)

    with pytest.raises(ValueError, match="expected 3 entries"):
        logo.draw_logos(
            indices=[0, 1, 2],
            rows=1,
            cols=3,
            highlight_ranges=[[(1, 3)], []],
        )

    with pytest.raises(ValueError, match="highlight_alpha"):
        logo.draw_logos(
            indices=[0, 1, 2],
            rows=1,
            cols=3,
            highlight_ranges=[[(1, 3)], [], [(5, 7)]],
            highlight_alpha=[0.2, 0.5],
        )


def test_draw_logos_per_logo_empty_entry_has_no_highlights():
    logo = _build_logo(n_logos=3, length=12)
    indices = [0, 1, 2]

    fig, axes = logo.draw_logos(
        indices=indices,
        rows=1,
        cols=3,
        highlight_ranges=[[(1, 2)], [], [(7, 9)]],
        highlight_colors=[["lightcyan"], [], ["honeydew"]],
        highlight_alpha=0.5,
    )

    expected_counts = [1, 0, 1]
    for i, ax in enumerate(_used_axes(axes, len(indices))):
        assert len(ax.patches) == expected_counts[i]

    plt.close(fig)
