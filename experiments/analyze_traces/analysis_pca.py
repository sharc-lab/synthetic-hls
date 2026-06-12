import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_dist import (
    DEFAULT_MODEL_DIR,
    FINAL_COLOR,
    METRICS,
    SEED_COLOR,
    _evaluation_files,
    _numeric_metric,
    _read_json,
)
from matplotlib import patches
from scipy.spatial import ConvexHull, QhullError
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "seed_vs_final_design_pca.png"
)
DEFAULT_MDS_OUTPUT = (
    Path(__file__).resolve().parent / "figures" / "seed_vs_final_design_mds.png"
)
STAGE_ORDER = ("Passing seed designs", "Passing final designs")
LOG_SKEW_THRESHOLD = 1.0


def collect_design_vectors(
    model_dir: Path,
    domains: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    available_domains = sorted(
        path.name
        for path in model_dir.iterdir()
        if path.is_dir()
        and (path / "seed_designs" / "pass_designs").is_dir()
        and (path / "final_designs").is_dir()
    )
    selected_domains = list(domains) if domains else available_domains
    unknown = sorted(set(selected_domains) - set(available_domains))
    if unknown:
        raise ValueError(
            f"Unknown domains: {', '.join(unknown)}. "
            f"Available domains: {', '.join(available_domains)}"
        )
    if not selected_domains:
        raise ValueError(f"No domain data found under {model_dir}")

    rows = []
    omitted = {stage: 0 for stage in STAGE_ORDER}
    for domain in selected_domains:
        domain_dir = model_dir / domain
        groups = (
            (
                STAGE_ORDER[0],
                domain_dir / "seed_designs" / "pass_designs",
            ),
            (
                STAGE_ORDER[1],
                domain_dir / "final_designs",
            ),
        )
        for stage, root in groups:
            for eval_file in _evaluation_files(root):
                data = _read_json(eval_file)
                if str(data.get("status", "")).strip().lower() != "pass":
                    print(f"[WARN] Skipping non-passing design: {eval_file}")
                    continue

                metric_values = {
                    metric.key: _numeric_metric(data, metric) for metric in METRICS
                }
                missing = [
                    metric.key
                    for metric in METRICS
                    if metric_values[metric.key] is None
                ]
                if missing:
                    omitted[stage] += 1
                    print(
                        f"[WARN] Omitting incomplete design {eval_file}: "
                        f"missing {', '.join(missing)}"
                    )
                    continue

                rows.append(
                    {
                        **metric_values,
                        "Design Stage": stage,
                        "Domain": domain,
                        "Design": eval_file.parent.name,
                        "Evaluation File": str(eval_file),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete passing design vectors were found")
    for stage in STAGE_ORDER:
        if not (frame["Design Stage"] == stage).any():
            raise ValueError(f"No complete vectors found for {stage}")

    counts = frame["Design Stage"].value_counts()
    print(
        f"Loaded {counts.get(STAGE_ORDER[0], 0)} complete seed vectors and "
        f"{counts.get(STAGE_ORDER[1], 0)} complete final vectors from "
        f"{len(selected_domains)} domain(s)."
    )
    if any(omitted.values()):
        print(
            f"[WARN] Omitted {omitted[STAGE_ORDER[0]]} seed and "
            f"{omitted[STAGE_ORDER[1]]} final incomplete design(s)"
        )
    return frame, selected_domains


def preprocess_metrics(frame: pd.DataFrame) -> np.ndarray:
    metric_columns = [metric.key for metric in METRICS]
    transformed = frame[metric_columns].astype(float).copy()
    log_metrics = []
    standard_metrics = []

    for column in metric_columns:
        values = transformed[column]
        if values.min() >= 0 and values.skew() > LOG_SKEW_THRESHOLD:
            transformed[column] = np.log1p(values)
            log_metrics.append(column)
        else:
            standard_metrics.append(column)

    print(
        "Log1p + standard scaling: "
        + (", ".join(log_metrics) if log_metrics else "none")
    )
    print(
        "Standard scaling: "
        + (", ".join(standard_metrics) if standard_metrics else "none")
    )
    return StandardScaler().fit_transform(transformed)


def compute_pca(frame: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    standardized = preprocess_metrics(frame)
    pca = PCA(n_components=2)
    components = pca.fit_transform(standardized)

    result = frame.copy()
    result["dim1"] = components[:, 0]
    result["dim2"] = components[:, 1]
    return result, pca


def compute_manifold(frame: pd.DataFrame) -> tuple[pd.DataFrame, MDS]:
    standardized = preprocess_metrics(frame)
    mds = MDS(n_components=2, init="random", random_state=42)
    components = mds.fit_transform(standardized)
    result = frame.copy()
    result["dim1"] = components[:, 0]
    result["dim2"] = components[:, 1]
    return result, mds


def arc_patch(center, radius, theta1, theta2, resolution=16, **kwargs):
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1])
    )
    poly = patches.Polygon(points.T, closed=True, **kwargs)
    return poly


def draw_rounded_hull(
    x,
    hull,
    ax,
    padding=1.0,
    line_kwargs=None,
    fill_kwargs=None,
):
    default_line_kwargs = dict(color="black", linewidth=1)
    if line_kwargs is None:
        line_kwargs = default_line_kwargs
    else:
        line_kwargs = {**default_line_kwargs, **line_kwargs}

    default_fill_kwargs = dict(alpha=0.2)
    if fill_kwargs is None:
        fill_kwargs = default_fill_kwargs
    else:
        fill_kwargs = {**default_fill_kwargs, **fill_kwargs}

    hull_points = x[hull.vertices]
    hull_points = np.concatenate([hull_points[[-1]], hull_points, hull_points[[0]]])

    arcs = []
    arc_fills = []
    lines = []
    padded_points = []

    diameter = padding * 2
    for i in range(1, hull_points.shape[0] - 1):
        # line
        # source: https://stackoverflow.com/a/1243676/991496

        norm_next = np.flip(hull_points[i] - hull_points[i + 1]) * [-1, 1]
        # print(norm_next)
        norm_next /= np.linalg.norm(norm_next)

        norm_prev = np.flip(hull_points[i - 1] - hull_points[i]) * [-1, 1]
        # print(norm_prev)
        norm_prev /= np.linalg.norm(norm_prev)

        # plot line
        line = hull_points[i : i + 2] + norm_next * diameter / 2
        lines.append(line)
        # ax.plot(line[:, 0], line[:, 1], **line_kwargs)

        padded_points.append(line[0])
        padded_points.append(line[1])

        # arc
        angle_next = np.rad2deg(np.arccos(np.dot(norm_next, [1, 0])))
        if norm_next[1] < 0:
            angle_next = 360 - angle_next

        angle_prev = np.rad2deg(np.arccos(np.dot(norm_prev, [1, 0])))
        if norm_prev[1] < 0:
            angle_prev = 360 - angle_prev

        # print(angle_prev, angle_next)

        arc = patches.Arc(
            hull_points[i],
            diameter,
            diameter,
            angle=0,
            theta1=angle_prev,
            theta2=angle_next,
            **line_kwargs,
        )
        arcs.append(arc)

        theta_0 = angle_prev
        theta_1 = angle_next
        if theta_1 < theta_0:
            theta_1 += 360

        thetas = np.linspace(theta_0, theta_1, 16)
        points = np.vstack(
            (
                diameter / 2 * np.cos(np.radians(thetas)) + hull_points[i, 0],
                diameter / 2 * np.sin(np.radians(thetas)) + hull_points[i, 1],
            )
        )
        arc_fill = patches.Polygon(points.T, closed=True, **fill_kwargs, linewidth=0)
        arc_fills.append(arc_fill)

    padded_points = np.array(padded_points)
    polygon = patches.Polygon(padded_points, closed=True, **fill_kwargs, linewidth=0)
    ax.add_patch(polygon)
    for arc_fill in arc_fills:
        ax.add_patch(arc_fill)
    for line in lines:
        ax.plot(line[:, 0], line[:, 1], **line_kwargs)
    for arc in arcs:
        ax.add_patch(arc)


def draw_stage_hulls(
    frame: pd.DataFrame,
    ax: plt.Axes,
    stage_styles: dict[str, tuple[str, str]],
) -> None:
    all_points = frame[["dim1", "dim2"]].to_numpy()
    data_span = np.ptp(all_points, axis=0)
    padding = 0.025 * np.linalg.norm(data_span)

    for stage in STAGE_ORDER:
        points = (
            frame.loc[
                frame["Design Stage"] == stage,
                ["dim1", "dim2"],
            ]
            .drop_duplicates()
            .to_numpy()
        )
        if len(points) < 3:
            print(f"[WARN] Skipping hull for {stage}: fewer than 3 unique points")
            continue

        color, _ = stage_styles[stage]
        try:
            hull = ConvexHull(points)
        except QhullError:
            print(f"[WARN] Skipping hull for {stage}: points are collinear")
            continue

        draw_rounded_hull(
            points,
            hull,
            ax,
            padding=padding,
            line_kwargs={
                "color": color,
                "linewidth": 1.5,
                "zorder": 1,
            },
            fill_kwargs={
                "facecolor": color,
                "alpha": 0.10,
                "zorder": 0,
            },
        )


map_stage_label = {
    STAGE_ORDER[0]: "Seed Designs",
    STAGE_ORDER[1]: "Final Designs",
}


def plot_pca(
    frame: pd.DataFrame,
    pca: PCA,
    domains: Sequence[str],
    output: Path,
    *,
    show: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        },
    )
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    stage_styles = {
        STAGE_ORDER[0]: (SEED_COLOR, "."),
        STAGE_ORDER[1]: (FINAL_COLOR, "^"),
    }
    draw_stage_hulls(frame, ax, stage_styles)
    for stage in STAGE_ORDER:
        stage_frame = frame[frame["Design Stage"] == stage]
        color, marker = stage_styles[stage]
        ax.scatter(
            stage_frame["dim1"],
            stage_frame["dim2"],
            color=color,
            marker=marker,
            s=70,
            alpha=0.78,
            label=map_stage_label[stage],
            zorder=2,
        )

    # remove the x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    explained = 100.0 * pca.explained_variance_ratio_
    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}% explained variance)", fontweight="bold", fontsize=11
    )
    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}% explained variance)", fontweight="bold", fontsize=11
    )
    domain_text = "all domains" if len(domains) > 1 else domains[0].replace("_", " ")
    ax.set_title(
        "PCA of Seed Design and Final Design Metrics",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    legend = ax.legend(
        title="Design Stage",
        frameon=True,
        fontsize=9,
        loc="best",
    )
    legend.get_title().set_fontweight("bold")
    # ax.axhline(0, color="0.55", linewidth=0.8, linestyle=":", zorder=0)
    # ax.axvline(0, color="0.55", linewidth=0.8, linestyle=":", zorder=0)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(
        f"Saved PCA figure to {output}. "
        f"PC1 + PC2 explain {explained.sum():.1f}% of variance."
    )

    plt.close(fig)


def plot_mds(
    frame: pd.DataFrame,
    mds: MDS,
    domains: Sequence[str],
    output: Path,
    *,
    show: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.grid": True,
            "grid.alpha": 0.5,
        },
    )
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    stage_styles = {
        STAGE_ORDER[0]: (SEED_COLOR, "."),
        STAGE_ORDER[1]: (FINAL_COLOR, "^"),
    }
    draw_stage_hulls(frame, ax, stage_styles)
    for stage in STAGE_ORDER:
        stage_frame = frame[frame["Design Stage"] == stage]
        color, marker = stage_styles[stage]
        ax.scatter(
            stage_frame["dim1"],
            stage_frame["dim2"],
            color=color,
            marker=marker,
            s=70,
            alpha=0.78,
            label=map_stage_label[stage],
            zorder=2,
        )

    ax.set_xlabel("MDS Dimension 1", fontweight="bold")
    ax.set_ylabel("MDS Dimension 2", fontweight="bold")
    domain_text = "all domains" if len(domains) > 1 else domains[0].replace("_", " ")
    ax.set_title(
        f"MDS of Passing Seed and Final Designs ({domain_text})",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    legend = ax.legend(
        title="Design Stage",
        frameon=True,
        fontsize=9,
        loc="best",
    )
    legend.get_title().set_fontweight("bold")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved MDS figure to {output}. Final stress: {mds.stress_:.3f}.")
    if show:
        plt.show()
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA of normalized metrics for passing seed and final designs."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Model data directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Domain directory names to include. By default, all domains are pooled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PCA output figure path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mds-output",
        type=Path,
        default=DEFAULT_MDS_OUTPUT,
        help=f"MDS output figure path (default: {DEFAULT_MDS_OUTPUT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    design_vectors, domains = collect_design_vectors(args.model_dir, args.domains)
    pca_frame, pca = compute_pca(design_vectors)
    plot_pca(pca_frame, pca, domains, args.output, show=args.show)
    mds_frame, mds = compute_manifold(design_vectors)
    plot_mds(mds_frame, mds, domains, args.mds_output, show=args.show)


if __name__ == "__main__":
    main()
