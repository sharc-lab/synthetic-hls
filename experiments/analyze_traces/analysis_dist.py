import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, FuncFormatter, MaxNLocator

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "multi_targets"
    / "workspace_multi_targets"
    / "run__2026-03-31_02-07-58"
    / "gpt-oss-120b"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "figures"
    / "seed_vs_final_design_distributions.png"
)

SEED_COLOR = "#1F4E8C"
FINAL_COLOR = "#2A9D8F"


def _format_density_tick(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:g}"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    section: tuple[str, ...]
    scale: float = 1.0
    integer_ticks: bool = False


METRICS = (
    Metric(
        "latency_average_cycles",
        "Average Latency (cycles)",
        ("vitis_hls_tool_out", "data_tool"),
    ),
    Metric(
        "resources_lut_used",
        "LUTs Used",
        ("vitis_hls_tool_out", "data_tool"),
        integer_ticks=True,
    ),
    Metric(
        "resources_ff_used",
        "Flip-Flops Used",
        ("vitis_hls_tool_out", "data_tool"),
        integer_ticks=True,
    ),
    Metric(
        "resources_dsp_used",
        "DSPs Used",
        ("vitis_hls_tool_out", "data_tool"),
        integer_ticks=True,
    ),
    Metric(
        "resources_bram_used",
        "BRAMs Used",
        ("vitis_hls_tool_out", "data_tool"),
        integer_ticks=True,
    ),
    Metric(
        "num_functions",
        "Number of Functions",
        ("kernel_ast_out",),
        integer_ticks=True,
    ),
    Metric(
        "max_call_chain_depth",
        "Maximum Call-Chain Depth",
        ("kernel_ast_out",),
        integer_ticks=True,
    ),
    Metric(
        "kernel_total_lines",
        "Kernel Lines of Code",
        ("kernel_ast_out",),
        integer_ticks=True,
    ),
    Metric("average_function_lines", "Average Function Lines", ("kernel_ast_out",)),
)

LOG_SCALE_METRICS = {
    "latency_average_cycles",
    "resources_lut_used",
    "resources_ff_used",
    # "resources_dsp_used",
    # "resources_bram_used",
}
SYMLOG_SCALE_METRICS = {
    "resources_dsp_used",
    "resources_bram_used",
}


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation data from {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _evaluation_files(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Design directory not found: {root}")
    return sorted(root.glob("*/single_eval_data.json"))


def _numeric_metric(data: Mapping[str, object], metric: Metric) -> float | None:
    section: object = data
    for key in metric.section:
        if not isinstance(section, dict):
            return None
        section = section.get(key)
    if not isinstance(section, dict):
        return None
    value = section.get(metric.key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value) * metric.scale
    return value if math.isfinite(value) else None


def collect_metric_values(
    model_dir: Path,
    domains: Sequence[str] | None = None,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]], List[str]]:
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

    seed_values: Dict[str, List[float]] = defaultdict(list)
    final_values: Dict[str, List[float]] = defaultdict(list)
    design_counts = {"seed": 0, "final": 0}

    for domain in selected_domains:
        domain_dir = model_dir / domain
        groups = (
            ("seed", domain_dir / "seed_designs" / "pass_designs", seed_values),
            ("final", domain_dir / "final_designs", final_values),
        )
        for group_name, root, destination in groups:
            for eval_file in _evaluation_files(root):
                data = _read_json(eval_file)
                if str(data.get("status", "")).strip().lower() != "pass":
                    print(f"[WARN] Skipping non-passing design: {eval_file}")
                    continue
                design_counts[group_name] += 1
                for metric in METRICS:
                    value = _numeric_metric(data, metric)
                    if value is not None:
                        destination[metric.key].append(value)

    if design_counts["seed"] == 0 or design_counts["final"] == 0:
        raise ValueError(
            "Both seed and final groups must contain at least one passing design"
        )

    print(
        f"Loaded {design_counts['seed']} passing seed designs and "
        f"{design_counts['final']} passing final designs from "
        f"{len(selected_domains)} domain(s)."
    )
    for metric in METRICS:
        seed_missing = design_counts["seed"] - len(seed_values[metric.key])
        final_missing = design_counts["final"] - len(final_values[metric.key])
        if seed_missing or final_missing:
            print(
                f"[WARN] {metric.key}: omitted {seed_missing} seed and "
                f"{final_missing} final missing/non-finite value(s)"
            )

    return dict(seed_values), dict(final_values), selected_domains


def _plot_distributions(
    ax: Axes,
    seed_values: Sequence[float],
    final_values: Sequence[float],
    *,
    log_scale: bool = False,
    symlog_scale: bool = False,
) -> None:
    # if log_scale:
    #     seed_values = [value for value in seed_values if value > 0]
    #     final_values = [value for value in final_values if value > 0]

    stage_values = (
        ("Passing seed designs", list(seed_values), SEED_COLOR, 1.25, "--"),
        ("Passing final designs", list(final_values), FINAL_COLOR, 2.0, "-"),
    )
    plot_data = {
        "Metric": [value for _, values, _, _, _ in stage_values for value in values],
        "Design Stage": [
            label for label, values, _, _, _ in stage_values for _ in values
        ],
    }

    if not plot_data["Metric"]:
        return

    collections_before = len(ax.collections)
    sns.kdeplot(
        data=plot_data,
        x="Metric",
        hue="Design Stage",
        hue_order=["Passing seed designs", "Passing final designs"],
        palette={
            "Passing seed designs": SEED_COLOR,
            "Passing final designs": FINAL_COLOR,
        },
        ax=ax,
        fill=True,
        alpha=0.22,
        linewidth=2.0,
        cut=0,
        # bw_adjust=0.85,
        clip=(0, None),
        common_norm=False,
        common_grid=False,
        warn_singular=False,
        legend=False,
        log_scale=log_scale,
    )

    line_styles = {
        to_rgba(SEED_COLOR): (1.25, "--"),
        to_rgba(FINAL_COLOR): (2.0, "-"),
    }
    for collection in ax.collections[collections_before:]:
        edgecolors = collection.get_edgecolor()
        if not len(edgecolors):
            continue
        color = tuple(edgecolors[0])
        for target_color, (linewidth, linestyle) in line_styles.items():
            if all(
                math.isclose(a, b, abs_tol=1e-3) for a, b in zip(color, target_color)
            ):
                collection.set_linewidth(linewidth)
                collection.set_linestyle(linestyle)
                break

    # KDE is undefined for one sample or zero variance, so show those groups
    # as vertical lines after the single seaborn call.
    for _, values, color, linewidth, linestyle in stage_values:
        if values and (len(values) < 2 or math.isclose(min(values), max(values))):
            ax.axvline(
                values[0],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.95,
            )


def plot_distributions(
    seed_values: Mapping[str, Sequence[float]],
    final_values: Mapping[str, Sequence[float]],
    domains: Sequence[str],
    output: Path,
    *,
    show: bool = False,
) -> None:
    # sns.set_theme(
    #     # context="paper",
    #     # style="whitegrid",
    #     rc={
    #         "font.family": "sans-serif",
    #         "font.sans-serif": ["DejaVu Sans"],
    #         # "axes.edgecolor": "0.25",
    #         # "axes.linewidth": 0.8,
    #         # "grid.linestyle": ":",
    #         # "grid.linewidth": 0.55,
    #         # "grid.alpha": 0.5,
    #     },
    # )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 8.0), constrained_layout=False)

    for ax, metric in zip(axes.flat, METRICS):
        seed_metric_values = list(seed_values.get(metric.key, ()))
        final_metric_values = list(final_values.get(metric.key, ()))
        log_scale = metric.key in LOG_SCALE_METRICS
        symlog_scale = metric.key in SYMLOG_SCALE_METRICS
        _plot_distributions(
            ax,
            seed_metric_values,
            final_metric_values,
            log_scale=log_scale,
            symlog_scale=symlog_scale,
        )
        combined_values = seed_metric_values + final_metric_values
        if log_scale:
            combined_values = [value for value in combined_values if value > 0]
        if combined_values:
            data_min = min(combined_values)
            data_max = max(combined_values)
            if math.isclose(data_min, data_max):
                if log_scale:
                    ax.set_xlim(data_min / 1.1, data_max * 1.1)
                else:
                    padding = max(abs(data_min) * 0.01, 0.5)
                    ax.set_xlim(data_min - padding, data_max + padding)
            else:
                ax.set_xlim(data_min, data_max)
        ax.set_title(metric.label, fontweight="semibold", pad=7)
        ax.set_xlabel("Metric", fontweight="bold", fontsize=9)
        ax.set_ylabel("Density", fontweight="bold", fontsize=9)
        # ax.margins(x=0.04)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_density_tick))
        if not log_scale:
            if metric.integer_ticks:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if metric.key in {
            "latency_average_cycles",
            "resources_lut_used",
            "resources_ff_used",
            "kernel_total_lines",
        }:
            ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        # sns.despine(ax=ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SEED_COLOR,
            linewidth=1.25,
            linestyle="--",
            label="Seed Designs",
        ),
        Line2D(
            [0],
            [0],
            color=FINAL_COLOR,
            linewidth=2.0,
            linestyle="-",
            label="Final Designs",
        ),
    ]
    domain_text = "all domains" if len(domains) > 1 else domains[0].replace("_", " ")
    fig.suptitle(
        "Seed vs. Final Design Distributions",
        fontsize=18,
        fontweight="semibold",
        y=0.985,
    )
    design_legend = axes[1][2].legend(
        handles=legend_handles,
        title="Design Stage",
        loc="upper right",
        ncol=1,
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )
    design_legend.get_title().set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99), pad=0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {output}")

    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare metric distributions for passing seed and final HLS designs."
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
        help=f"Output figure path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_values, final_values, domains = collect_metric_values(
        args.model_dir,
        args.domains,
    )
    plot_distributions(
        seed_values,
        final_values,
        domains,
        args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
