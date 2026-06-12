import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "multi_targets"
    / "workspace_multi_targets"
    / "run__2026-03-31_02-07-58"
    / "gpt-oss-120b"
)
DEFAULT_OUTPUT_STEM = (
    Path(__file__).resolve().parent / "figures" / "pass_at_k_by_stage_domains"
)
DEFAULT_PASSING_PERCENT_OUTPUT_STEM = (
    Path(__file__).resolve().parent / "figures" / "passing_percent_by_stage_domains"
)
DEFAULT_PLOT_TITLE = "Valid Design Pass Rates Across Generation and Mutation Stages"
DEFAULT_PASSING_PERCENT_TITLE = (
    "Average Valid Design Percentage Across Generation and Mutation Stages"
)
PASS_K_VALUES = (1, 8)

DOMAIN_LABELS = {
    "ml_ai": "ML / AI",
    "sci_sim": "Scientific Simulation",
    "fin_model": "Financial Modeling",
    "eng_sim": "Engineering Simulation",
    "data_big": "Data Analytics",
    "gfx_render": "Graphics Rendering",
    "crypto_bc": "Cryptography",
    "telecom_sp": "Telecommunications",
    "astro": "Astronomy",
    "health_med": "Healthcare",
}

DOMAIN_COLORS = {
    "astro": "#3366CC",
    "crypto_bc": "#E67E22",
    "data_big": "#00876C",
    "eng_sim": "#C62828",
    "fin_model": "#6A1B9A",
    "gfx_render": "#C2185B",
    "health_med": "#8C564B",
    "ml_ai": "#5B8FF9",
    "sci_sim": "#6B8E23",
    "telecom_sp": "#0097A7",
}

TARGET_LABELS = {
    "num_functions": "# of Func.",
    "max_call_chain_depth": "Max Call Depth",
    "average_function_lines": "Avg. Func. LoC",
    "pareto_scores": "Pareto Score",
}

_SEED_RE = re.compile(r"seed_design_(\d+)$")
_ITER_SAMPLES_RE = re.compile(r"iter_(\d+)_samples$")
_SAMPLE_RE = re.compile(r"sample_(\d+)(.*)$")


@dataclass(frozen=True)
class SamplePool:
    name: str
    total: int
    passed: int


@dataclass(frozen=True)
class StageResult:
    name: str
    pools: int
    samples: int
    passed: int
    pass_at_k: Dict[int, float]
    average_passing_percent: float


def _numeric_path_key(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.fullmatch(path.name)
    return int(match.group(1)) if match else math.inf


def _is_pass(eval_file: Path) -> bool:
    try:
        data = json.loads(eval_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation data from {eval_file}") from exc
    return str(data.get("status", "")).strip().lower() == "pass"


def _sample_retry_rank(suffix: str) -> Tuple[int, str]:
    """Rank retries above the original sample, with deterministic tie-breaking."""
    return (0 if suffix == "" else 1, suffix)


def _effective_sample_dirs(samples_dir: Path) -> List[Path]:
    """Return one result per sample index, preferring a retry/fix if present."""
    by_index: Dict[int, Path] = {}
    ranks: Dict[int, Tuple[int, str]] = {}

    for path in samples_dir.iterdir():
        if not path.is_dir():
            continue
        match = _SAMPLE_RE.fullmatch(path.name)
        if not match:
            continue
        sample_index = int(match.group(1))
        rank = _sample_retry_rank(match.group(2))
        if sample_index not in by_index or rank > ranks[sample_index]:
            by_index[sample_index] = path
            ranks[sample_index] = rank

    return [by_index[index] for index in sorted(by_index)]


def _pool_from_sample_dirs(name: str, sample_dirs: Iterable[Path]) -> SamplePool:
    eval_files = []
    for sample_dir in sample_dirs:
        eval_file = sample_dir / "single_eval_data.json"
        if eval_file.is_file():
            eval_files.append(eval_file)
        else:
            print(f"[WARN] Missing evaluation file: {eval_file}")

    return SamplePool(
        name=name,
        total=len(eval_files),
        passed=sum(_is_pass(eval_file) for eval_file in eval_files),
    )


def estimate_pass_at_k(total: int, passed: int, k: int) -> float:
    """Unbiased pass@k estimator using the stable product formulation."""
    if total <= 0:
        raise ValueError("pass@k requires at least one sample")
    if not 0 <= passed <= total:
        raise ValueError(f"Invalid pass count: {passed}/{total}")
    if k <= 0:
        raise ValueError("k must be positive")

    if total - passed < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(total - passed + 1, total + 1)))


def _summarize_stage(name: str, pools: Sequence[SamplePool]) -> StageResult:
    usable_pools = [pool for pool in pools if pool.total > 0]
    if not usable_pools:
        raise ValueError(f"No evaluated samples found for stage {name}")

    averages = {
        k: 100.0
        * sum(estimate_pass_at_k(pool.total, pool.passed, k) for pool in usable_pools)
        / len(usable_pools)
        for k in PASS_K_VALUES
    }
    return StageResult(
        name=name,
        pools=len(usable_pools),
        samples=sum(pool.total for pool in usable_pools),
        passed=sum(pool.passed for pool in usable_pools),
        pass_at_k=averages,
        average_passing_percent=(
            100.0
            * sum(pool.passed / pool.total for pool in usable_pools)
            / len(usable_pools)
        ),
    )


def collect_stage_results(data_dir: Path) -> List[StageResult]:
    seed_root = data_dir / "seed_designs"
    feedback_root = data_dir / "feedback_runs"
    if not seed_root.is_dir():
        raise FileNotFoundError(f"Seed directory not found: {seed_root}")
    if not feedback_root.is_dir():
        raise FileNotFoundError(f"Feedback directory not found: {feedback_root}")

    seed_dirs = sorted(
        (
            path
            for path in seed_root.iterdir()
            if path.is_dir() and _SEED_RE.fullmatch(path.name)
        ),
        key=lambda path: _numeric_path_key(path, _SEED_RE),
    )
    # Seed generations are independent samples from the same seed prompt.
    results = [
        _summarize_stage(
            "Seed",
            [_pool_from_sample_dirs("seed", seed_dirs)],
        )
    ]

    pools_by_iteration: Dict[int, List[SamplePool]] = {}
    trajectory_dirs = sorted(
        (
            path
            for path in feedback_root.iterdir()
            if path.is_dir() and _SEED_RE.fullmatch(path.name)
        ),
        key=lambda path: _numeric_path_key(path, _SEED_RE),
    )
    for trajectory_dir in trajectory_dirs:
        for samples_dir in trajectory_dir.iterdir():
            match = _ITER_SAMPLES_RE.fullmatch(samples_dir.name)
            if not samples_dir.is_dir() or not match:
                continue
            iteration = int(match.group(1))
            pool = _pool_from_sample_dirs(
                trajectory_dir.name,
                _effective_sample_dirs(samples_dir),
            )
            if pool.total:
                pools_by_iteration.setdefault(iteration, []).append(pool)

    for iteration in sorted(pools_by_iteration):
        results.append(
            _summarize_stage(
                f"Feedback {iteration}",
                pools_by_iteration[iteration],
            )
        )
    return results


def collect_domain_results(model_dir: Path) -> Dict[str, List[StageResult]]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    results = {}
    for domain_dir in sorted(model_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        if not (domain_dir / "seed_designs").is_dir():
            continue
        if not (domain_dir / "feedback_runs").is_dir():
            print(f"[WARN] Skipping domain without feedback runs: {domain_dir.name}")
            continue
        results[domain_dir.name] = collect_stage_results(domain_dir)

    if not results:
        raise ValueError(f"No domain data found under {model_dir}")
    return results


def collect_focus_schedule(model_dir: Path) -> List[str]:
    schedules = []
    for summary_file in sorted(model_dir.glob("*/domain_eval_data_summary.json")):
        try:
            summary = json.loads(summary_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        for seed_data in (summary.get("eval_data_by_seed") or {}).values():
            iterations = []
            for name, iteration_data in (seed_data.get("iters") or {}).items():
                match = re.fullmatch(r"iter_(\d+)", name)
                target = iteration_data.get("Target")
                if match and isinstance(target, str) and target:
                    iterations.append((int(match.group(1)), target))
            if iterations:
                schedules.append(tuple(target for _, target in sorted(iterations)))

    if not schedules:
        return []
    return list(Counter(schedules).most_common(1)[0][0])


def _focus_regions(schedule: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not schedule:
        return []

    regions = []
    start = 0
    target = schedule[0]
    for index, next_target in enumerate(schedule[1:], start=1):
        if next_target != target:
            regions.append((start, index - 1, target))
            start = index
            target = next_target
    regions.append((start, len(schedule) - 1, target))
    return regions


def write_results_csv(
    domain_results: Dict[str, List[StageResult]], output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "domain",
                "stage",
                "num_designs",
                "num_samples",
                "num_passed",
                "pass@1",
                "pass@8",
            ]
        )
        for domain, results in domain_results.items():
            for result in results:
                writer.writerow(
                    [
                        domain,
                        result.name,
                        result.pools,
                        result.samples,
                        result.passed,
                        f"{result.pass_at_k[1]:.6f}",
                        f"{result.pass_at_k[8]:.6f}",
                    ]
                )


def write_passing_percent_csv(
    domain_results: Dict[str, List[StageResult]], output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "domain",
                "stage",
                "num_designs",
                "num_samples",
                "num_passed",
                "average_passing_percent",
            ]
        )
        for domain, results in domain_results.items():
            for result in results:
                writer.writerow(
                    [
                        domain,
                        result.name,
                        result.pools,
                        result.samples,
                        result.passed,
                        f"{result.average_passing_percent:.6f}",
                    ]
                )


def _add_focus_indicators(
    ax: plt.Axes,
    x_values: Sequence[float],
    focus_schedule: Sequence[str],
) -> None:
    for start, end, target in _focus_regions(focus_schedule):
        # Feedback iteration i is plotted at x=i+1 because x=0 is the seed
        # stage. Align each span directly with its first and last stage centers.
        x_start = 1.0 + start
        x_end = 1.0 + end
        stage_spacing = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
        len_extra = 0.4 * stage_spacing
        indicator_start = max(x_start - len_extra, x_values[0])
        indicator_end = min(x_end + len_extra, x_values[-1])
        center = (indicator_start + indicator_end) / 2.0
        span_y = 108
        tick_half_height = 1.4
        ax.plot(
            [indicator_start, indicator_end],
            [span_y, span_y],
            color="0.45",
            linewidth=1.5,
            zorder=2,
        )
        ax.vlines(
            [indicator_start, indicator_end],
            span_y - tick_half_height,
            span_y + tick_half_height,
            color="0.45",
            linewidth=1.5,
            zorder=4,
        )
        ax.text(
            center,
            span_y,
            TARGET_LABELS.get(target, target.replace("_", " ").title()),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="0.25",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "0.45",
                "alpha": 1.0,
                "linewidth": 0.8,
            },
            zorder=10,
        )


def _stage_axis_data(
    domain_results: Dict[str, List[StageResult]],
) -> Tuple[List[str], List[int], List[str]]:
    stage_names = sorted(
        {result.name for results in domain_results.values() for result in results},
        key=lambda name: -1 if name == "Seed" else int(name.split()[-1]),
    )
    x_values = list(range(len(stage_names)))
    labels = [
        "Seed\nStage" if name == "Seed" else f"Feedback\nStage {name.split()[-1]}"
        for name in stage_names
    ]
    return stage_names, x_values, labels


def plot_domain_results(
    domain_results: Dict[str, List[StageResult]],
    output_stem: Path,
    focus_schedule: Sequence[str],
    title: str | None = None,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    stage_names, x_values, labels = _stage_axis_data(domain_results)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    domain_handles = []
    for domain, results in domain_results.items():
        color = DOMAIN_COLORS.get(domain, "#4C78A8")
        by_stage = {result.name: result for result in results}
        domain_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.5,
                label=DOMAIN_LABELS.get(domain, domain),
            )
        )
        for k, linestyle, marker, alpha, zorder in (
            (1, "--", "o", 0.45, 2),
            (8, "-", "s", 0.95, 4),
        ):
            ax.plot(
                x_values,
                [
                    by_stage[name].pass_at_k[k] if name in by_stage else math.nan
                    for name in stage_names
                ],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=3.5,
                color=color,
                alpha=alpha,
                zorder=zorder,
            )

    ax.set_xticks(x_values, labels)
    ax.set_xlim(-0.35, len(stage_names) - 0.65)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _add_focus_indicators(ax, x_values, focus_schedule)
    ax.set_xlabel("Design Generation / Mutation Stage", fontweight="bold")
    ax.set_ylabel("Avg. pass@k (%)", fontweight="bold")
    title_artist = fig.suptitle(
        title or DEFAULT_PLOT_TITLE,
        y=0.98,
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    domain_legend = fig.legend(
        handles=domain_handles,
        title="HLS Design Application Domain",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        # columnspacing=1.4,
        # handlelength=2.5,
        frameon=True,
        fontsize=9,
        title_fontsize=9,
    )
    domain_legend.get_title().set_fontweight("bold")
    style_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            marker="o",
            linewidth=1,
            markersize=2,
            label="pass@1",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="-",
            marker="s",
            linewidth=1,
            markersize=2,
            label="pass@8",
        ),
    ]
    metric_legend = ax.legend(
        handles=style_handles,
        title="Pass Rate",
        loc="lower right",
        frameon=True,
    )
    metric_legend.get_title().set_fontweight("bold")
    ax.text(
        0.5,
        0.03,
        "* Note: pass@1 is equivalent to the average pass rate *",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.2",
        fontstyle="italic",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    axes_center_x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    title_artist.set_x(axes_center_x)
    domain_legend.set_bbox_to_anchor(
        (axes_center_x, 0.94),
        transform=fig.transFigure,
    )

    for extension in ("png", "pdf"):
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_passing_percent_results(
    domain_results: Dict[str, List[StageResult]],
    output_stem: Path,
    focus_schedule: Sequence[str],
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    stage_names, x_values, labels = _stage_axis_data(domain_results)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    domain_handles = []
    for domain, results in domain_results.items():
        color = DOMAIN_COLORS.get(domain, "#4C78A8")
        by_stage = {result.name: result for result in results}
        domain_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.5,
                label=DOMAIN_LABELS.get(domain, domain),
            )
        )
        ax.plot(
            x_values,
            [
                (
                    by_stage[name].average_passing_percent
                    if name in by_stage
                    else math.nan
                )
                for name in stage_names
            ],
            linestyle="-",
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            alpha=0.9,
            zorder=3,
        )

    ax.set_xticks(x_values, labels)
    ax.set_xlim(-0.35, len(stage_names) - 0.65)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    _add_focus_indicators(ax, x_values, focus_schedule)
    ax.set_xlabel("Design Generation / Mutation Stage", fontweight="bold")
    ax.set_ylabel("Avg. Passing Designs (%)", fontweight="bold")
    title_artist = fig.suptitle(
        DEFAULT_PASSING_PERCENT_TITLE,
        y=0.98,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    domain_legend = fig.legend(
        handles=domain_handles,
        title="HLS Design Application Domain",
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        frameon=True,
        fontsize=9,
    )
    domain_legend.get_title().set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    axes_center_x = (ax.get_position().x0 + ax.get_position().x1) / 2.0
    title_artist.set_x(axes_center_x)
    domain_legend.set_bbox_to_anchor(
        (axes_center_x, 0.94),
        transform=fig.transFigure,
    )

    for extension in ("png", "pdf"):
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-domain average pass@k across generation stages."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model directory containing one seed/feedback directory per domain.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="Output path stem; .png, .pdf, and .csv are produced.",
    )
    parser.add_argument(
        "--passing-percent-output",
        type=Path,
        default=DEFAULT_PASSING_PERCENT_OUTPUT_STEM,
        help="Output stem for the average passing-percentage plot and CSV.",
    )
    parser.add_argument("--title", default=None, help="Optional figure title.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    domain_results = collect_domain_results(args.model_dir)
    focus_schedule = collect_focus_schedule(args.model_dir)
    write_results_csv(domain_results, args.output.with_suffix(".csv"))
    plot_domain_results(domain_results, args.output, focus_schedule, args.title)
    write_passing_percent_csv(
        domain_results,
        args.passing_percent_output.with_suffix(".csv"),
    )
    plot_passing_percent_results(
        domain_results,
        args.passing_percent_output,
        focus_schedule,
    )

    print("Domain          Stage        Designs  Passed/Samples  pass@1   pass@8")
    for domain, results in domain_results.items():
        for result in results:
            print(
                f"{domain:<15} {result.name:<12} {result.pools:>7}  "
                f"{result.passed:>4}/{result.samples:<7}  "
                f"{result.pass_at_k[1]:>6.2f}%  {result.pass_at_k[8]:>6.2f}%"
            )
    print(f"Wrote {args.output.with_suffix('.csv')}")
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.passing_percent_output.with_suffix('.csv')}")
    print(f"Wrote {args.passing_percent_output.with_suffix('.png')}")
    print(f"Wrote {args.passing_percent_output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
