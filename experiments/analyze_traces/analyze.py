import fnmatch
import json
import math
import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


from matplotlib.ticker import MultipleLocator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap

### User configs ###
# Root directory containing the run folders.
DATA_ROOT = Path(__file__).parent.parent / "multi_targets" / "workspace_multi_targets"


# experiments/multi_targets/workspace_multi_targets/run__2026-03-31_02-07-58
# Specify which run(s) to visualize, e.g., ["run__2026-01-22_18-05-56", "run__2026-01-24_04-12-05"]
RUN_NAMES: List[str] = [
    "run__2026-03-31_02-07-58"
]

### Merge flag:
# - False: plot per run
# - True : merge runs that share the same setting into one figure
MERGE: bool = False

### Separate in domains flag:
# - True : keep domains separate
# - False: merge all domains together into one figure
SEPARATE_IN_DOMAINS: bool = False

FIGS_DIR = Path(__file__).parent / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_LAST_ITER = 9
DISPLAY_XS = list(range(DISPLAY_LAST_ITER + 1))

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.titlesize": 13,
    # "axes.spines.top": False,
    # "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.50,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

_ITER_RE = re.compile(r"iter_(\d+)$")
_INT_RE = re.compile(r"(\d+)")
_MARK = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', 'h', '*', '+', 'x']
_CM = plt.colormaps.get_cmap("tab10")

TARGET_BASE_COLORS = {
    "num_functions":           "#E63946",   # red
    "max_call_chain_depth":    "#F4A261",   # teal
    "average_function_lines":  "#2A9D8F",   # orange
    "pareto_scores":           "#8BCAFF",   # purple
}
_FALLBACK_COLOR = "#8034C2"                 # blue


def _make_shades(hex_color: str, n: int):
    """Return *n* RGBA colours from a light tint to the full base colour."""
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(0.82 + 0.18 * c for c in rgb)
    cmap = LinearSegmentedColormap.from_list("sh", [light, rgb], N=max(n, 2))
    return [cmap(i / max(n - 1, 1)) for i in range(n)]

DISPLAY_LABELS = {
    "num_functions": "# of Func.",
    "max_call_chain_depth": "Max\nCall Depth",
    "average_function_lines": "Avg.\nFunc. LoC",
    "pareto_scores": "Pareto Score ",
    "LUTs_vs_latency": "LUT vs. Latency",
    "FFs_vs_latency": "FF vs. Latency",
}

SCHEDULE_LABELS = {
    "num_functions": "# of Func.",
    "max_call_chain_depth": "Max Call Depth",
    "average_function_lines": "Avg. Func. LoC",
    "pareto_scores": "Pareto Score",
}

DOMAIN_LABELS = {
    "ml_ai": "Machine Learning / AI",
    "sci_sim": "Scientific Simulation",
    "fin_model": "Financial Modeling",
    "eng_sim": "Engineering Simulation",
    "data_big": "Data Analytics / Big Data",
    "gfx_render": "Graphics Rendering",
    "crypto_bc": "Cryptography / Blockchain",
    "telecom_sp": "Telecommunications / Signal Processing",
    "astro": "Astronomy / Astrophysics",
    "health_med": "Healthcare / Medical Imaging",
}


def _pretty_label(key: str) -> str:
    return DISPLAY_LABELS.get(key, key.replace("_", " ").title())


def _pretty_domain(domain: str) -> str:
    return DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())


def _short_model_name(model_name: str, max_len: int = 42) -> str:
    m = str(model_name).replace("openrouter/", "")
    return (m[:max_len - 3] + "...") if len(m) > max_len else m


def _format_opt_schedule(plan: List[str]) -> str:
    if not plan:
        return "Opt. Schedule:\nN/A"

    groups = []
    cur = plan[0]
    count = 1
    for t in plan[1:]:
        if t == cur:
            count += 1
        else:
            groups.append((cur, count))
            cur = t
            count = 1
    groups.append((cur, count))

    parts = [f"({SCHEDULE_LABELS.get(t, t)}) x{n}" for t, n in groups]
    return "Opt. Schedule:\n" + " ➜ ".join(parts)


def _phase_boundaries_for_target(plan: List[str], target: str) -> List[int]:
    active = [i for i, t in enumerate(plan) if t == target]
    if not active:
        return []

    groups = []
    start = active[0]
    prev = active[0]
    for idx in active[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev + 1))
            start = idx
            prev = idx
    groups.append((start, prev + 1))

    bounds = []
    for a, b in groups:
        bounds.extend([a, b])
    return sorted(set(bounds))


def _phase_regions_for_target(plan: List[str], target: str) -> List[Tuple[int, int]]:
    """Return (xmin, xmax) pairs for contiguous runs where plan[i] == target."""
    active = [i for i, t in enumerate(plan) if t == target]
    if not active:
        return []
    groups = []
    start = active[0]
    prev = active[0]
    for idx in active[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev + 1))
            start = idx
            prev = idx
    groups.append((start, prev + 1))
    return groups


def _load_run_dirs() -> List[Path]:
    if not RUN_NAMES:
        runs = sorted([p for p in DATA_ROOT.glob("run__*") if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f"No run__* dirs under {DATA_ROOT}")
        return [runs[-1]]
    out = []
    for n in RUN_NAMES:
        rd = DATA_ROOT / n
        if not rd.is_dir():
            raise FileNotFoundError(f"Run dir not found: {rd}")
        out.append(rd)
    return out


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _norm_target_list(tl_raw) -> Dict[str, int]:
    if not isinstance(tl_raw, dict):
        return {}
    out = {}
    for k, v in tl_raw.items():
        if isinstance(v, int):
            out[str(k)] = v
        elif isinstance(v, str):
            m = _INT_RE.search(v)
            out[str(k)] = int(m.group(1)) if m else 0
        else:
            out[str(k)] = 0
    return out


def _plot_group(d: dict, out_fp: Path, run_label: str):
    if not d:
        return

    model_name = d.get("model_name", "unknown_model")
    domain = d.get("domain", None)
    tl = _norm_target_list(d.get("target_list") or {})
    seeds = d.get("eval_data_by_seed") or {}
    if not isinstance(seeds, dict) or not seeds:
        print(f"[WARN] No seeds => {out_fp}")
        return

    def infer_plan(seed_entry: dict) -> List[str]:
        iters = seed_entry.get("iters") or {}
        items = []
        for k, v in iters.items():
            if isinstance(k, str):
                m = _ITER_RE.match(k)
                if m:
                    items.append((int(m.group(1)), v))
        items.sort(key=lambda t: t[0])
        seq = []
        for _, it in items:
            t = it.get("Target")
            if isinstance(t, str) and t:
                seq.append(t)
        return seq

    plans = [tuple(infer_plan(se)) for se in seeds.values() if infer_plan(se)]
    plan = list(Counter(plans).most_common(1)[0][0]) if plans else []

    seen, targets = set(), []
    for t in plan:
        if t not in seen:
            seen.add(t)
            targets.append(t)
    for t in tl.keys():
        if t not in seen:
            targets.append(t)

    seed_names = sorted(seeds.keys())
    seed_plan = {sn: infer_plan(seeds[sn]) for sn in seed_names}

    _GREY = "0.72"
    _GREY_LW = 1.0
    _GREY_ALPHA = 0.40
    _ACTIVE_LW = 1.6
    _ACTIVE_ALPHA = 0.55
    _AVG_LW = 2.4

    def series(seed_entry: dict, key: str, subkey: Optional[str] = None) -> List[float]:
        def to_f(x):
            try:
                return float(x)
            except Exception:
                return math.nan

        seed0 = seed_entry.get("seed_design") or {}
        base_tr = seed0.get("target_results") or {}

        if key == "pareto_scores":
            base_ps = base_tr.get("pareto_scores") or {}
            y0 = to_f(base_ps.get(subkey, None))
        else:
            y0 = to_f(base_tr.get(key))

        iters = seed_entry.get("iters") or {}
        items = []
        for k, v in iters.items():
            if isinstance(k, str):
                m = _ITER_RE.match(k)
                if m:
                    items.append((int(m.group(1)), v))
        items.sort(key=lambda t: t[0])

        y = [y0]
        for _, it in items:
            tr = it.get("target_result") or {}
            if key == "pareto_scores":
                ps = tr.get("pareto_scores") or {}
                y.append(to_f(ps.get(subkey, None)))
            else:
                y.append(to_f(tr.get(key)))
        return y

    def fit_to_display(y: List[float]) -> List[float]:
        y = list(y[:DISPLAY_LAST_ITER + 1])
        if len(y) < DISPLAY_LAST_ITER + 1:
            y += [math.nan] * ((DISPLAY_LAST_ITER + 1) - len(y))
        return y

    def plot_grey(ax, y: List[float], ls="-"):
        ax.plot(
            DISPLAY_XS, y, ls,
            color=_GREY, lw=_GREY_LW, alpha=_GREY_ALPHA, zorder=2, 
    
        )

    def plot_active_segments(ax, y: List[float], active: List[int],
                             color, ls="-"):
        x = DISPLAY_XS
        for k in active:
            if k + 1 < len(x) and not (math.isnan(y[k]) or math.isnan(y[k + 1])):
                ax.plot(
                    [x[k], x[k + 1]], [y[k], y[k + 1]], ls,
                    color=color, lw=_ACTIVE_LW, alpha=_ACTIVE_ALPHA, zorder=3,
                )

    def plot_avg(ax, y: List[float], ls="-", label=None):
        ax.plot(
            DISPLAY_XS, y, ls,
            color="black", lw=_AVG_LW, zorder=4,
            marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1,
            # path_effects=[pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()],
            label=label,
        
        )

    n_rows = max(1, len(targets))

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(5.5, 1.2 * n_rows + 0.9),
        squeeze=False
    )

    short_model = _short_model_name(model_name)

    fig.suptitle(
        "Metric-Guided HLS Design Mutation Trajectories",
        y=0.965,
        fontweight='bold'
    )

    if SEPARATE_IN_DOMAINS:
        pretty_domain = _pretty_domain(domain if domain is not None else "unknown_domain")
        subtitle = f"{pretty_domain} | {short_model}"
    else:
        subtitle = f"{short_model}"

    # fig.text(
    #     0.5, 0.935,
    #     subtitle,
    #     ha="center", va="top", fontsize=10
    # )

    fig.text(
        0.5, 0.93,
        _format_opt_schedule(plan),
        ha="center", va="top", fontsize=10
    )

    for r, target in enumerate(targets):
        ax = axes[r, 0]

        for xline in _phase_boundaries_for_target(plan, target):
            if 0 <= xline <= DISPLAY_LAST_ITER:
                ax.axvline(
                    x=xline,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.2,
                    zorder=1,
                )

        FACE_ALPHA = 0.06
        EDGE_ALPHA = 0.3
        region_hex = TARGET_BASE_COLORS.get(target, _FALLBACK_COLOR)
        region_rgb = mcolors.to_rgb(region_hex)
        for xmin, xmax in _phase_regions_for_target(plan, target):
            ax.axvspan(
                xmin, xmax,
                facecolor=(*region_rgb, FACE_ALPHA),
                edgecolor=(*region_rgb, EDGE_ALPHA),
                linewidth=1,
                zorder=-5,
            )

        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for xmin, xmax in _phase_regions_for_target(plan, target):
            color = TARGET_BASE_COLORS.get(target, _FALLBACK_COLOR)
            # make this color a darker shade of the base color
            color_rgba = mcolors.to_rgba(color, 0.5)
            color_dark = (color_rgba[0] * 0.6, color_rgba[1] * 0.6, color_rgba[2] * 0.6, color_rgba[3])

            
            cx = (xmin + xmax) / 2.0
            label = SCHEDULE_LABELS.get(target, target)
            ax.text(
                cx, 0.85, label,
                ha="center", va="center",
                fontsize=7.25, fontweight="bold",
                transform=trans,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor=color_dark,
                    alpha=1.0,
                    linewidth=0.8,
                ),
                zorder=10,
            )

        all_ys = []
        all_ys_lut = []
        all_ys_ff = []

        base_hex = TARGET_BASE_COLORS.get(target, _FALLBACK_COLOR)
        shades = _make_shades(base_hex, len(seed_names))

        for i_seed, sn in enumerate(seed_names):
            se = seeds[sn]
            color = shades[i_seed]
            plan_sn = seed_plan[sn]
            active = [j for j, v in enumerate(plan_sn) if v == target]

            if target == "pareto_scores":
                y_lut = fit_to_display(series(se, "pareto_scores", "LUTs_vs_latency"))
                y_ff = fit_to_display(series(se, "pareto_scores", "FFs_vs_latency"))
                plot_grey(ax, y_lut, ls="-")
                plot_grey(ax, y_ff, ls="--")
                plot_active_segments(ax, y_lut, active, color, ls="-")
                plot_active_segments(ax, y_ff, active, color, ls="--")
                all_ys_lut.append(y_lut)
                all_ys_ff.append(y_ff)
            else:
                y_t = fit_to_display(series(se, target))
                plot_grey(ax, y_t)
                plot_active_segments(ax, y_t, active, color)
                all_ys.append(y_t)

        if target == "pareto_scores":
            if all_ys_lut:
                avg_lut = np.nanmean(all_ys_lut, axis=0).tolist()
                plot_avg(ax, avg_lut, ls="-", label="Avg LUT")
            if all_ys_ff:
                avg_ff = np.nanmean(all_ys_ff, axis=0).tolist()
                plot_avg(ax, avg_ff, ls="--", label="Avg FF")
        else:
            if all_ys:
                avg_y = np.nanmean(all_ys, axis=0).tolist()
                plot_avg(ax, avg_y, label="Average")

        ax.set_xlim(0, DISPLAY_LAST_ITER)
        ax.set_xticks(DISPLAY_XS)
        ax.set_xticklabels([str(i) for i in DISPLAY_XS])
        ax.set_xlabel("Design Optimization Iteration", fontweight='bold')
        # ax.tick_params(axis="both", which="major", length=3.2, width=0.7)
        # make the ticks also go into the plot
        ax.tick_params(axis="x", which="both", length=5, width=1.0, direction="inout")

        ax.set_ylim(0, None)
        # y_lim based on target
        if target == "pareto_scores":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(0, math.ceil(max([max(y) for y in all_ys])) + 2)
        
        if target == "max_call_chain_depth":
            # make sure the tick increemtns are steps of 2
            # dont explicitly set the ticks, but make the grid ticks are steps of 2
            ax.yaxis.set_major_locator(MultipleLocator(2))

        if target == "pareto_scores":
            # make the grid ticks are steps of 0.1
            ax.yaxis.set_major_locator(MultipleLocator(0.2))

        if target == "average_function_lines":
            ax.yaxis.set_major_locator(MultipleLocator(10))

        # remove x ticks except for the last one absed on r
        if r < len(targets) - 1:
            # ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel(None)
            # ax.tick_params(axis="x", which="both", length=0)

        if target == "pareto_scores":
            # ax.set_title("Pareto Score (LUT Solid, FF Dashed)")
            ax.set_ylabel("Pareto Score", fontweight='bold', fontsize=10)
        else:
            # ax.set_title(_pretty_label(target))
            ax.set_ylabel(_pretty_label(target), fontweight='bold')
        
        
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.label.set_verticalalignment('center')

        if target == "pareto_scores":
            bounds = _phase_boundaries_for_target(plan, target)
            start_x = bounds[0] if bounds else DISPLAY_LAST_ITER
            if start_x > 0:
                ymin, ymax = ax.get_ylim()
                y_text = ymin + 0.3 * (ymax - ymin)
                ax.text(
                    max(1.6, start_x / 2.0),
                    y_text,
                    "Pareto Scoring Is Performed Only During\nPareto-Score Optimization Phases\nDue to Runtime Cost",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.72,
                    ),
                    zorder=2,
                )
                y_text_2 = ymin + 0.7 * (ymax - ymin)
                ax.text(max(1.6, start_x / 2.0), y_text_2, "Low is Better",
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.72,
                        ),
                        zorder=2,
                    )

    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.0)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, bbox_inches="tight", pad_inches=0.02, dpi=300)

    plt.close(fig)
    print("Saved:", out_fp)

def fast_rglob(directory, pattern) -> List[Path]:
    stuff: List[Path] = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            stuff.append(Path(os.path.join(root, filename)))
    return stuff


def main():
    run_dirs = _load_run_dirs()

    if not MERGE:
        if SEPARATE_IN_DOMAINS:
            for rd in run_dirs:
                out_dir = FIGS_DIR / rd.name
                for fp in sorted(rd.rglob("domain_eval_data_summary.json")):
                    d = _read_json(fp)
                    domain = d.get("domain", fp.parent.name)
                    _plot_group(d, out_fp=out_dir / f"{domain}.png", run_label=rd.name)
        else:
            for rd in run_dirs:
                out_dir = FIGS_DIR / rd.name / "all_domains"
                grouped: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], dict] = {}

                # for fp in sorted(rd.rglob("domain_eval_data_summary.json")):
                for fp in fast_rglob(rd, "domain_eval_data_summary.json"):
                    d = _read_json(fp)
                    if not d:
                        continue
                    model = str(d.get("model_name", "unknown_model"))
                    tl = _norm_target_list(d.get("target_list") or {})
                    key = (model, tuple(sorted(tl.items())))

                    if key not in grouped:
                        grouped[key] = {
                            "model_name": model,
                            "domain": None,
                            "target_list": tl,
                            "eval_data_by_seed": {}
                        }

                    domain_name = str(d.get("domain", fp.parent.name))
                    for sn, se in (d.get("eval_data_by_seed") or {}).items():
                        grouped[key]["eval_data_by_seed"][f"{domain_name}/{sn}"] = se

                for (model, tl_key), combined in grouped.items():
                    s = ",".join([f"{k}:{v}" for k, v in sorted(dict(tl_key).items())])
                    sig = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
                    # out_fp = out_dir / f"all_domains__{model.replace('/','_')}__{sig}.png"
                    out_fp = out_dir / "all_domains__plot_trj.png"
                    _plot_group(combined, out_fp=out_fp, run_label=rd.name)
        return

    # MERGE mode
    if SEPARATE_IN_DOMAINS:
        groups: Dict[Tuple[str, str, Tuple[Tuple[str, int], ...]], List[Tuple[str, dict]]] = {}
        for rd in run_dirs:
            for fp in sorted(rd.rglob("domain_eval_data_summary.json")):
                d = _read_json(fp)
                if not d:
                    continue
                domain = str(d.get("domain", fp.parent.name))
                model = str(d.get("model_name", "unknown_model"))
                tl = _norm_target_list(d.get("target_list") or {})
                key = (domain, model, tuple(sorted(tl.items())))
                groups.setdefault(key, []).append((rd.name, d))

        merged_dir = FIGS_DIR / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        for (domain, model, tl_key), entries in groups.items():
            seeds: Dict[str, dict] = {}
            run_names = []
            for rn, d in entries:
                run_names.append(rn)
                for sn, se in (d.get("eval_data_by_seed") or {}).items():
                    seeds[f"{rn}/{sn}"] = se
            if not seeds:
                continue

            tl = dict(tl_key)
            combined = {
                "model_name": model,
                "domain": domain,
                "target_list": tl,
                "eval_data_by_seed": seeds,
            }

            s = ",".join([f"{k}:{tl[k]}" for k in sorted(tl.keys())])
            sig = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            run_names = sorted(set(run_names))
            label = (
                run_names[0]
                if len(run_names) == 1
                else ((" + ".join(run_names)) if len(run_names) <= 3 else f"{run_names[0]} + {len(run_names)-1} more")
            )
            # out_fp = merged_dir / f"{domain}__{model.replace('/','_')}__{sig}.png"
            out_fp = merged_dir / "merged__plot_trj.png"
            _plot_group(combined, out_fp=out_fp, run_label=label)
    else:
        groups: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], List[Tuple[str, str, dict]]] = {}
        for rd in run_dirs:
            for fp in sorted(rd.rglob("domain_eval_data_summary.json")):
                d = _read_json(fp)
                if not d:
                    continue
                domain = str(d.get("domain", fp.parent.name))
                model = str(d.get("model_name", "unknown_model"))
                tl = _norm_target_list(d.get("target_list") or {})
                key = (model, tuple(sorted(tl.items())))
                groups.setdefault(key, []).append((rd.name, domain, d))

        merged_dir = FIGS_DIR / "merged_all_domains"
        merged_dir.mkdir(parents=True, exist_ok=True)

        for (model, tl_key), entries in groups.items():
            seeds: Dict[str, dict] = {}
            run_names = []
            for rn, domain, d in entries:
                run_names.append(rn)
                for sn, se in (d.get("eval_data_by_seed") or {}).items():
                    seeds[f"{rn}/{domain}/{sn}"] = se
            if not seeds:
                continue

            tl = dict(tl_key)
            combined = {
                "model_name": model,
                "domain": None,
                "target_list": tl,
                "eval_data_by_seed": seeds,
            }

            s = ",".join([f"{k}:{tl[k]}" for k in sorted(tl.keys())])
            sig = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            run_names = sorted(set(run_names))
            label = (
                run_names[0]
                if len(run_names) == 1
                else ((" + ".join(run_names)) if len(run_names) <= 3 else f"{run_names[0]} + {len(run_names)-1} more")
            )
            out_fp = merged_dir / f"all_domains__{model.replace('/','_')}__{sig}.png"
            _plot_group(combined, out_fp=out_fp, run_label=label)


if __name__ == "__main__":
    main()