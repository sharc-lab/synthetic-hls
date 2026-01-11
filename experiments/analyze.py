import json, math, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

### User configs ###
# Root directory containing the run folders.
DATA_ROOT = Path(__file__).parent / "multi_targets" / "workspace_multi_targets"

# Specify which run(s) to visualize, e.g., ["run__2024-06-01_12-00-00", "run__2024-06-02_15-30-00"]
RUN_NAMES: List[str] = []

# Merge flag:
# - False: plot per run (output: FIGS_DIR/<run_name>/<domain>.png)
# - True : merge runs that share the same (domain, model_name, target_list) into one figure. 
#          This is for gathering results across multiple runs for the same setting.
#          (output: FIGS_DIR/merged/<domain>__<model>__<planhash>.png)
MERGE: bool = False

FIGS_DIR = Path(__file__).parent / "multi_targets" / "figs_dir"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.2,
    "legend.frameon": False,
})

_ITER_RE = re.compile(r"iter_(\d+)$")
_INT_RE = re.compile(r"(\d+)")
_MARK = ['o','s','D','^','v','<','>','P','X','h','*','+','x']
_CM = plt.colormaps.get_cmap("tab10")


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


def _plot_domain(d: dict, out_fp: Path, run_label: str):
    if not d:
        return

    model_name = d.get("model_name", "unknown_model")
    domain = d.get("domain", out_fp.stem)
    tl = _norm_target_list(d.get("target_list") or {})
    seeds = d.get("eval_data_by_seed") or {}
    if not isinstance(seeds, dict) or not seeds:
        print(f"[WARN] No seeds for domain={domain} => {out_fp}")
        return

    # Infer per-seed plan + take majority plan as representative row ordering
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

    # targets to plot = unique order in plan, plus any missing keys from tl
    seen, targets = set(), []
    for t in plan:
        if t not in seen:
            seen.add(t)
            targets.append(t)
    for t in tl.keys():
        if t not in seen:
            targets.append(t)

    seed_names = sorted(seeds.keys())

    # precompute per-seed plan and styles
    seed_plan = {sn: infer_plan(seeds[sn]) for sn in seed_names}
    seed_style = {}
    for i, sn in enumerate(seed_names):
        c = _CM(i % 10)
        seed_style[sn] = dict(
            color=c,
            marker=_MARK[i % len(_MARK)],
            ms=5.0,
            pe=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
        )

    def series(seed_entry: dict, key: str, subkey: Optional[str] = None) -> List[float]:
        def to_f(x):
            try:
                return float(x)
            except Exception:
                return math.nan

        seed0 = seed_entry.get("seed_design") or {}
        base_tr = (seed0.get("target_results") or {})

        if key == "pareto_scores":
            base_ps = base_tr.get("pareto_scores") or {}
            y0 = to_f(base_ps.get(subkey, 1.0))
        else:
            y0 = to_f(base_tr.get(key))

        # iterate in order
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
                y.append(to_f(ps.get(subkey, 1.0)))
            else:
                y.append(to_f(tr.get(key)))
        return y

    # unified plotter: base faint curve w/ markers + highlight segments w/ markers
    def plot_curve(ax, y: List[float], st, active: List[int], solid: bool, segments_only: bool):
        N = len(y)
        x = list(range(N))
        ls = "-" if solid else "--"

        if segments_only:
            yy = [math.nan] * N
            for k in active:
                if 0 <= k < N - 1:
                    yy[k] = y[k]
                    yy[k + 1] = y[k + 1]
            ax.plot(
                x, yy, ls, color=st["color"], lw=3.0, alpha=0.95,
                marker=st["marker"], markersize=st["ms"] + 0.5,
                markerfacecolor=st["color"], markeredgecolor=st["color"],
                path_effects=st["pe"],
            )
            return

        ax.plot(
            x, y, ls, color=st["color"], lw=2.2, alpha=0.35,
            marker=st["marker"], markersize=st["ms"],
            markerfacecolor=st["color"], markeredgecolor=st["color"],
            path_effects=st["pe"],
        )
        for k in active:
            if k + 1 < N and not (math.isnan(y[k]) or math.isnan(y[k + 1])):
                ax.plot(
                    [k, k + 1], [y[k], y[k + 1]], ls,
                    color=st["color"], lw=3.0, alpha=0.95,
                    marker=st["marker"], markersize=st["ms"] + 0.5,
                    markerfacecolor=st["color"], markeredgecolor=st["color"],
                    path_effects=st["pe"],
                )

    # max length for consistent x-axis
    N = 1
    for sn in seed_names:
        N = max(N, len(series(seeds[sn], "average_function_lines")))
        N = max(N, len(seed_plan[sn]) + 1)
    xs = list(range(N))

    n_rows = max(1, len(targets))
    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 3.6 * n_rows), squeeze=False)

    tl_line = ", ".join([f"{k}: {v}" for k, v in tl.items()]) if tl else "N/A"
    fig.suptitle(
        f"Run: {run_label} | Model: {model_name}\n"
        f"Domain: {domain} | Targets: {tl_line}",
        fontsize=12, y=0.975
    )

    for r, target in enumerate(targets):
        axL, axR = axes[r, 0], axes[r, 1]

        for sn in seed_names:
            st = seed_style[sn]
            plan_sn = seed_plan[sn]
            active = [i for i, v in enumerate(plan_sn) if v == target]
            se = seeds[sn]

            if target == "pareto_scores":
                y_lut = series(se, "pareto_scores", "LUTs_vs_latency")
                y_ff = series(se, "pareto_scores", "FFs_vs_latency")
                y_avg = series(se, "average_function_lines")
                y_lut += [math.nan] * (N - len(y_lut))
                y_ff += [math.nan] * (N - len(y_ff))
                y_avg += [math.nan] * (N - len(y_avg))

                plot_curve(axL, y_lut, st, active, solid=True,  segments_only=True)
                plot_curve(axL, y_ff,  st, active, solid=False, segments_only=True)
                plot_curve(axR, y_avg, st, active, solid=False, segments_only=False)
            else:
                y_t = series(se, target)
                y_avg = series(se, "average_function_lines")
                y_t += [math.nan] * (N - len(y_t))
                y_avg += [math.nan] * (N - len(y_avg))

                plot_curve(axL, y_t,   st, active, solid=True,  segments_only=False)
                plot_curve(axR, y_avg, st, active, solid=False, segments_only=False)

        for ax in (axL, axR):
            ax.set_xlim(-0.2, N - 1 + 0.2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(i) for i in xs], fontsize=8)
            ax.set_xlabel("Iteration (0 = seed)")

        if target == "pareto_scores":
            axL.set_title("pareto_scores (LUTs solid, FFs dashed)", fontsize=11)
            axL.set_ylabel("pareto_scores (lower is better)")
        else:
            axL.set_title(f"{target} (solid)", fontsize=11)
            axL.set_ylabel(target)

    fig.text(0.75, 0.93, "average_function_lines (dashed)", fontsize=11, ha="center", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp)
    plt.close(fig)
    print("Saved:", out_fp)


def main():
    run_dirs = _load_run_dirs()

    if not MERGE:
        for rd in run_dirs:
            out_dir = FIGS_DIR / rd.name
            for fp in sorted(rd.rglob("domain_eval_data_summary.json")):
                d = _read_json(fp)
                domain = d.get("domain", fp.parent.name)
                _plot_domain(d, out_fp=out_dir / f"{domain}.png", run_label=rd.name)
        return

    # MERGE: group by (domain, model_name, normalized target_list)
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
        label = run_names[0] if len(run_names) == 1 else ((" + ".join(run_names)) if len(run_names) <= 3 else f"{run_names[0]} + {len(run_names)-1} more")
        out_fp = merged_dir / f"{domain}__{model.replace('/','_')}__{sig}.png"
        _plot_domain(combined, out_fp=out_fp, run_label=label)


if __name__ == "__main__":
    main()