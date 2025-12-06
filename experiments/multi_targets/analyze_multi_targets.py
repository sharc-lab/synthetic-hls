import json, math, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb
from collections import Counter

DATA_ROOT = Path(__file__).parent / "workspace_multi_targets"
RUN_NAMES: List[str] = [] # Specify run names like ["run__..."] or leave empty to scan all
OUT_DIR = Path(__file__).parent / "figs_multi_targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

_VER_TAIL = re.compile(r"(\d+)$")

def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _collect_runs(root: Path, names: List[str] | None):
    if names:
        return [root / n for n in names if (root / n).is_dir()]
    return sorted([p for p in root.glob("run__*") if p.is_dir()])

def _model_dirs(run_dir: Path) -> List[Path]:
    return [p for p in run_dir.iterdir() if p.is_dir()]

def _find_seed_jsons_with_model(run_dir: Path):
    """
    Layout: workspace/<run>/<model>/feedback_runs/seed_design__*/all_data_summary.json
    Returns [(seed_id, json_fp, model_name)].
    """
    out = []
    for md in _model_dirs(run_dir):
        fb = md / "feedback_runs"
        if not fb.exists():
            continue
        for sd in sorted(fb.glob("seed_design__*")):
            jfp = sd / "all_data_summary.json"
            if jfp.exists():
                out.append((f"{run_dir.name}/{md.name}/{sd.name}", jfp, md.name))
    return out

# ---------- plan & target_list helpers ----------
def _choose_iters(data: dict) -> dict:
    if "iters" in data and isinstance(data["iters"], dict):
        return data["iters"]
    samples = (data.get("samples") or {})
    if len(samples) == 1:
        return next(iter(samples.values()))
    return samples.get("sample_0", {})

def _expand_from_target_list(tl: dict) -> List[str]:
    plan = []
    for k, v in tl.items():
        try:
            c = int(v)
            if c > 0:
                plan.extend([k] * c)
        except Exception:
            continue
    return plan

def _infer_plan(data: dict) -> Optional[List[str]]:
    iters = _choose_iters(data)
    if iters:
        ks = sorted((int(k.split("_")[-1]), v)
                    for k, v in iters.items() if k.startswith("iter_"))
        seq = [v.get("Target") for _, v in ks if isinstance(v.get("Target"), str)]
        if seq:
            return seq
    tl = data.get("target_list")
    if isinstance(tl, dict) and tl:
        plan = _expand_from_target_list(tl)
        return plan if plan else None
    return None

def _targets_from_plan(plan_tuple: Tuple[str, ...]) -> List[str]:
    seen, out = set(), []
    for t in plan_tuple:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _get_target_list_dict(jfp: Path) -> Dict[str, int]:
    d = _read_json(jfp)
    tl = d.get("target_list") or {}
    out = {}
    if isinstance(tl, dict):
        for k, v in tl.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                pass
    return out

def _canonical_plan_key(tl: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Order-insensitive key for exact equality of target_list dict."""
    return tuple(sorted(tl.items(), key=lambda kv: kv[0]))

def _plan_key_to_string(plan_key: Tuple[Tuple[str, int], ...]) -> str:
    return ",".join([f"{k}:{v}" for k, v in plan_key])

def _union_target_keys_from_entries(entries):
    ks, seen = [], set()
    for _, jfp, _ in entries:
        d = _read_json(jfp)
        tl = d.get("target_list") or {}
        if isinstance(tl, dict):
            for k in tl.keys():
                if k not in seen:
                    seen.add(k); ks.append(k)
    return ks

def _ver_index(k: str) -> int:
    m = _VER_TAIL.search(k)
    return int(m.group(1)) if m else -1

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan

def _pick_val(it: dict, key: str, subkey=None):
    tr = (it.get("target_result") or {})
    if key == "pareto_scores":
        ps = (tr.get("pareto_scores") or {})
        val = ps.get(subkey, None) if subkey else None
        return 1.0 if val is None else val
    return tr.get(key)

def _series(seed_json: Path, key: str, subkey=None):
    d = _read_json(seed_json)
    base = (d.get("seed_design") or {}).get("target_results") or {}
    if key != "pareto_scores":
        y0 = _to_float(base.get(key))
    else:
        ps = (base.get("pareto_scores") or {})
        y0 = _to_float(ps.get(subkey, 1.0))
    iters = _choose_iters(d)
    ks = sorted((int(k.split("_")[-1]), v)
                for k, v in iters.items() if k.startswith("iter_"))
    y = [y0]
    for _, v in ks:
        y.append(_to_float(_pick_val(v, key, subkey)))
    x = list(range(len(y)))
    return x, y

def _avg_series(seed_json: Path):
    return _series(seed_json, "average_function_lines")

def _iter_targets(seed_json: Path) -> List[str]:
    d = _read_json(seed_json)
    return _infer_plan(d) or []

# ---------- color / style ----------
def _boost_color(c, sat=1.25, val=1.2):
    r, g, b = to_rgb(c)
    h, s, v = rgb_to_hsv(np.array([[r, g, b]]))[0]
    s = np.clip(s * sat, 0, 1)
    v = np.clip(v * val, 0, 1)
    return tuple(hsv_to_rgb([[h, s, v]])[0])

def _palette():
    cm = plt.colormaps.get_cmap("tab10")
    base = [tuple(map(float, cm(i)))[:3] for i in range(10)]
    return [_boost_color(c) for c in base]

_COLS = _palette()
_MARK = ['o','s','D','^','v','<','>','P','X','h','*','+','x','1','2','3','4']

def _style(i):
    return dict(
        color=_COLS[i % len(_COLS)],
        marker=_MARK[i % len(_MARK)],
        linewidth=2.2,
        markersize=5.0,
        alpha=0.9,
        path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
    )

def _z(series_len):
    return 2000 - series_len

def _active_idx(plan, t) -> List[int]:
    return [i for i, v in enumerate(plan) if v == t]

def _plot_high(ax, x, y, st, active: List[int], solid: bool = True, lw_base: float = 2.2):
    ls = "-" if solid else "--"
    ax.plot(
        x, y, ls, marker=st["marker"], color=st["color"],
        linewidth=st["linewidth"], markersize=st["markersize"],
        alpha=0.30, path_effects=st["path_effects"], zorder=_z(len(y))
    )
    for k in active:
        if k + 1 < len(x) and not (math.isnan(y[k]) or math.isnan(y[k + 1])):
            ax.plot(
                [x[k], x[k + 1]], [y[k], y[k + 1]],
                ls, color=st["color"], lw=lw_base + 0.8, alpha=0.95, zorder=_z(len(y)) + 50
            )

def _segments_only_series(x: List[int], y: List[float], active: List[int]) -> Tuple[List[int], List[float]]:
    y_mask = [math.nan] * len(y)
    for k in active:
        if 0 <= k < len(y) - 1:
            y_mask[k]   = y[k]
            y_mask[k+1] = y[k+1]
    return x, y_mask

def _plot_segments_only(ax, x, y, st, active: List[int], solid: bool = True, lw_base: float = 2.6):
    ls = "-" if solid else "--"
    xx, yy = _segments_only_series(x, y, active)
    ax.plot(
        xx, yy, ls, marker=st["marker"], color=st["color"],
        linewidth=lw_base, markersize=st["markersize"],
        alpha=0.95, path_effects=st["path_effects"], zorder=_z(len(yy)) + 50
    )

def _plot_group(model_name: str,
                plan_key: Tuple[Tuple[str, int], ...],
                entries: List[Tuple[str, Path, str]]):
    if not entries:
        return

    # Derive representative plan sequence (majority among seeds with non-empty plans)
    plans = []
    for _, jfp, _ in entries:
        d = _read_json(jfp)
        p = _infer_plan(d)
        if p:
            plans.append(tuple(p))
    if plans:
        c = Counter(plans)
        plan_tuple = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        # fallback: expand target_list counts from plan_key
        exp = []
        for k, v in plan_key:
            exp.extend([k] * int(v))
        plan_tuple = tuple(exp)

    run_targets = _targets_from_plan(plan_tuple)
    extras = [k for k in _union_target_keys_from_entries(entries) if k not in run_targets]
    run_targets += extras
    n_rows = len(run_targets)

    entries_sorted = sorted(entries, key=lambda t: t[0])
    seed_styles = {sid: _style(i) for i, (sid, _, __) in enumerate(entries_sorted)}
    seed_plans = {sid: _iter_targets(jfp) for sid, jfp, __ in entries_sorted}

    fig, axes = plt.subplots(n_rows, 2, figsize=(13.5, 3.6 * n_rows), squeeze=False)

    alias = {"num_functions": "num_funcs", "max_call_chain_depth": "max_depth",
             "average_function_lines": "avg_func_lines", "pareto_scores": "pareto_scores"}
    counts_line = ", ".join(f"{alias.get(k, k)}: {v}" for k, v in plan_key)
    safe_plan_str = _plan_key_to_string(plan_key).replace("/", "_").replace(" ", "")

    fig.suptitle(
        f"Model: {model_name}; Target list: {counts_line}",
        fontsize=13, y=0.97
    )

    for r, target in enumerate(run_targets):
        axL, axR = axes[r, 0], axes[r, 1]

        # Determine global N (max iterations) for this row
        N = 0
        for sid, jfp, _ in entries_sorted:
            plan = seed_plans[sid]
            if plan:
                N = max(N, len(plan))
            x_tmp, _y_tmp = _avg_series(jfp)
            N = max(N, len(x_tmp) - 1)

        # Plot each seed with 0..N axis & NaN padding
        for sid, jfp, _ in entries_sorted:
            st = seed_styles[sid]
            plan = seed_plans[sid]
            active = _active_idx(plan, target) if plan else []

            if target == "pareto_scores":
                x1, y1 = _series(jfp, "pareto_scores", "LUTs_vs_latency")
                x2, y2 = _series(jfp, "pareto_scores", "FFs_vs_latency")
                y1 = y1 + [math.nan] * max(0, (N + 1 - len(y1)))
                y2 = y2 + [math.nan] * max(0, (N + 1 - len(y2)))
                _plot_segments_only(axL, list(range(N + 1)), y1, st, active, solid=True)
                _plot_segments_only(axL, list(range(N + 1)), y2, st, active, solid=False)

                xA, yA = _avg_series(jfp)
                yA = yA + [math.nan] * max(0, (N + 1 - len(yA)))
                _plot_high(axR, list(range(N + 1)), yA, st, active, solid=False)

            elif target == "average_function_lines":
                xA, yA = _avg_series(jfp)
                yA = yA + [math.nan] * max(0, (N + 1 - len(yA)))
                _plot_high(axL, list(range(N + 1)), yA, st, active, solid=True)
                _plot_high(axR, list(range(N + 1)), yA, st, active, solid=False)

            else:
                xT, yT = _series(jfp, target)
                yT = yT + [math.nan] * max(0, (N + 1 - len(yT)))
                _plot_high(axL, list(range(N + 1)), yT, st, active, solid=True)

                xA, yA = _avg_series(jfp)
                yA = yA + [math.nan] * max(0, (N + 1 - len(yA)))
                _plot_high(axR, list(range(N + 1)), yA, st, active, solid=False)

        for ax in (axL, axR):
            ax.set_xlim(-0.2, N + 0.2)
            ax.set_xticks(list(range(N + 1)))
            ax.set_xticklabels([str(i) for i in range(N + 1)], fontsize=8)
            ax.set_xlabel("Iteration (0 = seed)")
            ax.grid(True, ls=":", lw=0.5)

        # Row titles/labels
        if target == "pareto_scores":
            axL.set_title("pareto_scores (LUTs solid, FFs dashed)", fontsize=11)
            axL.set_ylabel("pareto_scores (lower is better)")
        elif target == "average_function_lines":
            axL.set_title("average_function_lines (solid)", fontsize=11)
            axL.set_ylabel("average_function_lines")
        else:
            axL.set_title(f"{target} (solid)", fontsize=11)
            axL.set_ylabel(target)

    # Right-column legend note
    fig.text(0.75, 0.93, "average_function_lines (dashed)", fontsize=11, ha="center", va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_fp = OUT_DIR / f"{model_name}__{safe_plan_str}.png"
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)
    print("Saved:", out_fp)

def main():
    runs = _collect_runs(DATA_ROOT, RUN_NAMES)
    if not runs:
        print("No runs found.")
        return

    # Build groups across ALL runs:
    # key = (model_name, plan_key)   where plan_key is canonicalized target_list dict
    groups: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], List[Tuple[str, Path, str]]] = {}

    for rd in runs:
        for sid, jfp, model_name in _find_seed_jsons_with_model(rd):
            tl_dict = _get_target_list_dict(jfp)
            plan_key = _canonical_plan_key(tl_dict)
            groups.setdefault((model_name, plan_key), []).append((sid, jfp, model_name))

    if not groups:
        print("No valid groups found.")
        return

    # Plot one figure per (model_name, plan_key) group
    for (model_name, plan_key), entries in groups.items():
        if not entries:
            continue
        _plot_group(model_name, plan_key, entries)

if __name__ == "__main__":
    main()
