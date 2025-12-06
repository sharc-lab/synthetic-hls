import json, math, re
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb
import numpy as np

DATA_ROOT = Path(__file__).parent / "workspace_single_target"
RUN_NAMES: List[str] = []  # Specify run names like ["run__..."] or leave empty to scan all
TARGETS = ["pareto_scores"]
OUT_DIR = Path(__file__).parent / "figs_pareto_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, only plot designs that completed all iterations in target_list
ONLY_COMPLETED = True

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

def is_number(x):
    try:
        return isinstance(x, (int, float)) or (isinstance(x, str) and x.lower() != "n/a" and float(x) is not None)
    except Exception:
        return False

def to_float(x):
    return float(x) if is_number(x) else math.nan

def _ver_index(k: str) -> int:
    m = _VER_TAIL.search(k)
    return int(m.group(1)) if m else -1

# ---------- folder parsing ----------
def _collect_runs(root: Path, run_names: list[str] | None):
    if run_names:
        runs = [root / rn for rn in run_names if (root / rn).is_dir()]
    else:
        runs = sorted([p for p in root.glob("run__*") if p.is_dir()])
    return runs

def _collect_model_dirs(run_dir: Path) -> List[Path]:
    """Return model subdirectories under a run (e.g. gpt-oss-120b, gpt-4o)."""
    return [p for p in run_dir.iterdir() if p.is_dir()]

def _find_seed_jsons(model_dir: Path):
    fb = model_dir / "feedback_runs"
    if not fb.exists():
        return []
    out = []
    for sd in sorted(fb.glob("seed_design__*")):
        jfp = sd / "all_data_summary.json"
        if jfp.exists():
            seed_id = f"{model_dir.parent.name}/{model_dir.name}/{sd.name}"
            out.append((seed_id, jfp))
    return out

def _get_target_list_dict(jfp: Path) -> Dict[str, int]:
    d = _read_json(jfp)
    tl = d.get("target_list") or {}
    if isinstance(tl, dict):
        out = {}
        for k, v in tl.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out
    return {}

def _canonical_plan_key(tl: Dict[str, int]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(tl.items(), key=lambda kv: kv[0]))

def _plan_key_to_string(plan_key: Tuple[Tuple[str, int], ...]) -> str:
    return ",".join([f"{k}:{v}" for k, v in plan_key])

# ---------- iteration helpers ----------
def _choose_iters(data: dict) -> dict:
    if "iters" in data and isinstance(data["iters"], dict):
        return data["iters"]
    samples = (data.get("samples") or {})
    if len(samples) == 1:
        return next(iter(samples.values()))
    return samples.get("sample_0", {})

def _max_iter_index_present(seed_json_fp: Path) -> int:
    """Return maximum k such that iter_k exists; -1 if none."""
    d = _read_json(seed_json_fp)
    iters = _choose_iters(d)
    max_k = -1
    for k in iters.keys():
        if k.startswith("iter_"):
            try:
                idx = int(k.split("_")[-1])
                max_k = max(max_k, idx)
            except Exception:
                pass
    return max_k

def _expected_total_iters_from_plan_key(plan_key: Tuple[Tuple[str, int], ...]) -> int:
    """Total number of iterations expected from target_list counts."""
    return sum(v for _, v in plan_key)

def _pick_from_iter_dict(iter_dict: dict, target_key: str, subkey: str | None = None):
    def extract_val(tr: dict):
        if target_key == "pareto_scores":
            ps = (tr.get("pareto_scores") or {})
            val = ps.get(subkey, None) if subkey else None
            return 1.0 if val is None else val
        return tr.get(target_key, "N/A")

    ver_keys = [k for k in iter_dict.keys() if k.startswith("ver_")]
    if ver_keys:
        ver_items = sorted(((_ver_index(k), iter_dict[k]) for k in ver_keys), key=lambda t: t[0])
        for _, v in ver_items:
            tr = (v.get("target_result") or {})
            val = extract_val(tr)
            if is_number(val):
                return to_float(val)
        return math.nan

    tr = (iter_dict.get("target_result") or {})
    val = extract_val(tr)
    return to_float(val) if is_number(val) else math.nan

def _series_from_summary(seed_json_fp: Path, target_key: str, subkey: str | None = None):
    data = _read_json(seed_json_fp)
    seed_block = (data.get("seed_design") or {}).get("target_results") or {}

    if target_key == "pareto_scores":
        ps = (seed_block.get("pareto_scores") or {})
        seed_val = ps.get(subkey, None) if subkey else None
        base_target = 1.0 if seed_val is None else to_float(seed_val)
    else:
        base_target = to_float(seed_block.get(target_key, None))

    iters_payload = _choose_iters(data)
    iter_items = sorted(
        ((int(k.split("_")[-1]), v) for k, v in iters_payload.items() if k.startswith("iter_")),
        key=lambda t: t[0]
    )

    y_t = [base_target]
    for _, iter_dict in iter_items:
        y_t.append(_pick_from_iter_dict(iter_dict, target_key, subkey=subkey))
    x = list(range(len(y_t)))
    return x, y_t

def _avg_series_from_summary(seed_json_fp: Path):
    return _series_from_summary(seed_json_fp, "average_function_lines")

# ---------- color & style ----------
def _boost_color(c, sat=1.3, val=1.25):
    r, g, b = to_rgb(c)
    hsv = rgb_to_hsv(np.array([[r, g, b]]))[0]
    h, s, v = hsv
    s = np.clip(s * sat, 0, 1)
    v = np.clip(v * val, 0, 1)
    return tuple(hsv_to_rgb([[h, s, v]])[0])

def _fixed_palette():
    cmaps = ["tab10", "Dark2", "Set1", "Paired"]
    colors = []
    for name in cmaps:
        cmap = plt.colormaps.get_cmap(name)
        if hasattr(cmap, "colors") and cmap.colors is not None:
            colors += [tuple(map(float, c))[:3] for c in cmap.colors]
        else:
            colors += [tuple(map(float, cmap(i / 9)))[:3] for i in range(10)]
    return [_boost_color(c) for c in colors]

_FIXED_COLOR_ORDER = _fixed_palette()
_MARKERS = ['o','s','D','^','v','<','>','P','X','h','*','+','x','1','2','3','4']

def _make_styles_for_seeds(ordered_seed_ids: list[str]):
    styles = {}
    for i, seed_id in enumerate(ordered_seed_ids):
        color = _FIXED_COLOR_ORDER[i % len(_FIXED_COLOR_ORDER)]
        marker = _MARKERS[i % len(_MARKERS)]
        styles[seed_id] = {
            "color": color,
            "marker": marker,
            "linewidth": 2.2,
            "markersize": 5.2,
            "alpha": 0.98,
            "path_effects": [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()],
            "zorder": 3 + (i % 6)
        }
    return styles

def _zorder_for_length(series_len, base=2000):
    return base - series_len

# ---------- legend helpers ----------
def _legend_single_avg(ax):
    handles = [plt.Line2D([0], [0], marker="o", linestyle="-", color="black")]
    labels = ["Solid: average_function_lines"]
    ax.legend(handles, labels, loc="lower right", ncol=1, fontsize=9)

def _legend_two_panel(fig, target_key: str):
    if target_key == "pareto_scores":
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="-",  color="black"),
            plt.Line2D([0], [0], marker="o", linestyle="--", color="black"),
        ]
        labels = ["Solid: LUTs_vs_latency (left)", "Dashed: FFs_vs_latency (left) / average_function_lines (right)"]
    else:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="-",  color="black"),
            plt.Line2D([0], [0], marker="o", linestyle="--", color="black"),
        ]
        labels = ["Solid: target (left)", "Dashed: average_function_lines (right)"]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)

# ---------- plotting ----------
def _plot_target_group(model_name: str, plan_key: Tuple[Tuple[str,int], ...], entries, target_key: str):
    """One figure for (model, target_list==plan_key, target_key)."""
    if not entries:
        return None

    if ONLY_COMPLETED:
        expected_N = _expected_total_iters_from_plan_key(plan_key)
        entries = [
            (sid, jfp) for (sid, jfp) in entries
            if _max_iter_index_present(jfp) >= (expected_N - 1 if expected_N > 0 else -1)
        ]
        if not entries:
            return None

    ordered_seed_ids = [sid for sid, _ in entries]
    seed_styles = _make_styles_for_seeds(ordered_seed_ids)

    # Single subplot for average_function_lines
    if target_key == "average_function_lines":
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for seed_id, jfp in entries:
            st = seed_styles[seed_id]
            x_a, y_a = _avg_series_from_summary(jfp)
            z_a = _zorder_for_length(len(y_a))
            ax.plot(x_a, y_a, "-", marker=st["marker"], color=st["color"],
                    linewidth=st["linewidth"], markersize=st["markersize"],
                    alpha=st["alpha"], path_effects=st["path_effects"], zorder=z_a)

        ax.set_title("average_function_lines", fontsize=11)
        ax.set_xlabel("Iteration (0 = seed)")
        ax.set_ylabel("average_function_lines")
        ax.grid(True, linestyle=":", linewidth=0.5)
        _legend_single_avg(ax)

        plan_str = _plan_key_to_string(plan_key) if plan_key else "plan:NA"
        fig.suptitle(f"Model: {model_name} — Target list: {plan_str}", fontsize=13, y=0.97)
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
        return fig

    # Two subplots for other targets
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.8, 5.4))
    for seed_id, jfp in entries:
        st = seed_styles[seed_id]
        if target_key == "pareto_scores":
            x_lut, y_lut = _series_from_summary(jfp, "pareto_scores", "LUTs_vs_latency")
            x_ff,  y_ff  = _series_from_summary(jfp, "pareto_scores", "FFs_vs_latency")
            max_len = max(len(x_lut), len(x_ff))
            x = list(range(max_len))
            y_lut += [math.nan] * (max_len - len(y_lut))
            y_ff  += [math.nan] * (max_len - len(y_ff))
            z = _zorder_for_length(max_len)

            ax_left.plot(x, y_lut, "-", marker=st["marker"], color=st["color"],
                         linewidth=st["linewidth"], markersize=st["markersize"],
                         alpha=st["alpha"], path_effects=st["path_effects"], zorder=z)
            ax_left.plot(x, y_ff, "--", marker=st["marker"], color=st["color"],
                         linewidth=st["linewidth"], markersize=st["markersize"],
                         alpha=st["alpha"], path_effects=st["path_effects"], zorder=z)

            x_a, y_a = _avg_series_from_summary(jfp)
            z_a = _zorder_for_length(len(y_a))
            ax_right.plot(x_a, y_a, "--", marker=st["marker"], color=st["color"],
                          linewidth=st["linewidth"], markersize=st["markersize"],
                          alpha=st["alpha"], path_effects=st["path_effects"], zorder=z_a)
        else:
            x_t, y_t = _series_from_summary(jfp, target_key)
            z_t = _zorder_for_length(len(y_t))
            ax_left.plot(x_t, y_t, "-", marker=st["marker"], color=st["color"],
                         linewidth=st["linewidth"], markersize=st["markersize"],
                         alpha=st["alpha"], path_effects=st["path_effects"], zorder=z_t)

            x_a, y_a = _avg_series_from_summary(jfp)
            z_a = _zorder_for_length(len(y_a))
            ax_right.plot(x_a, y_a, "--", marker=st["marker"], color=st["color"],
                          linewidth=st["linewidth"], markersize=st["markersize"],
                          alpha=st["alpha"], path_effects=st["path_effects"], zorder=z_a)

    plan_str = _plan_key_to_string(plan_key) if plan_key else "plan:NA"
    plan_str = plan_str.replace(":", " - ")
    if target_key == "pareto_scores":
        ax_left.set_title("pareto_scores (LUTs solid, FFs dashed)", fontsize=11)
        ax_left.set_ylabel("pareto_scores (lower is better)")
        ax_right.set_title("average_function_lines", fontsize=11)
    else:
        ax_left.set_title(f"{target_key}", fontsize=11)
        ax_right.set_title("average_function_lines", fontsize=11)
        ax_left.set_ylabel(target_key)
        ax_right.set_ylabel("average_function_lines")

    for ax in (ax_left, ax_right):
        ax.set_xlabel("Iteration (0 = seed)")
        ax.grid(True, linestyle=":", linewidth=0.5)

    _legend_two_panel(fig, target_key)
    fig.suptitle(f"Model: {model_name}, Target list: {plan_str}", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    return fig

# ---------- main ----------
def main():
    runs = _collect_runs(DATA_ROOT, RUN_NAMES)
    if not runs:
        print("[warn] No runs found under:", DATA_ROOT)
        return

    groups: Dict[Tuple[str, Tuple[Tuple[str,int], ...]], List[Tuple[str, Path]]] = {}
    for run_dir in runs:
        model_dirs = _collect_model_dirs(run_dir)
        for md in model_dirs:
            model_name = md.name
            for seed_id, jfp in _find_seed_jsons(md):
                tl_dict = _get_target_list_dict(jfp)
                plan_key = _canonical_plan_key(tl_dict)
                groups.setdefault((model_name, plan_key), []).append((seed_id, jfp))

    if not groups:
        print("[info] Nothing to plot.")
        return

    for (model_name, plan_key), entries in groups.items():
        plan_targets = [k for (k, v) in plan_key if v and k in TARGETS]
        for target_key in plan_targets:
            fig = _plot_target_group(model_name, plan_key, entries, target_key)
            if fig:
                safe_plan_str = _plan_key_to_string(plan_key).replace("/", "_").replace(" ", "")
                suffix = "__completed" if ONLY_COMPLETED else ""
                out_path = OUT_DIR / f"{model_name}__{safe_plan_str}__{target_key}{suffix}.png"
                fig.savefig(out_path, dpi=300)
                plt.close(fig)
                print("Saved:", out_path)

if __name__ == "__main__":
    main()
