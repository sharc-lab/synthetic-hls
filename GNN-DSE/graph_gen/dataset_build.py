from pathlib import Path

import re
import shutil
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "dataset_sources"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR =  # Path to SyntheticHLS experiments directory, e.g. Path("/path/to/workspace_multi_targets")
RUN_NAMES = [
    "run__2026-03-27_14-47-14",
]
FINAL_ITER = 8

# Leave empty / None to include everything found under each run.
MODEL_NAMES = ["gpt-oss-120b"]   # e.g. ["gpt-oss-120b"]
DOMAIN_NAMES = None  # e.g. ["eng_sim", "crypto_bc"]

FINAL_RE = re.compile(r"^(seed_design_\d+)_iter_(\d+)$")


def get_synthetic_hls_tcl_templates_dir() -> Path:
    import synthetic_hls
    tcl_dir = Path(synthetic_hls.__file__).resolve().parent / "tcl_templates"
    if not tcl_dir.is_dir():
        raise FileNotFoundError(f"Could not find synthetic_hls tcl_templates dir: {tcl_dir}")
    return tcl_dir


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copytree_fresh(src: Path, dst: Path) -> None:
    safe_rmtree(dst)
    shutil.copytree(src, dst)


def unique_name(base_name: str, used: dict[str, int]) -> str:
    if base_name not in used:
        used[base_name] = 0
        return base_name
    used[base_name] += 1
    return f"{base_name}_{used[base_name]}"


def find_kernel_files(design_dir: Path) -> tuple[Path, Path, Path]:
    tb_cpp = [p for p in design_dir.glob("*.cpp") if p.name.endswith("_tb.cpp")]
    kernel_cpp = [p for p in design_dir.glob("*.cpp") if not p.name.endswith("_tb.cpp")]
    kernel_h = []
    for ext in ("*.h", "*.hpp", "*.hh"):
        kernel_h.extend(design_dir.glob(ext))

    if len(tb_cpp) != 1:
        raise ValueError(f"Expected exactly 1 testbench cpp in {design_dir}, found {len(tb_cpp)}")
    if len(kernel_cpp) != 1:
        raise ValueError(f"Expected exactly 1 kernel cpp in {design_dir}, found {len(kernel_cpp)}")
    if len(kernel_h) != 1:
        raise ValueError(f"Expected exactly 1 kernel header in {design_dir}, found {len(kernel_h)}")

    return kernel_cpp[0], kernel_h[0], tb_cpp[0]


def patch_hls_template(design_dir: Path, kernel_name: str, top_function_name: str) -> None:
    hls_template_fp = design_dir / "hls_template.tcl"
    if not hls_template_fp.exists():
        raise FileNotFoundError(f"hls_template.tcl not found in {design_dir}")
    text = hls_template_fp.read_text()
    text = text.replace("[top_function_name]", top_function_name)
    text = text.replace("[kernel_name]", kernel_name)
    hls_template_fp.write_text(text)


def restructure_design_dir(design_dir: Path, tcl_templates_dir: Path) -> str:
    src_dir = design_dir / "src"
    tb_dir = design_dir / "tb"
    src_dir.mkdir(exist_ok=True)
    tb_dir.mkdir(exist_ok=True)

    kernel_cpp, kernel_h, tb_cpp = find_kernel_files(design_dir)

    shutil.move(str(kernel_cpp), src_dir / kernel_cpp.name)
    shutil.move(str(kernel_h), src_dir / kernel_h.name)
    shutil.move(str(tb_cpp), tb_dir / tb_cpp.name)

    for fp in tcl_templates_dir.iterdir():
        if fp.is_file():
            shutil.copy2(fp, design_dir / fp.name)

    top_fp = design_dir / "top.txt"
    if not top_fp.exists():
        raise FileNotFoundError(f"top.txt not found in {design_dir}")
    top_function_name = top_fp.read_text().strip()
    if not top_function_name:
        raise ValueError(f"top.txt is empty in {design_dir}")

    kernel_name = (src_dir / kernel_cpp.name).stem
    patch_hls_template(design_dir, kernel_name, top_function_name)
    return kernel_name


def process_one_design(
    src_design_dir: Path,
    dst_domain_dir: Path,
    used_names: dict[str, int],
    tcl_templates_dir: Path,
) -> Path:
    tmp_dst = dst_domain_dir / src_design_dir.name
    copytree_fresh(src_design_dir, tmp_dst)

    kernel_name = restructure_design_dir(tmp_dst, tcl_templates_dir)
    final_name = unique_name(kernel_name, used_names)
    final_dst = dst_domain_dir / final_name

    if tmp_dst != final_dst:
        safe_rmtree(final_dst)
        tmp_dst.rename(final_dst)

    return final_dst


def iter_target_domain_dirs(run_dir: Path):
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if MODEL_NAMES is not None and model_dir.name not in MODEL_NAMES:
            continue
        for domain_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if DOMAIN_NAMES is not None and domain_dir.name not in DOMAIN_NAMES:
                continue
            yield model_dir, domain_dir


def main() -> None:
    tcl_templates_dir = get_synthetic_hls_tcl_templates_dir()

    dataset_final_root = OUTPUT_DIR / "dataset_final"
    dataset_base_root = OUTPUT_DIR / "dataset_base"
    dataset_final_root.mkdir(parents=True, exist_ok=True)
    dataset_base_root.mkdir(parents=True, exist_ok=True)

    used_names_final: dict[str, dict[str, int]] = defaultdict(dict)
    used_names_base: dict[str, dict[str, int]] = defaultdict(dict)

    total_final = 0
    total_base = 0

    for run_name in RUN_NAMES:
        run_dir = EXPERIMENTS_DIR / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        for model_dir, domain_dir in iter_target_domain_dirs(run_dir):
            final_designs_dir = domain_dir / "final_designs"
            seed_pass_dir = domain_dir / "seed_designs" / "pass_designs"

            if not final_designs_dir.is_dir():
                print(f"[WARN] Missing final_designs: {final_designs_dir}")
                continue
            if not seed_pass_dir.is_dir():
                print(f"[WARN] Missing seed pass_designs: {seed_pass_dir}")
                continue

            out_final_domain = dataset_final_root / domain_dir.name
            out_base_domain = dataset_base_root / domain_dir.name
            out_final_domain.mkdir(parents=True, exist_ok=True)
            out_base_domain.mkdir(parents=True, exist_ok=True)

            for design_dir in sorted(p for p in final_designs_dir.iterdir() if p.is_dir()):
                m = FINAL_RE.match(design_dir.name)
                if not m:
                    continue

                seed_name = m.group(1)
                iter_idx = int(m.group(2))
                if iter_idx != FINAL_ITER:
                    continue

                seed_design_dir = seed_pass_dir / seed_name
                if not seed_design_dir.is_dir():
                    print(f"[WARN] Missing matching seed design for {design_dir}: {seed_design_dir}")
                    continue

                final_dst = process_one_design(
                    src_design_dir=design_dir,
                    dst_domain_dir=out_final_domain,
                    used_names=used_names_final[domain_dir.name],
                    tcl_templates_dir=tcl_templates_dir,
                )
                total_final += 1

                base_dst = process_one_design(
                    src_design_dir=seed_design_dir,
                    dst_domain_dir=out_base_domain,
                    used_names=used_names_base[domain_dir.name],
                    tcl_templates_dir=tcl_templates_dir,
                )
                total_base += 1

    print(f"[DONE] Copied {total_final} final designs into {dataset_final_root}")
    print(f"[DONE] Copied {total_base} base designs into {dataset_base_root}")


if __name__ == "__main__":
    main()
