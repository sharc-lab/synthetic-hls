from pathlib import Path

from hlsfactory.datasets_builtin import (
    datasets_builder,
)
from hlsfactory.flow_vitis import (
    VitisHLSImplFlow,
    VitisHLSImplReportFlow,
    VitisHLSSynthFlow,
)
from hlsfactory.framework import (
    Design,
    DesignDataset,
    DesignDatasetCollection,
    count_total_designs_in_dataset_collection,
)
# from hlsfactory.opt_dsl_frontend import OptDSLFrontend
from hlsfactory.opt_dsl_frontend_v2 import OptDSLFrontend
from hlsfactory.utils import (
    DirSource,
    ToolPathsSource,
    get_tool_paths,
    get_work_dir,
    remove_and_make_new_dir_if_exists,
)

from flow_harp_opt_dsl_v2_n import run_graph_gen

WORK_DIR_TOP = get_work_dir(dir_source=DirSource.ENVFILE)
WORK_DIR = WORK_DIR_TOP / "raw_data"
remove_and_make_new_dir_if_exists(WORK_DIR)

N_JOBS = 14
CPU_AFFINITY = list(range(N_JOBS))

VITIS_HLS_PATH, VIVADO_PATH = get_tool_paths(tool_paths_source=ToolPathsSource.ENVFILE)
VIVADO_BIN = VIVADO_PATH / "bin" / "vivado"
VITIS_HLS_BIN = VITIS_HLS_PATH / "bin" / "vitis_hls"

CURRENT_DIR = Path(__file__).parent

N_RANDOM_SAMPLES = 64
RANDOM_SAMPLE_SEED = 64

dataset_sources = CURRENT_DIR / "dataset_sources" / "dataset_final"

datasets: DesignDatasetCollection = {}

for dataset_dir in dataset_sources.glob("*"):
    dataset_name = dataset_dir.name
    designs = DesignDataset.from_dir(
        dataset_name,
        dataset_dir,
    ).copy_dataset(WORK_DIR)
    datasets[dataset_name] = designs

# OptDSL Frontend Execution
opt_dsl_frontend = OptDSLFrontend(
    WORK_DIR,
    random_sample=True,
    random_sample_num=N_RANDOM_SAMPLES,
    random_sample_seed=RANDOM_SAMPLE_SEED,
    log_execution_time=True,
)

datasets_post_frontend = opt_dsl_frontend.execute_multiple_design_datasets_fine_grained_parallel(
    datasets,
    True,
    lambda x: f"{x}__post_frontend",
    n_jobs=N_JOBS,
    cpu_affinity=CPU_AFFINITY,
)

post_frontend_roots = sorted(WORK_DIR.glob("*__post_frontend"))
if not post_frontend_roots:
    raise RuntimeError(f"No __post_frontend datasets found under: {WORK_DIR}")

# HARP Graph Generation
OUT_TOP = WORK_DIR_TOP / "harp_optdslv2_graphs"
OUT_TOP.mkdir(parents=True, exist_ok=True)

for designs_root in post_frontend_roots:
    dataset_out = OUT_TOP / designs_root.name
    dataset_out.mkdir(parents=True, exist_ok=True)

    # initial: generates .ll + prograML graph + pragma overlay + processed gexf
    run_graph_gen(
        mode="initial",
        designs_root=str(designs_root),
        out_root=str(dataset_out),
        opt_tcl_name="opt.tcl",
    )

    # auxiliary (base)
    run_graph_gen(
        mode="auxiliary",
        connected=False,
        designs_root=str(designs_root),
        out_root=str(dataset_out),
        opt_tcl_name="opt.tcl",
    )

    # # auxiliary (connected)
    # run_graph_gen(
    #     mode="auxiliary",
    #     connected=True,
    #     designs_root=str(designs_root),
    #     out_root=str(dataset_out),
    #     opt_tcl_name="opt.tcl",
    # )

    # # hierarchy (uses connected auxiliary + per-design .ll)
    # run_graph_gen(
    #     mode="hierarchy",
    #     connected=True,
    #     designs_root=str(designs_root),
    #     out_root=str(dataset_out),
    #     opt_tcl_name="opt.tcl",
    # )

print(f"[DONE] Graphs written under: {OUT_TOP}")

total_count = count_total_designs_in_dataset_collection(
    datasets_post_frontend,
)
print(f"Total Designs: {total_count}")

# Timeout for HLS synthesis and Vivado implementation
TIMEOUT_HLS_SYNTH = 60.0 * 24  # 24 minutes
TIMEOUT_HLS_IMPL = 60.0 * 90  # 90 minutes

total_time_estimation = (
    total_count * (TIMEOUT_HLS_SYNTH + TIMEOUT_HLS_IMPL) / N_JOBS
)
print(
    f"Estimated worst-case build time:\n{total_time_estimation} seconds\n{total_time_estimation / 60} minutes\n{total_time_estimation / 3600} hours",
)

toolflow_vitis_hls_synth = VitisHLSSynthFlow(
    vitis_hls_bin=str(VITIS_HLS_BIN),
    env_var_xilinx_hls=str(VITIS_HLS_PATH),
    env_var_xilinx_vivado=str(VIVADO_PATH),
)
datasets_post_hls_synth = (
    toolflow_vitis_hls_synth.execute_multiple_design_datasets_fine_grained_parallel(
        datasets_post_frontend,
        False,
        n_jobs=N_JOBS,
        cpu_affinity=CPU_AFFINITY,
        timeout=TIMEOUT_HLS_SYNTH,
    )
)

toolflow_vitis_hls_implementation = VitisHLSImplFlow(
    vitis_hls_bin=str(VITIS_HLS_BIN),
    env_var_xilinx_hls=str(VITIS_HLS_PATH),
    env_var_xilinx_vivado=str(VIVADO_PATH),
)
datasets_post_hls_implementation = toolflow_vitis_hls_implementation.execute_multiple_design_datasets_fine_grained_parallel(
    datasets_post_hls_synth,
    False,
    n_jobs=N_JOBS,
    cpu_affinity=CPU_AFFINITY,
    timeout=TIMEOUT_HLS_IMPL,
)

toolflow_vitis_hls_impl_report = VitisHLSImplReportFlow(
    vitis_hls_bin=str(VITIS_HLS_BIN),
    vivado_bin=str(VIVADO_BIN),
    env_var_xilinx_hls=str(VITIS_HLS_PATH),
    env_var_xilinx_vivado=str(VIVADO_PATH),
)
toolflow_vitis_hls_impl_report.execute_multiple_design_datasets_fine_grained_parallel(
    datasets_post_hls_implementation,
    False,
    n_jobs=N_JOBS,
    cpu_affinity=CPU_AFFINITY,
)