import datetime
from pathlib import Path
from dotenv import dotenv_values

from synthetic_hls.engine import SyntheticHLSEngine
from synthetic_hls.llm_models import build_model_remote_openrouter
from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool, auto_find_vitis_hls_dir
from synthetic_hls.utils import unwrap

### Setup Directories ###
DIR_CURRENT = Path(__file__).parent
DIR_TOP = DIR_CURRENT.parent.parent
DIR_WORKSPACE = DIR_CURRENT / "workspace_multi_targets"
DIR_TEMPLATE_FILES = DIR_TOP / "tcl_templates"

# Set clang and include paths for ast analysis. If not set, it will try to use system default clang.
CLANG_PATH = dotenv_values(".env")["CLANG_PATH"]
INCLUDE_PATHS = [p for p in dotenv_values(".env")["INCLUDE_PATHS"].split(",") if p]
vitis_hls_dir = unwrap(auto_find_vitis_hls_dir(), "Vitis HLS bin not auto found")

API_KEY_OPENROUTER = dotenv_values(".env")["OPENROUTER_API_KEY"]

### Setup Models ###
MODELS_NAMES = ["openai/gpt-oss-120b"]
models = [build_model_remote_openrouter(name, api_key=API_KEY_OPENROUTER) for name in MODELS_NAMES]

### Run Main Stuff ###
# Available Targets now(Working on including others): "num_functions", "max_call_chain_depth", "average_function_lines", "pareto_scores"
available_targets = {
    "n_funcs": "num_functions",
    "max_depth": "max_call_chain_depth",
    "avg_func_loc": "average_function_lines",
    "scores": "pareto_scores"
}

target_keys_list = ["n_funcs", "max_depth", "avg_func_loc", "scores"]  # Choose from available_targets keys
target_list = [available_targets[k] for k in target_keys_list]
run_name = f"run__{'__'.join(target_keys_list)}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
print(f"Run Name: {run_name}, Target: {target_list}, Models: {MODELS_NAMES}")


engine = SyntheticHLSEngine(
    run_name=run_name,
    dir_workspace=DIR_WORKSPACE,
    vitis_hls_tool_csim=VitisHLSCSimTool(vitis_hls_dir),
    vitis_hls_tool_synth=VitisHLSSynthTool(vitis_hls_dir),
    models=models,
    temperature=0.9,
    clang_path=CLANG_PATH,
    include_paths=INCLUDE_PATHS
)

"""
Experiment configuration (multi-targets):
- target_list: list[str], List of target metrics to optimize.
- n_seed_designs: int, Number of seed designs to start with.
- n_samples: int, Number of samples to generate in each iteration.
- n_feedback_iterations: list[int] | int (use list for multiple targets), For each target in target_list, how many
    feedback iterations to run.
    -- Same length as target_list; i.e., here: 2 iters for first 3 targets, 3 for scores.
- n_jobs_design: int, Total design-level parallelism (seed-level * sample-level).
    -- Note: This controls only the outer design loop. Inside the engine
       there is separate thread pool for LLM, C-simulation, and synthesis
       (EvalThreadPools: n_jobs_pool_llm, n_jobs_pool_csim, n_jobs_pool_synth),
       which further parallelize work per design.
- n_jobs_hlsfactory: int, Parallel jobs inside HLSFactory for Pareto scores evaluation.
- fix: bool, Whether to let the engine try to fix if all samples fail.
- run_vivado_impl: bool, If True, run full Vivado implementation; False = HLS-only exploration.
"""
engine.run(
    target_list=target_list,
    n_seed_designs=36,
    n_samples=12,
    n_feedback_iterations=[2,2,2,3],
    n_jobs_design=24,
    n_jobs_hlsfactory=24,
    fix=True,
    run_vivado_impl=False
)
