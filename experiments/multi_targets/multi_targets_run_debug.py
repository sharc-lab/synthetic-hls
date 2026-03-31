import datetime
from importlib.resources import files
from pathlib import Path

from dotenv import dotenv_values

from synthetic_hls.engine import SyntheticHLSEngine
from synthetic_hls.llm_models import build_model_remote_openrouter
from synthetic_hls.utils import unwrap
from synthetic_hls.vhls_tools import auto_find_vitis_hls_dir, auto_find_vivado_dir

# Directories and Paths
DIR_CURRENT = Path(__file__).resolve().parent
DIR_WORKSPACE = DIR_CURRENT / "workspace_multi_targets"

ENV_FP = DIR_CURRENT / ".env"
env = dotenv_values(ENV_FP)

CLANG_PATH = Path(env["CLANG_PATH"]) if env.get("CLANG_PATH") else None
INCLUDE_PATHS = [
    Path(p.strip()) for p in env.get("INCLUDE_PATHS", "").split(",") if p.strip()
]
vitis_hls_dir = unwrap(auto_find_vitis_hls_dir(), "Vitis HLS bin not auto found")
vivado_dir = unwrap(auto_find_vivado_dir(), "Vivado bin not auto found")

# Setup Models (OpenRouter)
API_KEY_OPENROUTER = env["OPENROUTER_API_KEY"]
MODELS_NAMES = ["openai/gpt-oss-120b"]
models = [
    build_model_remote_openrouter(name, api_key=API_KEY_OPENROUTER)
    for name in MODELS_NAMES
]

# Available Targets now
available_targets = [
    "num_functions",
    "max_call_chain_depth",
    "average_function_lines",
    "pareto_scores",
]

# Available Domains now
available_domains = {
    "sci_sim": "Scientific Research and Simulation",
    "ml_ai": "Machine Learning and Artificial Intelligence",
    "fin_model": "Financial Modeling and Analysis",
    "eng_sim": "Engineering and Design Simulation",
    "data_big": "Data Analytics and Big Data Processing",
    "gfx_render": "Graphics Rendering and Animation",
    "crypto_bc": "Cryptography and Blockchain",
    "telecom_sp": "Telecommunications and Signal Processing",
    "astro": "Astronomy and Astrophysics",
    "health_med": "Healthcare and Medical Imaging",
}

# -----------------------------------------------------------------------------
# Experiment Configurations
#
# domain_list: list of domain keys (must exist in keys of AVAILABLE_DOMAINS)
# target_list: list of metrics to optimize (must exist in AVAILABLE_TARGETS)
# n_feedback_iterations:
#   - int  -> run that many iterations for EVERY target in target_list
#   - list -> Must be the same length as target_list; per-target iteration counts
#   The engine executes: target_list[i] repeated n_feedback_iterations[i] times.
#   Multi-Targets Examples:
#       target_list=["num_functions","max_call_chain_depth"], n_feedback_iterations=3
#       -> n_funcs for first 3 iterations, max_depth for next 3 iterations
#       target_list=["num_functions","max_call_chain_depth","pareto_scores"], n_feedback_iterations=[2,2,3]
#       -> n_funcs for first 2 iterations, max_depth for next 2 iterations, pareto_scores for last 3 iterations
#
# n_seed_designs: Number of initial seed designs to generate per domain.
# n_samples: Number of candidate mutations generated per iteration, per seed design.
# -----------------------------------------------------------------------------

domain_list = [
    "sci_sim",
]
assert all(d in available_domains for d in domain_list), (
    "All domains in domain_list must be keys in available_domains."
)

target_list = [
    "pareto_scores",
]
n_feedback_iterations = [2]

n_seed_designs = 4
n_samples = 4

run_name = f"run__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
target_plan = ", ".join(
    [
        f"{t}: {iters} iteration(s)"
        for t, iters in zip(target_list, n_feedback_iterations)
    ]
)
print(
    f"Run Name: {run_name}, \n Models: {MODELS_NAMES}, \n Domains: {domain_list}, \n Target: {target_plan}"
)

# -----------------------------------------------------------------------------
# Engine Run Configurations
#
## Parallelism:
# - n_jobs_design: total design-level parallelism budget used by the engine
#       - Overall "design-level" concurrency budget for the engine's outer scheduling
#       (across model * domain * seed * sample tasks).
#       - Note: This does NOT set tool internal parallelism. Tool-side concurrency is
#       controlled by n_jobs_pool_llm / n_jobs_pool_csim / n_jobs_pool_synth, plus
#       n_jobs_hlsfactory for pareto_scores.
# - n_jobs_hlsfactory: parallel jobs inside HLSFactory (affects pareto_scores stage)
# - n_jobs_pool_*: internal stage pools for LLM / csim / synth
#
## Other Flags:
# - fix: whether to attempt to fix if all design samples fail in one iteration (default: True)
# - run_vivado_impl: whether to run Vivado implementation after synthesis (default: False)
# -----------------------------------------------------------------------------

engine = SyntheticHLSEngine(
    run_name=run_name,
    dir_workspace=DIR_WORKSPACE,
    vitis_hls_dir=vitis_hls_dir,
    vivado_dir=vivado_dir,
    models=models,
    n_jobs_pool_llm=72,
    n_jobs_pool_csim=72,
    n_jobs_pool_synth=72,
    n_jobs_pool_hlsfactory=64,
    temperature=0.8,
    clang_path=CLANG_PATH,
    include_paths=INCLUDE_PATHS,
)

engine.run(
    domain_list=domain_list,
    target_list=target_list,
    n_feedback_iterations=n_feedback_iterations,
    n_seed_designs=n_seed_designs,
    n_samples=n_samples,
    n_jobs_design=72,
    fix=True,
    run_vivado_impl=False,
)

print(f"Run {run_name} completed.")
