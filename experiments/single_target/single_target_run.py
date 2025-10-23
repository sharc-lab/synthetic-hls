import datetime
from pathlib import Path
from dotenv import dotenv_values
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from synthetic_hls.engine import SyntheticHLSEngine
from synthetic_hls.llm_models import build_model_remote_openrouter
from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool, auto_find_vitis_hls_dir
from synthetic_hls.utils import unwrap

### Setup Directories ###

DIR_CURRENT = Path(__file__).parent
DIR_TOP = DIR_CURRENT.parent.parent
DIR_WORKSPACE = DIR_CURRENT / "workspace"
DIR_TEMPLATE_FILES = DIR_TOP / "tcl_templates"

# Set clang and include paths for ast analysis. If not set, it will try to use system default clang.
CLANG_PATH = dotenv_values(".env")["CLANG_PATH"]
INCLUDE_PATHS = [p for p in dotenv_values(".env")["INCLUDE_PATHS"].split(",") if p]
vitis_hls_dir = unwrap(auto_find_vitis_hls_dir(), "Vitis HLS bin not auto found")

API_KEY_OPENROUTER = dotenv_values(".env")["OPENROUTER_API_KEY"]

### Setup Models ###

MODEL_NAME = "openai/gpt-oss-20b"

model = build_model_remote_openrouter(MODEL_NAME, api_key=API_KEY_OPENROUTER)

### Run Main Stuff ###

# Available Targets now(Working on including others): "num_functions", "max_call_chain_depth", "average_function_lines", "pareto_scores"
target = "max_call_chain_depth"
run_name = f"run__{target}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
print(f"Run Name: {run_name}, Target: {target}, Model: {MODEL_NAME}")

engine = SyntheticHLSEngine(
    run_name=run_name,
    dir_workspace=DIR_WORKSPACE / MODEL_NAME.split("/")[1],
    template_files_path=DIR_TEMPLATE_FILES,
    vitis_hls_tool_csim=VitisHLSCSimTool(vitis_hls_dir),
    vitis_hls_tool_synth=VitisHLSSynthTool(vitis_hls_dir),
    model=model,
    temperature=0.9,
    clang_path=CLANG_PATH,
    include_paths=INCLUDE_PATHS
)

engine.run(
    target_list=[target],
    n_seed_designs=24,
    n_samples=8,
    n_feedback_iterations=7,
    n_max_versions=4,
    fix=True
)
