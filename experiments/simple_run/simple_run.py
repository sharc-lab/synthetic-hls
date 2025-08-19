import datetime
from pathlib import Path

from synthetic_hls.engine import SyntheticHLSEngine
from synthetic_hls.llm_models import build_model_remote_openrouter

### Setup Directories ###

DIR_CURRENT = Path(__file__).parent
DIR_WORKSPACE = DIR_CURRENT / "workspace"

### Setup Models ###

MODEL_NAME = "openai/gpt-oss-20b"

model = build_model_remote_openrouter(MODEL_NAME)

### Run Main Stuff ###

run_name = f"run__{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

n_seed_designs = 10

engine = SyntheticHLSEngine(
    run_name=run_name,
    dir_workspace=DIR_WORKSPACE,
    model=model,
    n_seed_designs=n_seed_designs,
)
engine.run()
