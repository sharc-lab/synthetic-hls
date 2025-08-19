from pathlib import Path

from synthetic_hls.design import Design
from synthetic_hls.llm_models import Model


class SingleDesignLoop:
    def __init__(self, design, model: Model):
        self.design = design
        self.model = model

    def run(self):
        pass


class SeedDesignGenerator:
    def __init__(
        self,
        seed_design_dir: Path,
        model: Model,
        n_seed_designs: int,
    ):
        self.seed_design_dir = seed_design_dir
        self.model = model
        self.n_seed_designs = n_seed_designs

        self.seed_design_names = [
            f"seed_design__{i}" for i in range(self.n_seed_designs)
        ]

    def run(self):
        for seed_design_name in self.seed_design_names:
            self.run_single(seed_design_name)

    def run_single(self, seed_design_name: str):
        raise NotImplementedError()


class DesignFixer:
    def __init__(self, design: Design, model: Model, max_iterations: int):
        self.design = design
        self.model = model
        self.max_iterations = max_iterations

    def run(self):
        pass


class SyntheticHLSEngine:
    def __init__(
        self, run_name: str, dir_workspace: Path, model: Model, n_seed_designs=10
    ):
        self.run_name = run_name
        self.dir_workspace = dir_workspace
        self.model = model

        if not self.dir_workspace.exists():
            self.dir_workspace.mkdir(parents=True)

        self.run_dir = self.dir_workspace / self.run_name
        if not self.run_dir.exists():
            self.run_dir.mkdir()
        else:
            raise ValueError(
                f"Run directory {self.run_dir} already exists. Please choose a different run name."
            )

        self.dir_seed_designs = self.run_dir / "seed_designs"
        self.dir_seed_designs.mkdir()

        self.dir_final_designs = self.run_dir / "final_designs"
        self.dir_final_designs.mkdir()

        self.n_seed_designs = n_seed_designs

        # TODO: dirs for keeping track of mutation loops

    def run(self):
        seed_design_generator = SeedDesignGenerator(self.model, self.n_seed_designs)
        seed_design_generator.run()
