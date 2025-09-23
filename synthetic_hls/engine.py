import json
import shutil
from pathlib import Path
from joblib import Parallel, delayed
from typing import List, Optional

from synthetic_hls.design import Design, find_design_dirs
from synthetic_hls.llm_models import Model
from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool
from synthetic_hls.prompting import build_prompt_gen_zero_shot_no_input_with_opt, build_prompt_gen_zero_shot_single_input_with_opt, build_prompt_mutate_target
from synthetic_hls.design_evaluator import DesignEvaluator, EvalThreadPools


class FeedbackDesignLoop:
    """
    Runs an feedback loop to iteratively improve a single seed design 
    toward a specified target (Pareto scores or AST complexity metrics).

    The loop structure:
      - For each sample (to introduce variation / temperature diversity):
          - For up to n_max_iterations:
            - For up to n_max_versions:
              - Start from version 0: mutate toward target.
              - If error -> If fix mode, try to fix the error; else regenerate.
              - On success, the accepted version becomes the new baseline for the
                next iteration; otherwise the iteration terminates early.
    """
    def __init__(
        self, 
        output_data_dir: Path,
        final_designs_dir: Path,
        model: Model,
        evaluator: DesignEvaluator,
        n_samples: int,
        n_max_iterations: int,
        n_max_versions: int,
        pools: EvalThreadPools,
        fix: bool = True,
    ):
        self.output_data_dir = output_data_dir
        self.final_designs_dir = final_designs_dir
        self.model = model
        self.evaluator = evaluator
        self.n_samples = n_samples
        self.n_max_iterations = n_max_iterations
        self.n_max_versions = n_max_versions
        self.pools = pools
        self.fix = fix

    def run(self, seed_design: Design, target_list: list[str]):
        Parallel(n_jobs=4, backend="threading")(
            delayed(self.run_single)(sample_idx, seed_design, target_list) for sample_idx in range(self.n_samples)
        )

        root_dir = self.output_data_dir / f"{'__'.join(target_list)}" / seed_design.name
        seed_design_eval_data_fp = seed_design.design_dir / "single_eval_data.json"
        seed_design_eval_data = json.loads(seed_design_eval_data_fp.read_text())
        all_summary = {
            "target_label": target_list,
            "model_name": self.model.name,
            "seed_design": {
                "Name": seed_design.name,
                "Path": str(seed_design.design_dir),
                "target_results": {
                    **({"pareto_scores": seed_design_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None),}
                    if "pareto_scores" in target_list else {}),
                    "num_functions": seed_design_eval_data.get("kernel_ast_out", {}).get("num_functions", None),
                    "max_call_chain_depth": seed_design_eval_data.get("kernel_ast_out", {}).get("max_call_chain_depth", None),
                    "average_function_lines": seed_design_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None)
                    }
                },
            "samples": {}
        }
        # Summarize all feedback runs of each seed design
        for sample_idx in range(self.n_samples):
            s_entry = all_summary["samples"].setdefault(f"sample_{sample_idx}", {})
            for iter in range(len(target_list) * self.n_max_iterations):
                iter_entry = s_entry.setdefault(f"iter_{iter}", {})
                for ver in range(0, self.n_max_versions + 1):
                    ver_dir = root_dir / f"sample_{sample_idx}" / f"iter{iter}_v{ver}" 
                    ver_eval_data_fp = ver_dir / "single_eval_data.json"
                    if ver_eval_data_fp.exists():  
                        ver_entry = iter_entry.setdefault(f"ver_{ver}", {"Path": str(ver_dir), "target_result": {}})
                        ver_entry["Path"] = str(ver_dir)
                        ver_eval_data = json.loads(ver_eval_data_fp.read_text())
                        ver_entry["Status"] = ver_eval_data.get("status", None)
                        if "pareto_scores" in target_list:
                            ver_entry["target_result"]["pareto_scores"] = ver_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None)
                        else:
                            ver_entry["target_result"]["num_functions"] = ver_eval_data.get("kernel_ast_out", {}).get("num_functions", None)
                            ver_entry["target_result"]["max_call_chain_depth"] = ver_eval_data.get("kernel_ast_out", {}).get("max_call_chain_depth", None)
                            ver_entry["target_result"]["average_function_lines"] = ver_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None)
        (root_dir / "all_data_summary.json").write_text(json.dumps(all_summary, indent=4))

    def run_single(self, sample_idx: int, seed_design: Design, target_list: list[str]):
        design_to_improve = seed_design
        sample_output_dir = self.output_data_dir / f"{'__'.join(target_list)}" / seed_design.name / f"sample_{sample_idx}"
        seed_name = seed_design.name
        sample_eval_data = {}
        target_plan = [target for target in target_list for i in range(self.n_max_iterations)]

        for iter, target in enumerate(target_plan):
            sample_eval_data[f"iter_{iter}"] = {}
            max_ver_reached = False
            error_message = None
            current_design = None
            design_to_fix = None
            for ver in range(0, self.n_max_versions + 1):
                if error_message is None:
                    if current_design is None:
                        prompt = build_prompt_mutate_target(
                            design=design_to_improve,
                            target=target
                        )
                        error_message, current_design = self.evaluator.evaluate_design(
                            model=self.model,
                            pools=self.pools,
                            output_design_data_dir=sample_output_dir / f"iter{iter}_v{ver}",
                            prompt=prompt,
                            design_id=f"{seed_name}_s{sample_idx}_iter{iter}_v{ver}",
                            output_format="FULL_CODE",
                            full_flow=True if target == "pareto_scores" else False,
                            seed_design=design_to_improve,
                        )
                    else:
                        design_to_improve = current_design
                        if iter == len(target_list) * self.n_max_iterations - 1:
                            final_design_eval_data_fp = design_to_improve.design_dir.parent / "single_eval_data.json"
                            final_design_dir = self.final_designs_dir / f"{'__'.join(target_list)}" / seed_design.name / f"{seed_design.name}_{sample_idx}"
                            design_to_improve.copy_to(final_design_dir)
                            shutil.copy(final_design_eval_data_fp, final_design_dir)
                        break
                else:
                    if current_design is None:
                        design_to_fix = design_to_improve if ver == 1 else design_to_fix
                    else:
                        design_to_fix = current_design
                        prompt = build_prompt_mutate_target(
                            design=design_to_fix if self.fix else design_to_improve,
                            fix=self.fix,
                            error_message=error_message,
                            target=target
                        )
                    error_message, current_design = self.evaluator.evaluate_design(
                        model=self.model,
                        pools=self.pools,
                        output_design_data_dir=sample_output_dir / f"iter{iter}_v{ver}",
                        prompt=prompt,
                        design_id=f"{seed_name}_s{sample_idx}_iter{iter}_v{ver}",
                        output_format="OPTDSL" if error_message == "OptDSL Error" and self.fix else "FULL_CODE",
                        full_flow=True if target == "pareto_scores" else False,
                        seed_design=design_to_fix if self.fix else design_to_improve,
                    )
                ver_eval_data_fp = sample_output_dir / f"iter{iter}_v{ver}" / "single_eval_data.json"
                if ver_eval_data_fp.exists():
                    sample_eval_data[f"iter_{iter}"][f"ver_{ver}"] = json.loads(ver_eval_data_fp.read_text())
                if ver == self.n_max_versions:
                    max_ver_reached = True
            if max_ver_reached:
                break
        sample_eval_data_fp = sample_output_dir / "sample_eval_data.json"
        sample_eval_data_fp.write_text(json.dumps(sample_eval_data, indent=4))


class SeedDesignGenerator:
    """
    Generates an initial set of seed designs either from scratch (zero-shot)
    or by conditioning on a provided reference design.
    """
    def __init__(
        self,
        seed_design_dir: Path,
        model: Model,
        evaluator: DesignEvaluator, 
        target: str,
        n_seed_designs: int,
        pools: EvalThreadPools,
    ):
        self.seed_design_dir = seed_design_dir
        self.model = model
        self.evaluator = evaluator
        self.target = target
        self.n_seed_designs = n_seed_designs
        self.pools = pools

        self.seed_design_names = [
            f"seed_design__{i}" for i in range(self.n_seed_designs)
        ]

    def run(self, reference_design: Design = None):
        Parallel(n_jobs=4, backend="threading")(
            delayed(self.run_single)(seed_design_name, reference_design) for seed_design_name in self.seed_design_names
        )
        # Aggregate all eval data from all seed designs
        all_data = {}
        for seed_design_name in self.seed_design_names:
            output_seed_data_dir = self.seed_design_dir / seed_design_name
            seed_eval_data_fp = output_seed_data_dir / "single_eval_data.json"
            if seed_eval_data_fp.exists():
                all_data[seed_design_name] = json.loads(seed_eval_data_fp.read_text())
        all_data_fp = self.seed_design_dir / "all_eval_data.json"
        all_data_fp.write_text(json.dumps(all_data, indent=4))


    def run_single(self, seed_design_name: str, reference_design: Design = None):
        if reference_design is None:
            prompt = build_prompt_gen_zero_shot_no_input_with_opt()
        else:
            seed_design_name = f"{seed_design_name}__ref_{reference_design.name}"
            prompt = build_prompt_gen_zero_shot_single_input_with_opt(
                design_description_fp=reference_design.description_fp,
                design_h=reference_design.design_h_fp,
                design_kernel=reference_design.design_kernel_fp,
                design_tb=reference_design.design_tb_fp,
                design_opt=reference_design.design_opt_fp,
                design_pareto_score=reference_design.design_pareto_score_fp,
            )
        error_message, current_design = self.evaluator.evaluate_design(
            model=self.model,
            pools=self.pools,
            output_design_data_dir=self.seed_design_dir / seed_design_name,
            prompt=prompt,
            design_id=seed_design_name,           
            output_format="FULL_CODE",
            full_flow=True if self.target == "pareto_scores" else False,
            seed_design=None if reference_design is None else reference_design,
        )
        if error_message is None:
            pass_designs_dir = self.seed_design_dir / "pass_designs"
            current_design.copy_to(pass_designs_dir / seed_design_name)
            current_design_eval_data_fp = self.seed_design_dir / seed_design_name / "single_eval_data.json"
            shutil.copy(current_design_eval_data_fp, pass_designs_dir / seed_design_name)


class SyntheticHLSEngine:
    """
    The main engine to run the full synthetic HLS design generation and improvement loop.
    Creates a workspace structure like:
      <dir_workspace>/<run_name>/
        ├─ seed_designs/            # generated seed designs
        ├─ feedback_runs/           # full per-target/per-seed/per-sample iteration/version artifacts
        └─ final_designs/           # final improved designs per target and seed
    """
    def __init__(
        self, 
        run_name: str, 
        dir_workspace: Path,
        template_files_path: Path,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        model: Model,
        temperature: float = 0.7,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None, 
    ):
        self.run_name = run_name
        self.dir_workspace = dir_workspace
        self.model = model
        self.temperature = temperature
        self.clang_path = clang_path
        self.include_paths = include_paths

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

        self.dir_feedback_runs = self.run_dir / "feedback_runs"
        self.dir_feedback_runs.mkdir()

        self.dir_final_designs = self.run_dir / "final_designs"
        self.dir_final_designs.mkdir()

        self.template_files_path = template_files_path
        self.vitis_hls_tool_csim = vitis_hls_tool_csim
        self.vitis_hls_tool_synth = vitis_hls_tool_synth

        self.pools = EvalThreadPools(
            n_jobs_pool_llm=8,
            n_jobs_pool_csim=8,
            n_jobs_pool_synth=4,
        )
        self.evaluator = DesignEvaluator(
            vitis_hls_tool_csim=self.vitis_hls_tool_csim,
            vitis_hls_tool_synth=self.vitis_hls_tool_synth,
            template_files_path=self.template_files_path,
            temperature=self.temperature,
            clang_path=self.clang_path,
            include_paths=self.include_paths,
        )

    def run(
        self,
        target_list: list[str] = ["pareto_scores"],
        reference_designs: list[Design] = None,  # List of reference designs to base seed designs on
        n_seed_designs: int = 5,
        n_samples: int = 8,
        n_feedback_iterations: int = 5,
        n_max_versions: int = 16,
        fix: bool = False,  # Whether to try to fix errors
    ):
        seed_design_generator = SeedDesignGenerator(
            seed_design_dir=self.dir_seed_designs,
            model=self.model,
            evaluator=self.evaluator,
            target=target_list[0],
            n_seed_designs=n_seed_designs,
            pools=self.pools,
        )
        # Generate seed designs either from scratch or based on reference designs
        max_seed_gen_attempts = 5
        seed_gen_attempt = 0
        input_designs = []
        while seed_gen_attempt < max_seed_gen_attempts:
            if reference_designs is not None and len(reference_designs) > 0:
                for reference_design in reference_designs:
                    seed_design_generator.run(reference_design=reference_design)
            else:
                seed_design_generator.run()

            seed_designs_dirs = find_design_dirs(self.dir_seed_designs / "pass_designs")
            seed_designs = [
                Design(d, name=d.name) for d in seed_designs_dirs
            ]
            if len(seed_designs) >= 1:
                input_designs = seed_designs
                break
            else:
                if seed_gen_attempt < max_seed_gen_attempts - 1:
                    print(f"Seed design generation attempt {seed_gen_attempt + 1} failed, retrying...")
                else:
                    raise ValueError("Seed design generation failed after maximum attempts.")
                for d in self.dir_seed_designs.iterdir():
                    if d.is_dir():
                        shutil.rmtree(d, ignore_errors=True)
                    else:
                        d.unlink(missing_ok=True)
            seed_gen_attempt += 1

        # Run feedback design loop for each target
        feedback_design_loop = FeedbackDesignLoop(
            output_data_dir=self.dir_feedback_runs,
            final_designs_dir=self.dir_final_designs,
            model=self.model,
            evaluator=self.evaluator,
            n_samples=n_samples,
            n_max_iterations=n_feedback_iterations,
            n_max_versions=n_max_versions,
            pools=self.pools,
            fix=fix,
        )

        Parallel(n_jobs=4, backend="threading")(
            delayed(feedback_design_loop.run)(input_design, target_list) for input_design in input_designs
        )
        result_designs_dirs = find_design_dirs(self.dir_final_designs / f"{'__'.join(target_list)}")
        result_designs = [
            Design(d, name=d.name) for d in result_designs_dirs
        ]
        input_designs = result_designs