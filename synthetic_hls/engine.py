import json
import shutil
import random
from pathlib import Path
from joblib import Parallel, delayed
from typing import List, Optional

from synthetic_hls.design import Design, find_design_dirs
from synthetic_hls.llm_models import Model, normalize_model_name
from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool
from synthetic_hls.prompting import build_prompt_gen_zero_shot_no_input_with_opt, build_prompt_gen_zero_shot_single_input_with_opt, build_prompt_gen_optdsl_v2, build_prompt_mutate_target
from synthetic_hls.design_evaluator import DesignEvaluator, EvalThreadPools

class FeedbackDesignLoop:
    """
    Runs an feedback loop to iteratively improve a single seed design 
    toward a specified target (Pareto scores or AST complexity metrics).

    The loop structure:
      - For each sample (to introduce variation / temperature diversity):
          - For up to n_iterations:
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
        pools: EvalThreadPools,
        n_samples: int = 8,
        n_iterations: int | list[int] = 5,
        n_jobs_sample_per_seed: int = 8,
        fix: bool = True
    ):
        self.output_data_dir = output_data_dir
        self.final_designs_dir = final_designs_dir
        self.model = model
        self.evaluator = evaluator
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.pools = pools
        self.fix = fix
        self.n_jobs_sample_per_seed = n_jobs_sample_per_seed
        self.model_name_normalized = normalize_model_name(model.name)

    def run(self, seed_design: Design, target_list: list[str]):
        targets_plan = []
        if isinstance(self.n_iterations, int):
            iters = [self.n_iterations] * len(target_list)
        else:
            iters = list(self.n_iterations)
        targets_plan = [t for t, n in zip(target_list, iters) for _ in range(n)]

        prev_design = seed_design
        root_dir = self.output_data_dir / seed_design.name
        for iter_idx, target in enumerate(targets_plan):
            target_next = targets_plan[iter_idx + 1] if iter_idx + 1 < len(targets_plan) else None
            iter_samples_dir = root_dir / f"iter_{iter_idx}_samples"
            iter_output_dir = root_dir / f"iter_{iter_idx}"
            pass_designs_dirs = []
            best_design, best_dir, best_val_1, best_val_2 = None, None, float("-inf"), float("-inf")
            shutil.rmtree(iter_samples_dir, ignore_errors=True)
            print(f"Starting iteration {iter_idx} target {target} for seed design {seed_design.name}")
            Parallel(n_jobs=self.n_jobs_sample_per_seed, backend="threading")(
                delayed(self.run_single)(iter_samples_dir, sample_idx, prev_design, target, prev_error_message=None) for sample_idx in range(self.n_samples)
            )
            pass_designs_dirs = find_design_dirs(iter_samples_dir/"pass_designs")

            if len(pass_designs_dirs) == 0 and self.fix:
                samples_to_fix = []
                for sample_idx in range(self.n_samples):
                    prev_design_fix_dir = iter_samples_dir / f"sample_{sample_idx}"
                    if not prev_design_fix_dir.exists():
                        continue
                    prev_design_fix = Design(prev_design_fix_dir / "design_generated")
                    prev_error_message = None
                    prev_design_eval_data_fp = prev_design_fix_dir / "single_eval_data.json"
                    if prev_design_eval_data_fp.exists():
                        prev_design_eval_data = json.loads(prev_design_eval_data_fp.read_text())
                        prev_error_message = prev_design_eval_data.get("error_message", None)
                    samples_to_fix.append((sample_idx, prev_design_fix, prev_error_message))
                
                if len(samples_to_fix) > 0:
                    Parallel(n_jobs=self.n_jobs_sample_per_seed, backend="threading")(
                        delayed(self.run_single)(iter_samples_dir, sample_idx, prev_design_fix, target, prev_error_message=prev_error_message) 
                        for (sample_idx, prev_design_fix, prev_error_message) in samples_to_fix
                    )
                pass_designs_dirs = find_design_dirs(iter_samples_dir/"pass_designs")

            # Evaluate passing samples and select the best one under target.
            if len(pass_designs_dirs) > 0:
                iter_eval_data = {}
                candidates = []
                for d in pass_designs_dirs:
                    sample_eval_data_fp = d / "single_eval_data.json"
                    if sample_eval_data_fp.exists():
                        sample_eval_data = json.loads(sample_eval_data_fp.read_text())
                        if target == "pareto_scores":
                            avg_func_loc = float(sample_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None))
                            candidates.append((d, avg_func_loc))
                        else:
                            target_val = float(sample_eval_data.get("kernel_ast_out", {}).get(target, None))
                            avg_func_loc = float(sample_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None))
                            candidates.append((d, target_val, avg_func_loc))
                
                if target == "pareto_scores":
                    best_dir, best_val_1 = random.choice(candidates)
                else:
                    best_dir, best_val_1, best_val_2 = max(candidates, key=lambda x: (x[1], x[2]))
    
                best_design = Design(best_dir, name=best_dir.name)

            if best_design is None:
                print(f"Iteration {iter_idx} target {target} found no passing designs. Stopping early.")
                break
            else:
                print(f"Iteration {iter_idx} target {target} found best design {best_design.design_dir}.")

            # If current or next target is pareto_scores, run pareto scores evaluation
            if target == "pareto_scores" or target_next == "pareto_scores":
                luts_ps, ffs_ps = 1.0, 1.0
                num_tries = 0
                candidates_final = []
                while True:
                    num_tries += 1
                    opt_out, out_dir = self.evaluator.run_hlsfactory_flow(
                        design_generated_dir=best_dir,
                        eval_dir_top=best_dir.parent,
                        eval_id=f"{iter_samples_dir.parent.name}_{iter_samples_dir.name}_s_{best_dir.name.split('_')[-1]}__{self.model_name_normalized}",
                        eval_design_id=f"{best_dir.name}",
                        top_function_name=Path(best_dir / "top.txt").read_text().strip(),
                    )
                    best_design_eval_data_fp = out_dir / "single_eval_data.json"
                    if best_design_eval_data_fp.exists():
                        best_design_eval_data = json.loads(best_design_eval_data_fp.read_text())
                        best_design_eval_data["opt_dsl_out"] = opt_out
                        self.evaluator._serialize_eval_data(f"{iter_samples_dir.parent.name}_{iter_samples_dir.name}", best_dir, best_design_eval_data)
                        ps = best_design_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None)
                        luts_ps = float(ps.get("LUTs_vs_latency", 1.0)) if ps.get("LUTs_vs_latency", 1.0) is not None else 1.0
                        ffs_ps = float(ps.get("FFs_vs_latency", 1.0)) if ps.get("FFs_vs_latency", 1.0) is not None else 1.0
                       
                        if target == "pareto_scores":
                            # Select best design based on pareto scores; different logic if running Vivado implementation or not
                            candidates.remove((best_dir, best_val_1))
                            if self.evaluator.run_vivado_impl:
                                if (luts_ps == 1.0 and ffs_ps == 1.0) and len(candidates) > 0 and num_tries < 3:                                   
                                    best_dir, best_val_1 = random.choice(candidates)
                                    best_design = Design(best_dir, name=best_dir.name)
                                    continue
                            else:
                                candidates_final.append((best_dir, best_val_1, luts_ps, ffs_ps))
                                if len(candidates) > 0:
                                    best_dir, best_val_1 = random.choice(candidates)
                                    best_design = Design(best_dir, name=best_dir.name)
                                    continue
                                else:
                                    best_dir, best_val_1, luts_ps, ffs_ps = min(candidates_final, key=lambda x: (x[2] + x[3]))
                                    best_design = Design(best_dir, name=best_dir.name)
                        break
                    else:
                        raise ValueError(f"Expected eval data file {best_design_eval_data_fp} not found.")

            best_design = best_design.copy_to(iter_output_dir)
            prev_design = best_design

            for sub in ("output_designs", "raw_data"):
                shutil.rmtree(iter_samples_dir / sub, ignore_errors=True)            

        # After finishing all iterations (or early stopping), write the final design.
        final_output_dir = self.final_designs_dir / f"{seed_design.name}_iter_{iter_idx}"
        prev_design.copy_to(final_output_dir)

        seed_design_eval_data_fp = seed_design.design_dir / "single_eval_data.json"
        seed_design_eval_data = json.loads(seed_design_eval_data_fp.read_text())
        all_summary = {
            "target_list": {
                target_label: iter_nums for target_label, iter_nums in zip(target_list, self.n_iterations if isinstance(self.n_iterations, list) else [self.n_iterations]*len(target_list))
            },
            "model_name": self.model.name,
            "seed_design": {
                "Name": seed_design.name,
                "Path": str(seed_design.design_dir),
                "target_results": {
                    "num_functions": seed_design_eval_data.get("kernel_ast_out", {}).get("num_functions", None),
                    "max_call_chain_depth": seed_design_eval_data.get("kernel_ast_out", {}).get("max_call_chain_depth", None),
                    "average_function_lines": seed_design_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None),
                    **({"pareto_scores": seed_design_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None),}
                    if "pareto_scores" in target_list else {}),                    
                    }
                },
            "iters": {}
        }
        # Summarize all feedback runs of each seed design
        for iter in range(len(targets_plan)):
            iter_dir = root_dir / f"iter_{iter}"
            iter_eval_data_fp = iter_dir / "single_eval_data.json"
            if iter_eval_data_fp.exists():
                iter_entry = all_summary["iters"].setdefault(f"iter_{iter}", {})
                iter_eval_data = json.loads(iter_eval_data_fp.read_text())
                iter_entry["Path"] = str(iter_dir)
                iter_entry["Status"] = iter_eval_data.get("status", None)
                iter_entry["Target"] = targets_plan[iter]
                iter_entry["target_result"] = {
                    "num_functions": iter_eval_data.get("kernel_ast_out", {}).get("num_functions", None),
                    "max_call_chain_depth": iter_eval_data.get("kernel_ast_out", {}).get("max_call_chain_depth", None),
                    "average_function_lines": iter_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None)
                }                
                if "pareto_scores" in target_list:
                    iter_entry["target_result"]["pareto_scores"] = (
                        iter_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None)
                    )
            else:
                break
        (root_dir / "all_data_summary.json").write_text(json.dumps(all_summary, indent=4))

    def run_single(self, iter_samples_dir: Path, sample_idx: int, prev_design: Design, target: str, prev_error_message: Optional[str] = None):
        design_to_improve = prev_design
        if prev_error_message is not None and self.fix:
            sample_output_dir = iter_samples_dir / f"sample_{sample_idx}__fix"
            if sample_output_dir.exists():
                shutil.rmtree(sample_output_dir, ignore_errors=True)
        else:
            sample_output_dir = iter_samples_dir / f"sample_{sample_idx}"
            if sample_output_dir.exists():
                shutil.rmtree(sample_output_dir, ignore_errors=True)

        seed_name = prev_design.name
        iter_eval_data = {}

        prompt = build_prompt_mutate_target(
            design=design_to_improve,
            fix=self.fix,
            error_message=prev_error_message,
            target=target,
        )
        current_error_message, current_design = self.evaluator.evaluate_design(
            model=self.model,
            pools=self.pools,
            output_design_data_dir=sample_output_dir,
            prompt=prompt,
            design_id=f"{iter_samples_dir.parent.name}_{iter_samples_dir.name}_s_{sample_idx}",
            output_format="FULL_CODE",
            full_flow=False,
            seed_design=design_to_improve,
        )
        if current_design is None:
            # Delete the sample output dir if paste failed
            shutil.rmtree(sample_output_dir, ignore_errors=True)
        elif current_error_message is None:
            current_design_eval_data_fp = sample_output_dir / "single_eval_data.json"
            pass_design_dir = iter_samples_dir / "pass_designs" / f"sample_{sample_idx}"
            current_design.copy_to(pass_design_dir)
            shutil.copy(current_design_eval_data_fp, pass_design_dir)

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
        pools: EvalThreadPools,
        n_seed_designs: int = 16,
        n_jobs_seed: int = 8,
        output_mode: str = "FULL_CODE",
        target: Optional[str] = None
    ):
        self.seed_design_dir = seed_design_dir
        self.model = model
        self.evaluator = evaluator
        self.target = target
        self.n_seed_designs = n_seed_designs
        self.pools = pools
        self.output_mode = output_mode
        self.n_jobs_seed = n_jobs_seed

        self.seed_design_names = []

    def run(self, reference_designs: List[Design] = None):
        self.seed_design_names = []
        Parallel(n_jobs=self.n_jobs_seed, backend="threading")(
            delayed(self.run_single)(f"seed_design__{i}", reference_design) for i in range(self.n_seed_designs) for reference_design in (reference_designs if reference_designs else [None])
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
            if self.output_mode == "OPTDSL":
                prompt = build_prompt_gen_optdsl_v2(
                    design_description_fp=reference_design.kernel_description_fp,
                    design_h=reference_design.h_files[0],
                    design_kernel=reference_design.kernel_fp,
                    design_tb=reference_design.tb_file,
                )
            else:
                prompt = build_prompt_gen_zero_shot_single_input_with_opt(
                    design_description_fp=reference_design.kernel_description_fp,
                    design_h=reference_design.h_files[0],
                    design_kernel=reference_design.kernel_fp,
                    design_tb=reference_design.tb_file,
                    design_opt=reference_design.opt_fp,
                    design_pareto_score=reference_design.pareto_scores_fp,
                )
        self.seed_design_names.append(seed_design_name)
        error_message, current_design = self.evaluator.evaluate_design(
            model=self.model,
            pools=self.pools,
            output_design_data_dir=self.seed_design_dir / seed_design_name,
            prompt=prompt,
            design_id=seed_design_name,           
            output_format=self.output_mode,
            full_flow=True if self.target == "pareto_scores" else False,
            seed_design=None if reference_design is None else reference_design,
        )
        if error_message is None:
            pass_designs_dir = self.seed_design_dir / "pass_designs"
            if (pass_designs_dir / seed_design_name).exists() is False:
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
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        models: list[Model],
        temperature: float = 0.7,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None, 
    ):
        self.run_name = run_name
        self.dir_workspace = dir_workspace
        self.models = models
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

        self.vitis_hls_tool_csim = vitis_hls_tool_csim
        self.vitis_hls_tool_synth = vitis_hls_tool_synth

        # Setup evaluation pools and evaluator
        self.pools = EvalThreadPools(
            n_jobs_pool_llm=8,
            n_jobs_pool_csim=12,
            n_jobs_pool_synth=12,
        )
        # DesignEvaluator shared across all models
        self.evaluator = DesignEvaluator(
            vitis_hls_tool_csim=self.vitis_hls_tool_csim,
            vitis_hls_tool_synth=self.vitis_hls_tool_synth,
            temperature=self.temperature,
            clang_path=self.clang_path,
            include_paths=self.include_paths,
        )

    def run(
        self,
        target_list: Optional[list[str]] = None,
        reference_designs: list[Design] = None,  # List of reference designs to base seed designs on
        n_seed_designs: int = 5,
        n_samples: int = 12,
        n_feedback_iterations: int | list[int] = 5,
        n_jobs_design: int = 24,
        n_jobs_hlsfactory: int = 24,
        output_mode: str = "FULL_CODE",
        fix: bool = False,  # Whether to try to fix errors
        run_vivado_impl: bool = True
    ):
        self.evaluator.run_vivado_impl = run_vivado_impl
        self.evaluator.n_jobs_hlsfactory = n_jobs_hlsfactory

        Parallel(n_jobs=len(self.models), backend="threading")(
            delayed(self.run_single_model)(
                model,
                target_list=target_list,
                reference_designs=reference_designs,
                n_seed_designs=n_seed_designs,
                n_samples=n_samples,
                n_feedback_iterations=n_feedback_iterations,
                n_jobs_design=n_jobs_design,
                fix=fix,
                output_mode=output_mode
            ) for model in self.models
        )

    def run_single_model(
        self,
        model: Model,
        target_list: Optional[list[str]],
        reference_designs: list[Design],  # List of reference designs to
        n_seed_designs: int,
        n_samples: int,
        n_feedback_iterations: int | list[int],
        n_jobs_design: int,
        fix: bool,
        output_mode: str
    ):
        if isinstance(n_feedback_iterations, list) and len(n_feedback_iterations) != len(target_list):
            raise ValueError("Length of n_feedback_iterations list must match length of target_list.")

        dir_single_model = self.run_dir / model.name.split("/")[1]
        dir_single_model.mkdir(parents=True, exist_ok=True)

        dir_seed_designs = dir_single_model / "seed_designs"
        dir_seed_designs.mkdir()

        dir_feedback_runs = dir_single_model / "feedback_runs"
        dir_feedback_runs.mkdir()

        dir_final_designs = dir_single_model / "final_designs"
        dir_final_designs.mkdir()

        seed_design_generator = SeedDesignGenerator(
            seed_design_dir=dir_seed_designs,
            model=model,
            evaluator=self.evaluator,
            pools=self.pools,
            n_seed_designs=n_seed_designs,
            n_jobs_seed=n_jobs_design,
            target=target_list[0] if target_list else None,
            output_mode=output_mode
        ) 

        # Generate seed designs either from scratch or based on reference designs
        if reference_designs is not None and len(reference_designs) > 0:
            seed_design_generator.run(reference_designs=reference_designs)
        else:
            seed_design_generator.run()
        seed_designs_dirs = find_design_dirs(dir_seed_designs / "pass_designs")
        seed_designs = [
            Design(d, name=d.name) for d in seed_designs_dirs
        ]
        if len(seed_designs) == 0:
            raise ValueError("No seed designs were generated successfully.")

        if target_list is None:
            shutil.rmtree(dir_feedback_runs, ignore_errors=True)
            shutil.rmtree(dir_final_designs, ignore_errors=True)
            print("No target list provided, generated seed designs only; skipping feedback design loop.")
            return
        
        n_jobs_seed = min(n_jobs_design, len(seed_designs))

        # Run feedback design loop for each target
        feedback_design_loop = FeedbackDesignLoop(
            output_data_dir=dir_feedback_runs,
            final_designs_dir=dir_final_designs,
            model=model,
            evaluator=self.evaluator,
            pools=self.pools,
            n_samples=n_samples,
            n_iterations=n_feedback_iterations,
            n_jobs_sample_per_seed=int(n_jobs_design/n_jobs_seed),
            fix=fix
        )

        Parallel(n_jobs=n_jobs_seed, backend="threading")(
            delayed(feedback_design_loop.run)(seed_design, target_list) for seed_design in seed_designs
        )