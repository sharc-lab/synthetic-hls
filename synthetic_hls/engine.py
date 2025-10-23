import json
import shutil
from pathlib import Path
from joblib import Parallel, delayed
from typing import List, Optional

from synthetic_hls.design import Design, find_design_dirs
from synthetic_hls.llm_models import Model
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
        n_iterations: int | list[int],
        n_max_versions: int,
        pools: EvalThreadPools,
        fix: bool = True,
    ):
        self.output_data_dir = output_data_dir
        self.final_designs_dir = final_designs_dir
        self.model = model
        self.evaluator = evaluator
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_max_versions = n_max_versions
        self.pools = pools
        self.fix = fix

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
            iter_candidate_dir = root_dir / f"iter_{iter_idx}_candidate"
            prev_sample_eval_data_fp = prev_design.design_dir / "single_eval_data.json"
            prev_sample_eval_data = json.loads(prev_sample_eval_data_fp.read_text())
            pass_designs_dirs = []
            cycle_count = 0
            best_design, best_dir, best_val_1, best_val_2 = None, None, float("-inf"), float("-inf")
            pass_iter = False
            while cycle_count < self.n_max_versions:
                shutil.rmtree(iter_samples_dir, ignore_errors=True)
                print(f"Starting iteration {iter_idx} target {target} cycle {cycle_count} for seed design {seed_design.name}")
                Parallel(n_jobs=4, backend="threading")(
                    delayed(self.run_single)(iter_samples_dir, sample_idx, prev_design, target, prev_error_message=None) for sample_idx in range(self.n_samples)
                )
                pass_designs_dirs = find_design_dirs(iter_samples_dir/"pass_designs")

                if len(pass_designs_dirs) == 0 and self.fix:
                    for sample_idx in range(self.n_samples):
                        prev_design_fix_dir = iter_samples_dir / f"sample_{sample_idx}"
                        if not prev_design_fix_dir.exists():
                            prev_design_fix = seed_design
                            prev_error_message = None
                        else:
                            print(f"Retrying iter {iter_idx} sample {sample_idx} from previous design {prev_design_fix_dir}.")
                            prev_design_fix = Design(prev_design_fix_dir / "design_generated")
                            prev_design_eval_data_fp = prev_design_fix_dir / "single_eval_data.json"
                            if prev_design_eval_data_fp.exists():
                                prev_design_eval_data = json.loads(prev_design_eval_data_fp.read_text())
                                prev_error_message = prev_design_eval_data.get("error_message", None)
                        self.run_single(iter_samples_dir, sample_idx, prev_design_fix, target, prev_error_message=prev_error_message)
                        pass_designs_dirs = find_design_dirs(iter_samples_dir/"pass_designs")
                        if len(pass_designs_dirs) > 0 or cycle_count >= self.n_max_versions:
                            break

                # Evaluate all passing samples and select best
                if len(pass_designs_dirs) > 0:
                    iter_eval_data = {}
                    candidates = []
                    for d in pass_designs_dirs:
                        sample_eval_data_fp = d / "single_eval_data.json"
                        if sample_eval_data_fp.exists():
                            sample_eval_data = json.loads(sample_eval_data_fp.read_text())
                            if target == "pareto_scores":
                                ps = sample_eval_data.get("opt_dsl_out", {}).get("pareto_scores", None)
                                luts_ps = float(ps.get("LUTs_vs_latency", 1.0)) if ps.get("LUTs_vs_latency", 1.0) is not None else 1.0
                                ffs_ps = float(ps.get("FFs_vs_latency", 1.0)) if ps.get("FFs_vs_latency", 1.0) is not None else 1.0
                                candidates.append((d, luts_ps, ffs_ps))
                            else:
                                target_val = float(sample_eval_data.get("kernel_ast_out", {}).get(target, None))
                                avg_func_loc = float(sample_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None))
                                candidates.append((d, target_val, avg_func_loc))
                    
                    if target == "pareto_scores":
                        best_dir, best_val_1, best_val_2 = min(candidates, key=lambda x: (x[1] + x[2]))
                        prev_val_1 = float(prev_sample_eval_data.get("opt_dsl_out", {}).get("pareto_scores", {}).get("LUTs_vs_latency", 1.0))
                        prev_val_2 = float(prev_sample_eval_data.get("opt_dsl_out", {}).get("pareto_scores", {}).get("FFs_vs_latency", 1.0))
                        if best_val_1 == 1.0 and best_val_2 == 1.0:
                            print(f"All candidates in iter {iter_idx} cycle {cycle_count} have invalid Pareto scores, continuing to next cycle.")
                        else:
                            pass_iter = True
                            
                    else:
                        prev_val_1 = float(prev_sample_eval_data.get("kernel_ast_out", {}).get(target, None))
                        prev_val_2 = float(prev_sample_eval_data.get("kernel_ast_out", {}).get("average_function_lines", None))
                        # Filter only candidates that improve target
                        filtered_candidates = [
                            (dir, val1, val2) 
                            for dir, val1, val2 in candidates
                            if val1 > prev_val_1
                        ]
                        if len(filtered_candidates) == 0:
                            if best_design is None:
                                best_val_1, best_val_2 = prev_val_1, prev_val_2
                            current_best_dir, current_best_val_1, current_best_val_2 = max(candidates, key=lambda x: (x[1], x[2]))
                            if current_best_val_1 < best_val_1 or (current_best_val_1 == best_val_1 and current_best_val_2 <= best_val_2):
                                print(f"No candidates improved {target} in iter {iter_idx} cycle {cycle_count}, but current best {best_dir} has value1 {best_val_1} and value2 {best_val_2}. Continuing to next cycle.")
                                cycle_count += 1
                                continue
                            best_dir, best_val_1, best_val_2 = current_best_dir, current_best_val_1, current_best_val_2
                            print(f"No candidates improved {target} in iter {iter_idx} cycle {cycle_count}, but current best {best_dir} has value1 {best_val_1} and value2 {best_val_2}. Continuing to next cycle.")
                        else:
                            pass_iter = True
                            best_dir, best_val_1, best_val_2 = max(filtered_candidates, key=lambda x: (x[2]))
                    if best_design is not None:        
                        shutil.rmtree(best_design.design_dir, ignore_errors=True)        
                    best_design = Design(best_dir, name=best_dir.name)
                    best_design = best_design.copy_to(iter_candidate_dir)

                if pass_iter:
                    break
                cycle_count += 1

            if best_design is None:
                print(f"Iteration {iter_idx} target {target} found no passing designs after {self.n_max_versions} cycles. Using previous design {prev_design.design_dir}.")
                best_design = prev_design.copy_to(iter_candidate_dir)
            else:
                print(f"Iteration {iter_idx} target {target} found best design {best_design.design_dir} with value {best_val_1} and avg_func_loc {best_val_2} after {cycle_count + 1} cycles.")
            
            # If next target is pareto_scores, run pareto scores evaluation for the selected design
            if target_next == "pareto_scores":
                opt_out, out_dir = self.evaluator.run_hlsfactory_flow(
                    design_generated_dir=best_design.design_dir,
                    eval_dir_top=best_design.design_dir.parent,
                    eval_id=f"{iter_samples_dir.parent.name}_{iter_samples_dir.name}",
                    top_function_name=Path(best_design.design_dir / "top.txt").read_text().strip(),
                )
                best_design_eval_data_fp = out_dir / "single_eval_data.json"
                if best_design_eval_data_fp.exists():
                    best_design_eval_data = json.loads(best_design_eval_data_fp.read_text())
                    best_design_eval_data["opt_dsl_out"] = opt_out
                    self.evaluator._serialize_eval_data(f"{iter_samples_dir.parent.name}_{iter_samples_dir.name}", best_design.design_dir, best_design_eval_data)

            best_design = best_design.copy_to(iter_output_dir)
            prev_design = best_design
            shutil.rmtree(iter_samples_dir, ignore_errors=True)
            shutil.rmtree(iter_candidate_dir, ignore_errors=True)

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
            iter_entry = all_summary["iters"].setdefault(f"iter_{iter}", {})
            iter_dir = root_dir / f"iter_{iter}"
            iter_eval_data_fp = iter_dir / "single_eval_data.json"
            if iter_eval_data_fp.exists():
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
            full_flow=True if target == "pareto_scores" else False,
            seed_design=design_to_improve,
        )
        if current_design is None:
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
        n_seed_designs: int,
        pools: EvalThreadPools,
        output_mode: str = "FULL_CODE",
        target: Optional[str] = None,
    ):
        self.seed_design_dir = seed_design_dir
        self.model = model
        self.evaluator = evaluator
        self.target = target
        self.n_seed_designs = n_seed_designs
        self.pools = pools
        self.output_mode = output_mode

        self.seed_design_names = []

    def run(self, reference_designs: List[Design] = None):
        self.seed_design_names = []
        Parallel(n_jobs=4, backend="threading")(
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
            if self.target == "pareto_scores":
                current_design_eval_data_fp = current_design.design_dir / "single_eval_data.json"
                current_design_eval_data = json.loads(current_design_eval_data_fp.read_text())
                current_LUTs_ps = float(current_design_eval_data.get("opt_dsl_out", {}).get("pareto_scores", {}).get("LUTs_vs_latency", 1.0))
                current_FFs_ps = float(current_design_eval_data.get("opt_dsl_out", {}).get("pareto_scores", {}).get("FFs_vs_latency", 1.0))
                if current_LUTs_ps == 1.0 and current_FFs_ps == 1.0:
                    print(f"Generated seed design {seed_design_name} has invalid Pareto scores, skipping.")
                    return
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
        target_list: Optional[list[str]] = None,
        reference_designs: list[Design] = None,  # List of reference designs to base seed designs on
        n_seed_designs: int = 5,
        n_samples: int = 5,
        n_feedback_iterations: int | list[int] = 5,
        n_max_versions: int = 16,
        fix: bool = False,  # Whether to try to fix errors
        output_mode: str = "FULL_CODE"
    ):
        if isinstance(n_feedback_iterations, list) and len(n_feedback_iterations) != len(target_list):
            raise ValueError("Length of n_feedback_iterations list must match length of target_list.")
            
        seed_design_generator = SeedDesignGenerator(
            seed_design_dir=self.dir_seed_designs,
            model=self.model,
            evaluator=self.evaluator,
            n_seed_designs=n_seed_designs,
            pools=self.pools,
            target=target_list[0] if target_list else None,
            output_mode=output_mode
        )
        # Generate seed designs either from scratch or based on reference designs
        max_seed_gen_attempts = 5
        seed_gen_attempt = 0
        input_designs = []
        while seed_gen_attempt < max_seed_gen_attempts:
            if reference_designs is not None and len(reference_designs) > 0:
                seed_design_generator.run(reference_designs=reference_designs)
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

        if target_list is None:
            shutil.rmtree(self.dir_feedback_runs, ignore_errors=True)
            shutil.rmtree(self.dir_final_designs, ignore_errors=True)
            print("No target list provided, generated seed designs only; skipping feedback design loop.")
            return

        # Run feedback design loop for each target
        feedback_design_loop = FeedbackDesignLoop(
            output_data_dir=self.dir_feedback_runs,
            final_designs_dir=self.dir_final_designs,
            model=self.model,
            evaluator=self.evaluator,
            n_samples=n_samples,
            n_iterations=n_feedback_iterations,
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