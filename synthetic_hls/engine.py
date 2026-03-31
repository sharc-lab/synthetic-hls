import json
import shutil
import random
from pathlib import Path
from joblib import Parallel, delayed
from typing import List, Optional

from synthetic_hls.design import Design, find_design_dirs
from synthetic_hls.llm_models import Model, normalize_model_name
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

    def run(self, seed_design: Design, domain: str, target_list: list[str]):
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
                delayed(self.run_single)(iter_samples_dir, sample_idx, prev_design, domain, target, prev_error_message=None) for sample_idx in range(self.n_samples)
            )
            pass_designs_dirs = find_design_dirs(iter_samples_dir/"pass_designs")

            if len(pass_designs_dirs) == 0 and self.fix:
                samples_to_fix = []
                for sample_idx in range(self.n_samples):
                    prev_design_fix_dir = iter_samples_dir / f"sample_{sample_idx}" 
                    if not (prev_design_fix_dir / "design_generated").exists():
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
                        delayed(self.run_single)(iter_samples_dir, sample_idx, prev_design_fix, domain, target, prev_error_message=prev_error_message) 
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
                        eval_id=f"{iter_samples_dir.parent.name}_{domain}_{iter_samples_dir.name}_s_{best_dir.name.split('_')[-1]}__{self.model_name_normalized}",
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
            "model_name": self.model.name,
            "domain": domain,
            "target_list": {
                target_label: f"{iter_nums} iteration(s)" for target_label, iter_nums in zip(target_list, self.n_iterations if isinstance(self.n_iterations, list) else [self.n_iterations]*len(target_list))
            },
            "num_iterations_run": iter_idx + 1,
            "num_samples_per_iteration": self.n_samples,
            "seed_design": {
                "eval_id": seed_design.name,
                "kernel_name": seed_design_eval_data.get("kernel_name", None),
                "path": str(seed_design.design_dir),
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
        (root_dir / "eval_data_summary.json").write_text(json.dumps(all_summary, indent=4))

    def run_single(self, iter_samples_dir: Path, sample_idx: int, prev_design: Design, domain: str, target: str, prev_error_message: Optional[str] = None):
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
            design_id=f"{iter_samples_dir.parent.name}_{domain}_{iter_samples_dir.name}_s_{sample_idx}",
            output_format="FULL_CODE",
            full_flow=False,
            seed_design=design_to_improve,
        )

        if current_error_message is None and current_design is not None:
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

    def run(self, reference_designs: List[Design] = None, domain: str = None):
        ref_list = reference_designs if reference_designs else [None]
        jobs = (
            delayed(self.run_single)(f"seed_design_{i}", reference_design, domain=domain)
            for i in range(self.n_seed_designs)
            for reference_design in ref_list
        )
        self.seed_design_names = Parallel(n_jobs=self.n_jobs_seed, backend="threading")(jobs)
        
        # Aggregate all eval data from all seed designs
        all_data = {
            "model_name": self.model.name,
            "domain": domain,
            "num_seed_designs": self.n_seed_designs,
            "num_passing_seed_designs": 0
        }
        for seed_design_name in self.seed_design_names:
            output_seed_data_dir = self.seed_design_dir / seed_design_name
            seed_eval_data_fp = output_seed_data_dir / "single_eval_data.json"
            if seed_eval_data_fp.exists():
                all_data[seed_design_name] = json.loads(seed_eval_data_fp.read_text())
                if all_data[seed_design_name].get("status", None) == "Pass":
                    all_data["num_passing_seed_designs"] += 1
        all_data_fp = self.seed_design_dir / "all_eval_data.json"
        all_data_fp.write_text(json.dumps(all_data, indent=4))


    def run_single(self, seed_design_name: str, reference_design: Design = None, domain: str = None) -> str:
        if reference_design is None:
            prompt = build_prompt_gen_zero_shot_no_input_with_opt(domain=domain)
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

        error_message, current_design = self.evaluator.evaluate_design(
            model=self.model,
            pools=self.pools,
            output_design_data_dir=self.seed_design_dir / seed_design_name,
            prompt=prompt,
            design_id=f"{seed_design_name}__{domain}",           
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
        
        return seed_design_name

class SyntheticHLSEngine:
    """
    The main engine to run the full synthetic HLS design generation and improvement loop.
    Creates a workspace structure like:
      <dir_workspace>/<run_name>/
        ├─ <model_name>/              # per-model subdirectory
        │   ├─ <domain_name>/           # per-domain subdirectory
                ├─ seed_designs/            # generated seed designs
                ├─ feedback_runs/           # full per iteration feedback data
                └─ final_designs/           # final improved designs per target and seed
    """
    def __init__(
        self, 
        run_name: str, 
        dir_workspace: Path,
        vitis_hls_dir: Path,
        vivado_dir: Path,
        models: list[Model],
        n_jobs_pool_llm: int = 12,
        n_jobs_pool_csim: int = 24,
        n_jobs_pool_synth: int = 24,
        n_jobs_pool_hlsfactory: int = 24,
        temperature: float = 0.7,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None, 
    ):
        self.run_name = run_name
        self.dir_workspace = dir_workspace
        self.models = models
        self.pools = EvalThreadPools(
            n_jobs_pool_llm=n_jobs_pool_llm,
            n_jobs_pool_csim=n_jobs_pool_csim,
            n_jobs_pool_synth=n_jobs_pool_synth,
            n_jobs_pool_hlsfactory=n_jobs_pool_hlsfactory,
        )
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

        # Setup evaluation evaluator
        self.evaluator = DesignEvaluator(
            vitis_hls_dir=vitis_hls_dir,
            vivado_dir=vivado_dir,
            temperature=self.temperature,
            clang_path=self.clang_path,
            include_paths=self.include_paths,
            pools=self.pools,
        )

    def run(
        self,
        domain_list: Optional[list[str]] = None,
        target_list: Optional[list[str]] = None,
        reference_designs: list[Design] = None,  # List of reference designs to base seed designs on
        n_seed_designs: int = 5,
        n_samples: int = 12,
        n_feedback_iterations: int | list[int] = 5,
        n_jobs_design: int = 24,
        output_mode: str = "FULL_CODE",
        fix: bool = False,  # Whether to try to fix errors
        run_vivado_impl: bool = True
    ):
        self.evaluator.run_vivado_impl = run_vivado_impl

        if domain_list is None or len(domain_list) == 0:
            domain_list = ["ml_ai", "sci_sim", "fin_model"]

        n_jobs_per_model = max(1, int(n_jobs_design / len(self.models)))
        Parallel(n_jobs=len(self.models), backend="threading")(
            delayed(self.run_single_model)(
                model,
                domain_list=domain_list,
                target_list=target_list,
                reference_designs=reference_designs,
                n_seed_designs=n_seed_designs,
                n_samples=n_samples,
                n_feedback_iterations=n_feedback_iterations,
                n_jobs_design=n_jobs_per_model,
                fix=fix,
                output_mode=output_mode
            ) for model in self.models
        )

    def run_single_model(
        self,
        model: Model,
        domain_list: list[str],
        target_list: Optional[list[str]],
        reference_designs: list[Design],  # List of reference designs to
        n_seed_designs: int,
        n_samples: int,
        n_feedback_iterations: int | list[int],
        n_jobs_design: int,
        fix: bool,
        output_mode: str
    ):
        if target_list is not None and len(target_list) > 0:
            if isinstance(n_feedback_iterations, list) and len(n_feedback_iterations) != len(target_list):
                raise ValueError("Length of n_feedback_iterations list must match length of target_list.")

        dir_single_model = self.run_dir / model.name.split("/")[1]
        dir_single_model.mkdir(parents=True, exist_ok=True)

        n_jobs_domain = min(len(domain_list), 10)  # how many domains concurrently for this model
        n_jobs_design_per_domain = max(32, int(n_jobs_design / n_jobs_domain))

        Parallel(n_jobs=n_jobs_domain, backend="threading")(
            delayed(self._run_single_domain)(
                model=model,
                dir_single_model=dir_single_model,
                domain=domain,
                target_list=target_list,
                reference_designs=reference_designs,
                n_seed_designs=n_seed_designs,
                n_samples=n_samples,
                n_feedback_iterations=n_feedback_iterations,
                n_jobs_design=n_jobs_design_per_domain,
                fix=fix,
                output_mode=output_mode,
            ) for domain in domain_list
        )
        
    def _run_single_domain(
        self,
        model: Model,
        dir_single_model: Path,
        domain: str,
        target_list: Optional[list[str]],
        reference_designs: list[Design],
        n_seed_designs: int,
        n_samples: int,
        n_feedback_iterations: int | list[int],
        n_jobs_design: int,
        fix: bool,
        output_mode: str,
    ):
        dir_single_domain = dir_single_model / domain
        dir_single_domain.mkdir(exist_ok=True)

        dir_seed_designs = dir_single_domain / "seed_designs"
        dir_seed_designs.mkdir(exist_ok=True)
        
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
            seed_design_generator.run(reference_designs=reference_designs, domain=domain)
        else:
            seed_design_generator.run(domain=domain)
        seed_designs_dirs = find_design_dirs(dir_seed_designs / "pass_designs")
        seed_designs = [
            Design(d, name=d.name) for d in seed_designs_dirs
        ]

        if len(seed_designs) == 0:
            print(f"No passing seed designs generated for model {model.name} in domain {domain}, skipping feedback design loop.")
            return

        if target_list is None or len(target_list) == 0:
            print("No target list provided, generated seed designs only; skipping feedback design loop.")
            return

        dir_feedback_runs = dir_single_domain / "feedback_runs"
        dir_feedback_runs.mkdir(exist_ok=True)

        dir_final_designs = dir_single_domain / "final_designs"
        dir_final_designs.mkdir(exist_ok=True)
        
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
            n_jobs_sample_per_seed=max(32, int(n_jobs_design/n_jobs_seed)),
            fix=fix
        )

        Parallel(n_jobs=n_jobs_seed, backend="threading")(
            delayed(feedback_design_loop.run)(seed_design, domain, target_list) for seed_design in seed_designs
        )

        all_data_domain = {
            "model_name": model.name,
            "domain": domain,
            "target_list": {
                target_label: f"{iter_nums} iteration(s)" for target_label, iter_nums in zip(target_list, n_feedback_iterations if isinstance(n_feedback_iterations, list) else [n_feedback_iterations]*len(target_list))
            },
            "num_generated_seed_designs": n_seed_designs,
            "num_passing_seed_designs": len(seed_designs),
            "num_samples_per_iteration": n_samples,
            "eval_data_by_seed": {}
        }

        for seed_design in seed_designs:
            seed_feedback_dir = dir_feedback_runs / seed_design.name
            summary_fp = seed_feedback_dir / "eval_data_summary.json"

            all_data_domain["eval_data_by_seed"][seed_design.name] = {}
            if summary_fp.exists():
                data = json.loads(summary_fp.read_text())
                all_data_domain["eval_data_by_seed"][seed_design.name] = {
                    "seed_design": data.get("seed_design", None),
                    "iters": data.get("iters", None),
                }
            else:
                print(f"Expected eval_data_summary.json file not found for {seed_design.name} in domain {domain}.")

        out_fp = dir_single_domain / "domain_eval_data_summary.json"
        out_fp.write_text(json.dumps(all_data_domain, indent=4))