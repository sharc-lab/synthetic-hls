import os
import shutil
import itertools
import json
import logging
import zipfile
import threading
import time
import random
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import BoundedSemaphore
from pathlib import Path
from llm import Response
from typing import Dict, List, Iterable, Any, Optional
from collections import defaultdict
from statistics import mean

from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool
from synthetic_hls.design import Design
from synthetic_hls.llm_models import Model, normalize_model_name
from synthetic_hls.prompting import approx_num_tokens, extract_code_xml_from_llm_output
from hlsfactory.datasets_builtin import (
    datasets_builder,
)
from hlsfactory.flow_vitis import (
    VitisHLSImplFlow,
    VitisHLSImplReportFlow,
    VitisHLSSynthFlow,
)
from hlsfactory.framework import (
    DesignDataset,
    DesignDatasetCollection,
    count_total_designs_in_dataset_collection,
)
from hlsfactory.opt_dsl_frontend_v2 import OptDSLFrontend
from hlsfactory.opt_dsl_v2.opt_dsl import OptDSL
from hlsfactory.utils import (
    ToolPathsSource,
    get_tool_paths,
    remove_and_make_new_dir_if_exists,
)
from hlsfactory.data_packaging import DataAggregatorXilinx


class EvalThreadPools:
    def __init__(
        self,
        n_jobs_pool_llm: int,
        n_jobs_pool_csim: int,
        n_jobs_pool_synth: int,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
    ) -> None:
        self.n_jobs_pool_llm = n_jobs_pool_llm
        self.n_jobs_pool_csim = n_jobs_pool_csim
        self.n_jobs_pool_synth = n_jobs_pool_synth

        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute

        if n_jobs_pool_llm <= 1:
            raise ValueError("n_jobs_pool_llm must be greater than 1")
        if n_jobs_pool_csim <= 1:
            raise ValueError("n_jobs_pool_csim must be greater than 1")
        if n_jobs_pool_synth <= 1:
            raise ValueError("n_jobs_pool_synth must be greater than 1")

        self.pool_llm = ThreadPoolExecutor(max_workers=n_jobs_pool_llm)
        self.pool_csim = ThreadPoolExecutor(max_workers=n_jobs_pool_csim)
        self.pool_synth = ThreadPoolExecutor(max_workers=n_jobs_pool_synth)

    def shutdown(self):
        self.pool_llm.shutdown(wait=True)
        self.pool_csim.shutdown(wait=True)
        self.pool_synth.shutdown(wait=True)


class ASTAnalyzer:
    """
    Analyze C/C++ source code using Clang to generate an AST in JSON format, extract:
        - `num_functions` (int): number of unique function names currently detected.
        - `max_call_chain_depth` (int): longest path length (in nodes) from the top-level caller to any leaf callee.
        - `functions` (list[str]): names of functions present in the current design.
        - `kernel_total_lines` (int): total lines of code in the kernel.
        - `function_line_counts` (dict[str, int]): per-function lines of code (LOC) measured from the AST.
        - `average_function_lines` (float): mean LOC across functions.
        - `edges` (list[{{"caller": str, "callee": str}}]): directed edges; each means `caller()` invokes `callee()`.
    Output a JSON file named call_graph.json with these metrics.
    """
    def __init__(
        self, 
        source_code_path: Path, 
        output_dir: Path,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None,
        ) -> None:
        self.source_code_path = Path(source_code_path)
        self.output_dir = Path(output_dir)
        self.clang_path = Path(clang_path) if clang_path else None
        self.include_paths = [Path(p) for p in include_paths] if include_paths else None
        self.ast_json_path = self.output_dir / "kernel_ast.json"
        self._data = None
        self._funcs: Dict[str, dict] = {}
        self._graph: Dict[str, List[str]] = {}
        self._metrics: Dict[str, Any] = {}
        self._func_line_counts: Dict[str, int] = {}

    def generate_ast_json(self) -> None:
        include_args: list[str] = []
        if self.include_paths:
            include_args = list(
                itertools.chain.from_iterable(
                    ("-isystem", str(p)) for p in self.include_paths if p.exists()
                )
            )
        command = [
            f"{self.clang_path}" if self.clang_path else "clang++",
            "-fsyntax-only", 
            "-Xclang", 
            "-ast-dump=json",
            *include_args,
            f"{self.source_code_path}",
            ]
        with open(self.ast_json_path, "w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.PIPE, check=True)

    def filter_ast_only_source_file(self) -> None:
        new_inner = []
        first_occurrence_of_main_file = False
        json_ast = json.loads(self.ast_json_path.read_text())
        for entry in json_ast['inner']:
            if not first_occurrence_of_main_file:
                if entry.get('isImplicit', False):
                    continue

                file_name = None
                loc = entry.get('loc', {})
                if 'file' in loc:
                    file_name = loc['file']

                if 'expansionLoc' in loc:
                    if 'file' in loc['expansionLoc']:
                        file_name = loc['expansionLoc']['file']

                if file_name != f"{self.source_code_path}":
                    continue

                new_inner.append(entry)
                first_occurrence_of_main_file = True
            else:
                new_inner.append(entry)
        json_ast['inner'] = new_inner 
        json.dump(json_ast, open(self.ast_json_path, 'w'), indent=2)

    def analyze(self) -> None:
        self.generate_ast_json()
        self.filter_ast_only_source_file()
        self._data = json.loads(self.ast_json_path.read_text())
        self._funcs = self._collect_functions(self._data)
        self._graph = self._build_call_graph(self._funcs)

        self._func_line_counts = self._compute_function_line_counts(self._funcs)
        lines_list = list(self._func_line_counts.values())
        avg_lines = (sum(lines_list) / len(lines_list)) if lines_list else 0.0
        depth = self._max_call_depth(self._graph)

        self._metrics = {
            "num_functions": len(self._funcs),
            "max_call_chain_depth": depth,
            "functions": sorted(self._funcs.keys()),
            "kernel_total_lines": sum(lines_list),            
            "function_line_counts": self._func_line_counts,
            "average_function_lines": round(avg_lines, 2),
            "edges": [
                {"caller": caller, "callee": callee}
                for caller, callees in self._graph.items()
                for callee in callees
            ],
        }

    def to_json(self) -> None:
        out_path = self.output_dir / "call_graph.json"
        out_path.write_text(json.dumps(self._metrics, indent=2))
        self.ast_json_path.unlink(missing_ok=True)

    # internal methods
    @staticmethod
    def _walk(node: dict) -> Iterable[dict]:
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            inner = n.get("inner") or []
            stack.extend(reversed(inner))

    def _collect_functions(self, data: dict) -> Dict[str, dict]:
        funcs = {}
        for n in self._walk(data):
            if n.get("kind") == "FunctionDecl" and bool(n.get("name")):
                funcs[n["name"]] = n
        return funcs

    @staticmethod
    def _function_loc_lines(fn_node: dict) -> Optional[int]:
        loc = fn_node.get("loc") or {}
        rng = fn_node.get("range") or {}
        begin = rng.get("begin") or {}

        decl_line = loc.get("line") or begin.get("line")
        end_line = (rng.get("end") or {}).get("line")

        if isinstance(decl_line, int) and isinstance(end_line, int) and end_line >= decl_line:
            return end_line - decl_line + 1
        return None

    def _compute_function_line_counts(self, funcs: Dict[str, dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name, node in funcs.items():
            loc_lines = self._function_loc_lines(node)
            if isinstance(loc_lines, int):
                counts[name] = loc_lines
        return counts

    def _find_callees_in_fn(self, fn_node: dict, func_whitelist: set[str]) -> List[str]:
        callees = set()
        for n in self._walk(fn_node):
            kind = n.get("kind")
            if kind == "DeclRefExpr":
                ref = n.get("referencedDecl") or {}
                ref_kind = ref.get("kind")
                if ref_kind in ("FunctionDecl", "CXXMethodDecl"):
                    name = ref.get("name")
                    if name and ((not name.startswith("operator")) and (name in func_whitelist)):
                        callees.add(name)
            elif kind in ("MemberExpr", "CXXMemberCallExpr"):
                ref = n.get("referencedDecl") or {}
                if ref.get("kind") == "CXXMethodDecl":
                    name = ref.get("name")
                    if name and ((not name.startswith("operator")) and (name in func_whitelist)):
                        callees.add(name)
        return sorted(callees)

    def _build_call_graph(self, funcs: Dict[str, dict]) -> Dict[str, List[str]]:
        graph = defaultdict(list)
        whitelist = set(funcs.keys())
        for name, node in funcs.items():
            graph[name] = self._find_callees_in_fn(node, whitelist)
        return graph

    @staticmethod
    def _max_call_depth(graph: Dict[str, List[str]]) -> int:
        seen: Dict[str, int] = {}
        visiting: set[str] = set()

        def dfs(f: str) -> int:
            if f in seen:
                return seen[f]
            if f in visiting:
                return 1
            visiting.add(f)
            children = graph.get(f, [])
            seen[f] = 1 + max((dfs(c) for c in children), default=0)
            visiting.remove(f)
            return seen[f]

        return max((dfs(f) for f in graph), default=0)


class HLSFactoryFlow:
    """
    Run HLSFactory flow to synthesize, implement and evaluate designs:
      1) OptDSL frontend sampling (design space instantiation)
      2) Vitis HLS synthesis + implementation
      3) Vivado reporting
      4) Pareto scoring and summary export
    Output a pareto_scores.txt file and pareto_scores_summary.json file with detailed metrics.
    """
    def __init__(
        self,
        design_dir: Path,
        work_dir: Path,
        n_random_samples: int = 64,
        random_sample_seed: int = 64,
        n_jobs: int = 6,
    ):
        self.design_dir = design_dir
        self.dataset_dir = design_dir.parent
        self.work_dir = work_dir
        self.vitis_hls_dir, self.vivado_dir = get_tool_paths(tool_paths_source=ToolPathsSource.ENVFILE)
        self.vitis_hls_bin = self.vitis_hls_dir / "bin" / "vitis_hls"
        self.vivado_bin = self.vivado_dir / "bin" / "vivado"
        self.n_random_samples = n_random_samples
        self.random_sample_seed = random_sample_seed
        self.n_jobs = n_jobs

    def pareto_frontier(self, df, x_col, y_col) -> pd.DataFrame:
        if y_col not in df.columns or x_col not in df.columns:
            return pd.DataFrame()
        df = df.sort_values(by=[x_col, y_col], ascending=[True, False])  # Sort by x (asc) and y (desc)
        df = df.dropna(subset=[x_col, y_col])
        pareto_points = []
        current_best_y = float("inf")
        for _, row in df.iterrows():
            if row[y_col] < current_best_y:  # Keep points that have better y (lower latency)
                pareto_points.append(row)
                current_best_y = row[y_col]

        return pd.DataFrame(pareto_points)        

    def delta_gap_based(self, df, x_col, y_col) -> float:
        pareto_df = self.pareto_frontier(df, x_col, y_col)

        if pareto_df.empty or len(pareto_df) < 2:
            summary = {
                "pareto_score": None,
                "n_points": int(len(df)),
                "n_pareto_frontier_points": int(len(pareto_df)),
            }
            return None, summary

        pareto_df = pareto_df.copy()
        pareto_df[x_col] = (pareto_df[x_col] - df[x_col].min()) / (df[x_col].max() - df[x_col].min())
        pareto_df[y_col] = (pareto_df[y_col] - df[y_col].min()) / (df[y_col].max() - df[y_col].min())

        pareto_df = pareto_df.sort_values(by=x_col).reset_index(drop=True)
        points = pareto_df[[x_col, y_col]].values

        y_star = np.array([[0.0, 1.0], [1.0, 0.0]])
        d_ext = [np.min(np.linalg.norm(points - y_s, axis=1)) for y_s in y_star]
        sum_d_ext = np.sum(d_ext)

        inter_dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
        max_idx = int(np.argmax(inter_dists))
        max_gap = float(np.max(inter_dists))
        point1, point2 = points[max_idx], points[max_idx + 1]
        curve_length = np.sum(inter_dists)

        numerator = sum_d_ext + max_gap
        denominator = sum_d_ext + curve_length  # normalize against unit scale
        score = float(numerator / denominator)
        summary = {
            "pareto_score": score,
            "n_points": int(len(df)),
            "n_pareto_frontier_points": int(len(pareto_df)),
            "resource_range": [float(df[x_col].min()), float(df[x_col].max())],
            "latency_range": [float(df[y_col].min()), float(df[y_col].max())],
            "start_point_to_corner": float(d_ext[0]),
            "end_point_corner": float(d_ext[1]),
            "max_gap/curve_length": float(max_gap/curve_length) if curve_length > 0 else None,
            "max_gap_points": (point1.tolist(), point2.tolist()),
        }
        return score, summary

    def analyze(self, design_generated_dir:Path, design_dir:Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = self.dataset_dir.name
        dataset = DesignDataset.from_dir(dataset_name, self.work_dir / f"{dataset_name}__post_frontend" )
        designs = dataset.designs

        xilinx_aggregator = DataAggregatorXilinx()

        data = xilinx_aggregator.gather_multiple_designs(designs, n_jobs=16)
        output_archive_fp = output_dir / f"{dataset_name}.zip"
        xilinx_aggregator.aggregated_data_to_archive(
            data,
            output_archive_fp,
            )
        print(output_archive_fp)

        data_all_zip_fp = "data_all.csv"
        with zipfile.ZipFile(output_archive_fp, "r") as z, z.open(data_all_zip_fp) as f:
            data_all = pd.read_csv(f)

        resource_map = {
            "LUTs": "implementation__utilization__Total LUTs",
            "FFs": "implementation__utilization__FFs",
        }

        all_summary = {}
        with open(design_dir / "pareto_scores.txt", "a") as f:
            for resource_name, resource_column in resource_map.items():
                pareto_delta_score, current_summary = self.delta_gap_based(
                    data_all, resource_column, "synthesis__latency_average_cycles"
                )
                all_summary[f"{resource_name}_vs_latency"] = current_summary
                f.write(f"pareto_score_{resource_name}_vs_latency = {pareto_delta_score}\n")

        with open(design_dir / "pareto_scores_summary.json", "w") as jf:
            json.dump(all_summary, jf, indent=4)

        # shutil.copy(design_dir / "pareto_scores.txt", design_generated_dir / "pareto_scores.txt")
        shutil.copy(design_dir / "pareto_scores_summary.json", design_generated_dir / "pareto_scores_summary.json")

    def opt_dsl_check(self) -> bool:
        opt_template_fp = self.design_dir / "opt_template.tcl"

        with open(opt_template_fp) as file:
            opt_dsl = OptDSL(file.read())
        
        return opt_dsl.opt_dsl_error, opt_dsl.error_message
    
    def run(self):
        datasets: DesignDatasetCollection = {}

        dataset_name = self.dataset_dir.name
        designs = DesignDataset.from_dir(
            dataset_name,
            self.dataset_dir,
        ).copy_dataset(self.work_dir)
        datasets[dataset_name] = designs

        TIMEOUT_FRONTEND = 60.0 * 8
        opt_dsl_frontend = OptDSLFrontend(
            self.work_dir,
            random_sample=True,
            random_sample_num=self.n_random_samples,
            random_sample_seed=self.random_sample_seed,
            log_execution_time=True,
        )

        datasets_post_frontend = (
            opt_dsl_frontend.execute_multiple_design_datasets_fine_grained_parallel(
                datasets,
                True,
                lambda x: f"{x}__post_frontend",
                n_jobs=self.n_jobs,
                cpu_affinity=list(range(self.n_jobs)),
                timeout=TIMEOUT_FRONTEND,
            )
        )
        TIMEOUT_HLS_SYNTH = 60.0 * 32
        TIMEOUT_HLS_IMPL = 60.0 * 90

        toolflow_vitis_hls_synth = VitisHLSSynthFlow(
            vitis_hls_bin=str(self.vitis_hls_bin),
            env_var_xilinx_hls=str(self.vitis_hls_dir),
            env_var_xilinx_vivado=str(self.vivado_dir),
        )
        datasets_post_hls_synth = (
            toolflow_vitis_hls_synth.execute_multiple_design_datasets_fine_grained_parallel(
                datasets_post_frontend,
                False,
                n_jobs=self.n_jobs,
                cpu_affinity=list(range(self.n_jobs)),
                timeout=TIMEOUT_HLS_SYNTH,
            )
        )

        toolflow_vitis_hls_implementation = VitisHLSImplFlow(
            vitis_hls_bin=str(self.vitis_hls_bin),
            env_var_xilinx_hls=str(self.vitis_hls_dir),
            env_var_xilinx_vivado=str(self.vivado_dir),
        )
        datasets_post_hls_implementation = toolflow_vitis_hls_implementation.execute_multiple_design_datasets_fine_grained_parallel(
            datasets_post_hls_synth,
            False,
            n_jobs=self.n_jobs,
            cpu_affinity=range(self.n_jobs),
            timeout=TIMEOUT_HLS_IMPL,
        )

        ### Vivado Reporting Flow ###
        toolflow_vitis_hls_impl_report = VitisHLSImplReportFlow(
            vitis_hls_bin=str(self.vitis_hls_bin),
            vivado_bin=str(self.vivado_bin),
            env_var_xilinx_hls=str(self.vitis_hls_dir),
            env_var_xilinx_vivado=str(self.vivado_dir),
        )
        toolflow_vitis_hls_impl_report.execute_multiple_design_datasets_fine_grained_parallel(
            datasets_post_hls_implementation,
            False,
            n_jobs=self.n_jobs,
            cpu_affinity=range(self.n_jobs),
        )


class DesignEvaluator:
    """
    Generate designs using LLM, evaluate them using Vitis HLS toolflow, AST analyzer and HLSFactory flow.
    """
    FULL_FLOW_LOCK = threading.Lock()
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        template_files_path: Path,
        temperature: float = 0.7,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None,
    ) -> None:
        self.cpp_compiler_tool = vitis_hls_tool_csim
        self.vitis_hls_tool = vitis_hls_tool_synth
        self.template_files_path = Path(template_files_path)
        self.temperature = temperature
        self.clang_path = clang_path
        self.include_paths = include_paths
        self.logger = logging.getLogger(__name__)

    def _serialize_eval_data(self, eval_id: str, eval_output_dir: Path, single_eval_data: dict):
        print(f"[{eval_id}] Saving eval data to json...")
        single_eval_data_json = json.dumps(single_eval_data, indent=4)
        (eval_output_dir / "single_eval_data.json").write_text(str(single_eval_data_json))

    # Generate error message based on evaluation data of previous design
    def _generate_error_message(self, prev_design_eval_data: dict) -> str:
        prev_design_syn_data = prev_design_eval_data["vitis_hls_tool_out"]["data_execution"]
        if prev_design_eval_data["status"] == "Pass":
            return None
        else:
            c = ""
            if prev_design_eval_data["opt_dsl_out"]["error"] is not None:
                return ("OptDSL Error")
            if (
                "timeout" in prev_design_syn_data
                and prev_design_syn_data["timeout"] is True
            ):
                e = ""
                e += "The generated code could not be synthesized with Vitis HLS. Please fix the issue and regenerate the corrected code.\n Also regenerate the updated OptDSLv2 optimization template file `opt_template.tcl` file that matches the corrected kernel structure and defines the proper design space.\n"
                e += c 
                e += "Error:\n"
                e += "The synthesis process timed out under the user defined timeout limit for HLS synthesis.\n"
                e += "This can be due to the way the code is written in combination with specific pragma settings and options that cause HLS scheduling and binding to take an unreasonably long time.\n"
                e += "The main cause of this issue is the use of #prgama HLS PIPELINE (with or without II=1) when the code can not be pipelined or has nontrivial loop or carry dependencies.\n"
                e += "If this is the case, remove the pragma HLS PIPELINE pragmas and try again.\n"
                e += "Consider the following suggestions:\n"
                e += "- Manually refactor the code and check pragma settings to improve the parallelism and data locality to reduce the scheduling and binding time of the HLS tool.\n"
                e += "- Check for instances that can cause a high II (iteration interval) for code that will be pipelined.\n"
                e += "- Avoid naively pipelining code with PIPELINE or PIPELINE II=1, as this can cause a high II if a loop or function body can not be easly pipelined or have nontrivial loop or carry dependencies.\n"
                e += "- Check the generated code for any other potential issues that may be causing the synthesis process to take an unreasonably long time.\n"
                return e

            synth_log = prev_design_syn_data["stdout"]
            error_lines = [
                line for line in synth_log.split("\n") if line.startswith("ERROR: ")
            ]
            return (
                "The generated code could not be synthesized with Vitis HLS. Please fix the issue and regenerate the corrected code. \nError Messages:\n"
                + c
                + "\n".join(error_lines)
            )

    def evaluate_design(
        self,
        model: Model,
        pools: EvalThreadPools,
        output_design_data_dir: Path,
        prompt: str,
        design_id: str = "",       
        output_format: str = "FULL_CODE",
        full_flow: bool = False,
        seed_design: Design | None = None,
        **kwargs,
    ):
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)

        eval_data: dict[str, Any] = {}

        eval_id = design_id
        eval_data["eval_type"] = "hls_gen_zero_shot"
        eval_data["eval_id"] = eval_id
        eval_data["status"] = "Fail"
        eval_data["model_name"] = model_name
        eval_data["model_name_normalized"] = model_name_normalized
        eval_data["temperature"] = self.temperature

        self.logger.info(f"[{eval_id}] Running eval...")

        eval_dir = output_design_data_dir
        eval_dir_top = output_design_data_dir.parent

        if eval_dir.exists():
            self.logger.info(f"Removing existing sample eval dir: {eval_dir}")
            shutil.rmtree(eval_dir)
        eval_dir.mkdir(parents=True)

        if seed_design is not None:
            eval_data["seed_design_name"] = seed_design.name
            eval_data["seed_design_tags"] = seed_design.tags_all
            design_dir = eval_dir / "design"
            seed_design.copy_to(design_dir)

        eval_data["prompt"] = prompt
        (eval_dir / "raw_llm_prompt.txt").write_text(prompt)

        n_tokens_guess = approx_num_tokens(prompt)
        llm_pool = pools.pool_llm
        llm = model.llm
        t0 = time.monotonic()

        # A stronger version of call_model with retries
        def call_model(
            prompt
        ) -> tuple[
            Response | None,
            str | None,
            dict | None,
            bool,
            bool,
            float,
            float,
            float,
        ]:
            print(f"[{eval_id}] Calling model...")
            print(f"[{eval_id}] Waiting for {n_tokens_guess} tokens")
            t_0 = time.monotonic()

            max_retries = 3
            base_sleep = 1.5
            r: Response | None = None
            r_text: str | None = None
            r_json: dict | None = None
            model_timeout = False
            prompt_too_long = False

            for attempt in range(max_retries):
                try:
                    r = llm.prompt(
                        prompt=prompt,
                        stream=False,
                        temperature=self.temperature,
                    )
                    if hasattr(r, "_force"):
                        try:
                            r._force()
                        except Exception:
                            pass
                    r_text = r.text()
                    try:
                        r_json = r.json()
                    except Exception:
                        r_json = None
            
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    model_timeout = False
                    prompt_too_long = False

                    return r, r_text, r_json, model_timeout, prompt_too_long, t_0, t1, dt
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.25)
                        time.sleep(sleep_s)
                        continue
                    model_timeout = True
                    print(f"[{eval_id}] Model call failed after {max_retries} attempts. Error: {e}")
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    return None, None, None, model_timeout, prompt_too_long, t_0, t1, dt
                    
            t1 = time.monotonic()
            dt = t1 - t_0
            model_timeout = True
            print(f"[{eval_id}] Model call failed after {max_retries} attempts.")
            return None, None, None, model_timeout, prompt_too_long, t_0, t1, dt  # treat as timeout

        future_llm = llm_pool.submit(call_model, prompt)
        r, r_text, r_json, model_timeout, prompt_too_long, t0, t1, dt = (
            future_llm.result()
        )

        eval_data["model_timeout"] = model_timeout
        eval_data["prompt_too_long"] = prompt_too_long
        eval_data["llm_execution_time"] = {"t0": t0, "t1": t1, "execution_time": dt}

        if model_timeout or prompt_too_long:
            self._serialize_eval_data(eval_id, eval_dir, eval_data)
            return "ModelTimeout" if model_timeout else "PromptTooLong", None

        assert r is not None
        assert r_text is not None

        if r.response_json is not None:
            eval_data["response_json"] = r.response_json

        eval_data["raw_output"] = str(r_text)
        (eval_dir / "raw_llm_output.txt").write_text(data=r_text)

        print(f"[{eval_id}] Extracting code from output...")

        try:
            generated_code = extract_code_xml_from_llm_output(r_text)
            if output_format == "FULL_CODE":
                assert len(generated_code) == 7
                assert len([k for k in generated_code.keys() if k.endswith(".h")]) == 1
                assert (
                    len([k for k in generated_code.keys() if k.endswith(".cpp")]) == 2
                )
                assert (
                    len([k for k in generated_code.keys() if k.endswith("_tb.cpp")])
                    == 1
                )
                assert(
                    len([k for k in generated_code.keys() if k.endswith(".txt")]) == 1
                )
                assert(
                    len([k for k in generated_code.keys() if k.endswith(".md")]) == 1
                )
                assert (
                    len([k for k in generated_code.keys() if k.endswith(".tcl")]) == 1
                )
                assert(
                    len([k for k in generated_code.keys() if k.endswith(".toml")]) == 1
                )
            elif output_format == "OPTDSL":
                assert len(generated_code) == 1
                assert len([k for k in generated_code.keys() if k.endswith(".tcl")]) == 1
            else:
                raise ValueError(f"Unknown output_format: {output_format}")
            eval_data["generated_code"] = generated_code
            eval_data["can_parse_output"] = True
        except Exception:
            print(f"[{eval_id}] Error extracting code from LLM output")
            eval_data["can_parse_output"] = False
            self._serialize_eval_data(eval_id, eval_dir, eval_data)
            return "ParseError", None

        design_generated_dir: Path = eval_dir / "design_generated"
        design_generated_dir.mkdir()

        if output_format == "OPTDSL":    
            shutil.copytree(design_dir, design_generated_dir, dirs_exist_ok=True)
            for f in design_generated_dir.glob("*"):
                if f.name == "opt_template.tcl":
                    f.unlink()

        for file_name, code in generated_code.items():
            (design_generated_dir / f"{file_name}").write_text(code)

        build_dir = eval_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        build_dir_source_files = sorted(
            list(design_generated_dir.glob("*.cpp"))
            + list(design_generated_dir.glob("*.h"))
        )
        build_dir_not_source_files: list[Path] = sorted(
            list(set(design_generated_dir.glob("*")) - set(build_dir_source_files))
        )

        pool_csim = pools.pool_csim

        print(f"[{eval_id}] Compiling and running the LLM version of the design...")

        future_tool_cpp = pool_csim.submit(
            self.cpp_compiler_tool.run,
            build_dir,
            build_dir_source_files,
            build_dir_not_source_files,
            eval_id,
        )

        c_compile_out, c_run_out = future_tool_cpp.result()

        eval_data["c_compile_out"] = {}
        eval_data["c_compile_out"]["data_execution"] = {
            "return_code": c_compile_out.data_execution.return_code,
            "stdout": c_compile_out.data_execution.stdout,
            "stderr": c_compile_out.data_execution.stderr,
            "t0": c_compile_out.data_execution.t0,
            "t1": c_compile_out.data_execution.t1,
            "execution_time": c_compile_out.data_execution.execution_time,
            "timeout": c_compile_out.data_execution.timeout,
        }

        print(
            f"[{eval_id}] Testbench compile return code: {c_compile_out.data_execution.return_code}"
        )

        if c_run_out:
            eval_data["c_run_out"] = {}
            eval_data["c_run_out"]["data_execution"] = {
                "return_code": c_run_out.data_execution.return_code,
                "stdout": c_run_out.data_execution.stdout,
                "stderr": c_run_out.data_execution.stderr,
                "t0": c_run_out.data_execution.t0,
                "t1": c_run_out.data_execution.t1,
                "execution_time": c_run_out.data_execution.execution_time,
                "timeout": c_run_out.data_execution.timeout,
            }

            print(
                f"[{eval_id}] Testbench return code: {c_run_out.data_execution.return_code}"
            )

        pool_synth = pools.pool_synth

        print(f"[{eval_id}] Synthesizing the LLM version of the design...")
        generated_top_file = design_generated_dir / "top.txt"
        top_function_name = generated_top_file.read_text().strip()

        future_tool_hls = pool_synth.submit(
            self.vitis_hls_tool.run,
            build_dir,
            build_dir_source_files,
            build_name=eval_id,
            hls_top_function=top_function_name,
        )
        vitis_hls_tool_output = future_tool_hls.result()

        eval_data["vitis_hls_tool_out"] = {}
        eval_data["vitis_hls_tool_out"]["data_execution"] = {
            "return_code": vitis_hls_tool_output.data_execution.return_code,
            "stdout": vitis_hls_tool_output.data_execution.stdout,
            "stderr": vitis_hls_tool_output.data_execution.stderr,
            "t0": vitis_hls_tool_output.data_execution.t0,
            "t1": vitis_hls_tool_output.data_execution.t1,
            "execution_time": vitis_hls_tool_output.data_execution.execution_time,
            "timeout": vitis_hls_tool_output.data_execution.timeout,
        }
        eval_data["vitis_hls_tool_out"]["data_tool"] = {}
        if vitis_hls_tool_output.data_tool:
            for k, v in vitis_hls_tool_output.data_tool.items():
                eval_data["vitis_hls_tool_out"]["data_tool"][k] = v
        print(
            f"[{eval_id}] Vitis HLS return code: {vitis_hls_tool_output.data_execution.return_code}"
        )
        
        eval_data["kernel_ast_out"] = {}
        eval_data["opt_dsl_out"] = {}
        eval_data["opt_dsl_out"]["error"] = None
        eval_data["opt_dsl_out"]["pareto_scores"] = {}
        error_message = None
        if vitis_hls_tool_output.data_execution.return_code == 0:
            kernel_name = (next(design_generated_dir.glob("*.h"), None)).stem
            output_design_dir = eval_dir_top / "output_designs" / eval_id / kernel_name

            for f in design_generated_dir.glob("*.cpp"):
                if not f.name.endswith("_tb.cpp"):
                    print(f"[{eval_id}] Analyzing AST for kernel file: {f.name}...")
                    kernel_ast_analyzer = ASTAnalyzer(
                        source_code_path=f,
                        output_dir=design_generated_dir,
                        clang_path=self.clang_path,
                        include_paths=self.include_paths,
                    )
                    kernel_ast_analyzer.analyze()
                    kernel_ast_analyzer.to_json()
                    print(f"[{eval_id}] Saving AST analysis results to eval data...")
                    eval_data["kernel_ast_out"] = json.loads((design_generated_dir / "call_graph.json").read_text())
                
                                        
            current_design = Design(design_generated_dir, name=f"{eval_id}")
            output_design = current_design.copy_to(output_design_dir)

            src_dir = output_design_dir / "src"
            tb_dir = output_design_dir / "tb"
            src_dir.mkdir(parents=True, exist_ok=True)
            tb_dir.mkdir(parents=True, exist_ok=True)

            for f in output_design_dir.glob("*.cpp"):
                if not f.name.endswith("_tb.cpp"):
                    shutil.move(str(f), src_dir)
                else:
                    shutil.move(str(f), tb_dir)

            for f in output_design_dir.glob("*.h"):
                shutil.move(str(f), src_dir)

            tcl_files_dir = self.template_files_path
            for tcl in tcl_files_dir.iterdir():
                if tcl.is_file():
                    shutil.copy(tcl, output_design_dir)

            hls_template = output_design_dir / "hls_template.tcl"
            content = hls_template.read_text()
            content = content.replace("[top_function_name]", top_function_name)    
            content = content.replace("[kernel_name]", kernel_name)
            hls_template.write_text(content)

            work_dir = eval_dir_top / "raw_data"
            remove_and_make_new_dir_if_exists(work_dir)
            design_hlsfactory_flow = HLSFactoryFlow(
                design_dir = output_design_dir,
                work_dir = work_dir,
                n_random_samples = 64,
                random_sample_seed = 64,
                n_jobs = 8,
            )

            # Check OptDSL template files for errors
            opt_dsl_error, opt_dsl_error_message = design_hlsfactory_flow.opt_dsl_check()
            if opt_dsl_error:
                print(f"[{eval_id}] OptDSL error found: {opt_dsl_error_message}")
                eval_data["opt_dsl_out"]["error"] = opt_dsl_error_message
                shutil.rmtree(output_design_dir.parent)
            else:
                eval_data["status"] = "Pass"
                if full_flow:
                    with DesignEvaluator.FULL_FLOW_LOCK:
                        print(f"[{eval_id}] Running full HLSFactory flow...")
                        design_hlsfactory_flow.run()
                        design_hlsfactory_flow.analyze(
                            design_generated_dir=design_generated_dir,
                            design_dir=output_design_dir,
                            output_dir=eval_dir_top / "zip_data" / eval_id,
                        )
                        print(f"[{eval_id}] Analyzing and saving pareto scores...")
                        pareto_scores_summary = json.loads((output_design_dir / "pareto_scores_summary.json").read_text())
                        eval_data["opt_dsl_out"]["pareto_scores"]["LUTs_vs_latency"] = pareto_scores_summary["LUTs_vs_latency"]["pareto_score"]
                        eval_data["opt_dsl_out"]["pareto_scores"]["FFs_vs_latency"] = pareto_scores_summary["FFs_vs_latency"]["pareto_score"]

        self._serialize_eval_data(eval_id, eval_dir, eval_data)
        error_message = self._generate_error_message(eval_data)
        final_output_design = Design(design_generated_dir, name=f"{eval_id}")

        return error_message, final_output_design
