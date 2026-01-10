import os
import io
import shutil
import json
import zipfile
import subprocess
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Iterable, Optional
from collections import defaultdict
from statistics import mean

from hlsfactory.datasets_builtin import (
    datasets_builder,
)
from hlsfactory.flow_vitis import (
    VitisHLSImplFlow,
    VitisHLSImplReportFlow,
    VitisHLSSynthFlow,
)
from hlsfactory.framework import (
    Design,
    DesignDataset,
    DesignDatasetCollection,
    count_total_designs_in_dataset_collection,
)
from hlsfactory.opt_dsl_frontend_v2 import OptDSLFrontend
from hlsfactory.opt_dsl_v2.opt_dsl import OptDSL
from hlsfactory.utils import remove_and_make_new_dir_if_exists
from hlsfactory.data_packaging import DataAggregatorXilinx, CompleteHLSData

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
        with open(self.ast_json_path, "r") as rf: 
            json_ast = json.load(rf)   

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
        with open(self.ast_json_path, "w") as wf:    
            json.dump(json_ast, wf, indent=2)          

    def analyze_to_json(self) -> None:
        """
        Generate, filter, analyze AST and write call_graph.json in one pass,
        without storing large intermediates on self to avoid memory retention.
        """
        self.generate_ast_json()
        self.filter_ast_only_source_file()

        with open(self.ast_json_path, "r") as f:
            data = json.load(f)

        funcs = self._collect_functions(data)
        graph = self._build_call_graph(funcs)

        func_line_counts = self._compute_function_line_counts(funcs)
        lines_list = list(func_line_counts.values())
        avg_lines = (sum(lines_list) / len(lines_list)) if lines_list else 0.0
        depth = self._max_call_depth(graph)

        metrics = {
            "num_functions": len(funcs),
            "max_call_chain_depth": depth,
            "functions": sorted(funcs.keys()),
            "kernel_total_lines": sum(lines_list),
            "function_line_counts": func_line_counts,
            "average_function_lines": round(avg_lines, 2),
            "edges": [
                {"caller": caller, "callee": callee}
                for caller, callees in graph.items()
                for callee in callees
            ],
        }

        out_path = self.output_dir / "call_graph.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        try:
            self.ast_json_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Clear large locals and nudge GC (optional but helpful in long runs)
        del data, funcs, graph, func_line_counts, lines_list, metrics
        import gc; gc.collect()

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
        vitis_hls_dir: Path,
        vivado_dir: Path,
        n_random_samples: int = 64,
        random_sample_seed: int = 64,
        n_jobs: int = 8,
        run_vivado_impl: bool = True,
    ):
        self.design_dir = design_dir
        self.dataset_dir = design_dir.parent
        self.work_dir = work_dir
        self.vitis_hls_dir = vitis_hls_dir
        self.vivado_dir = vivado_dir
        self.vitis_hls_bin = self.vitis_hls_dir / "bin" / "vitis_hls"
        self.vivado_bin = self.vivado_dir / "bin" / "vivado"
        self.n_random_samples = n_random_samples
        self.random_sample_seed = random_sample_seed
        self.n_jobs = n_jobs
        self.run_vivado_impl = run_vivado_impl

    def package_designs(self, designs, output_dir, dataset_name):
        xilinx_aggregator = DataAggregatorXilinx()

        if self.run_vivado_impl:
            # full flow
            data = xilinx_aggregator.gather_multiple_designs(designs, n_jobs=16)
        else:
            # HLS-only
            data = []
            for design in designs:
                design_meta = xilinx_aggregator.gather_hls_design_data(design)
                synth_meta  = xilinx_aggregator.gather_hls_synthesis_data(design)
                d = CompleteHLSData(
                    design=design_meta,
                    synthesis=synth_meta,
                    implementation=None,
                    execution=None,
                    artifacts=io.BytesIO(b""),
                )
                data.append(d)

        output_archive_fp = output_dir / f"{dataset_name}.zip"
        xilinx_aggregator.aggregated_data_to_archive(data, output_archive_fp)
        print(f"Data saved: {output_archive_fp}")

        with zipfile.ZipFile(output_archive_fp, "r") as z, z.open("data_all.csv") as f:
            df_all = pd.read_csv(f)
        return df_all

    def pareto_frontier(self, df, x_col: str, y_col: str) -> pd.DataFrame:
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

    def delta_gap_based(self, df, x_col: str, y_col: str) -> float:
        pareto_df = self.pareto_frontier(df, x_col, y_col)

        if pareto_df.empty or len(pareto_df) < 3:
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
        data_all = self.package_designs(designs, output_dir, dataset_name)

        if self.run_vivado_impl:  
            resource_map = {
                "LUTs": "implementation__utilization__Total LUTs",
                "FFs": "implementation__utilization__FFs",
            }
        else:
            resource_map = {
                "LUTs": "synthesis__resources_lut_used",
                "FFs": "synthesis__resources_ff_used",
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
        )
        datasets[dataset_name] = designs

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
            )
        )

        TIMEOUT_HLS_SYNTH = 60.0 * 16 
        TIMEOUT_HLS_IMPL = 60.0 * 45

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
        
        if self.run_vivado_impl:
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
                datasets_post_hls_implementation if self.run_vivado_impl else datasets_post_hls_synth,
                False,
                n_jobs=self.n_jobs,
                cpu_affinity=range(self.n_jobs),
            )