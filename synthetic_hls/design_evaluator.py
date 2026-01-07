import os
import shutil
import json
import logging
import threading
import time
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from llm import Response
from typing import List, Any, Optional

from synthetic_hls.vhls_tools import VitisHLSCSimTool, VitisHLSSynthTool
from synthetic_hls.design import Design
from synthetic_hls.llm_models import Model, normalize_model_name
from synthetic_hls.prompting import approx_num_tokens, extract_code_xml_from_llm_output
from synthetic_hls.design_eval_tools import ASTAnalyzer, HLSFactoryFlow

from hlsfactory.utils import remove_and_make_new_dir_if_exists

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

class DesignEvaluator:
    """
    Generate designs using LLM, evaluate them using Vitis HLS toolflow, AST analyzer and HLSFactory flow.
    """
    FULL_FLOW_LOCK = threading.Lock()
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        temperature: float = 0.7,
        clang_path: Optional[Path] = None,
        include_paths: Optional[List[Path]] = None,
        run_vivado_impl: bool = True,
        n_jobs_hlsfactory: int = 24
    ) -> None:
        self.cpp_compiler_tool = vitis_hls_tool_csim
        self.vitis_hls_tool = vitis_hls_tool_synth
        self.template_files_path = Path(__file__).resolve().parent / "tcl_templates"
        self.temperature = temperature
        self.clang_path = clang_path
        self.include_paths = include_paths
        self.logger = logging.getLogger(__name__)
        self.run_vivado_impl = run_vivado_impl
        self.n_jobs_hlsfactory = n_jobs_hlsfactory

    def _serialize_eval_data(self, eval_id: str, eval_output_dir: Path, single_eval_data: dict):
        print(f"[{eval_id}] Saving eval data to json...")
        single_eval_data_json = json.dumps(single_eval_data, indent=4)
        (eval_output_dir / "single_eval_data.json").write_text(str(single_eval_data_json))

    def _generate_error_message(self, prev_design_eval_data: dict) -> str:
        if prev_design_eval_data["c_compile_out"]["data_execution"]["return_code"] != 0:
            c_compile_log = prev_design_eval_data["c_compile_out"]["data_execution"]["stdout"]
            error_lines = [
                line for line in c_compile_log.split("\n") if ("error: ") in line
            ]
            return (
                f"The generated code could not be compiled with a traditional C++ compiler. Please fix the issue and regenerate the corrected code. \nError Messages:\n"
                + "\n".join(error_lines)
            )
        
        if "c_run_out" in prev_design_eval_data and prev_design_eval_data["c_run_out"]["data_execution"]["return_code"] != 0:             
            if prev_design_eval_data["c_run_out"]["data_execution"]["stderr"] == "":
                e = "The generated code could not be executed after compiling with traditional C++ compiler.\n"
                e += f"It seems that stderr is empty and the process returned a non-zero exit code: {prev_design_eval_data['c_run_out']['data_execution']['return_code']}\n"
                e += "This likely means the process segfaulted or had a critical error.\n"
                e += "Likely causes include:\n"
                e += "- Floating point exceptions\n"
                e += "- Segmentation faults\n"
                e += "- Memory corruption\n"
                e += "Please consider fixing the issue and regenerating the corrected code."
                return e
            else:
                error_lines = prev_design_eval_data["c_run_out"]["data_execution"]["stderr"][:1000]
                return (
                    f"The generated code could not be executed after compiling with traditional C++ compiler. Please fix the issue and regenerate the corrected code.\nError Message: {error_lines}\n"
                )
        
        if "vitis_hls_tool_out" in prev_design_eval_data:
            prev_design_syn_data = prev_design_eval_data["vitis_hls_tool_out"]["data_execution"]
            if prev_design_eval_data["vitis_hls_tool_out"]["data_execution"]["return_code"] != 0:
                if (
                    "timeout" in prev_design_syn_data
                    and prev_design_syn_data["timeout"] is True
                ):
                    e = ""
                    e += "The generated code could not be synthesized with Vitis HLS. Please fix the issue and regenerate the corrected code.\n Also regenerate the updated OptDSLv2 optimization template file `opt_template.tcl` file that matches the corrected kernel structure and defines the proper design space.\n"
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
                    + "\n".join(error_lines)
                )
            else:
                if prev_design_eval_data["opt_dsl_out"]["return_code"] == 1:
                    return ("OptDSL Error")
        return None
    
    def _prepare_hlsfactory_inputs(
        self,
        design_generated_dir: Path,
        eval_dir_top: Path,
        eval_design_id: str,
        top_function_name: str,
    ) -> tuple[Path, Path, "HLSFactoryFlow", str]:
        """
        Prepares output_design_dir and work_dir for HLSFactory flow.
        Returns (output_design_dir, work_dir, design_hlsfactory_flow, kernel_name).
        """
        kernel_header = next(design_generated_dir.glob("*.h"), None)
        assert kernel_header is not None, "No kernel header (*.h) found."
        kernel_name = kernel_header.stem

        output_design_dir = eval_dir_top / "output_designs" / eval_design_id / kernel_name
        current_design = Design(design_generated_dir, name=f"{eval_design_id}")
        output_design = current_design.copy_to(output_design_dir)

        # split into src/tb
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

        # Copy template TCLs and patch hls_template
        tcl_files_dir = self.template_files_path
        for tcl in tcl_files_dir.iterdir():
            if tcl.is_file():
                shutil.copy(tcl, output_design_dir)

        hls_template = output_design_dir / "hls_template.tcl"
        content = hls_template.read_text()
        content = content.replace("[top_function_name]", top_function_name)
        content = content.replace("[kernel_name]", kernel_name)
        hls_template.write_text(content)

        work_dir = eval_dir_top / "raw_data" / eval_design_id
        remove_and_make_new_dir_if_exists(work_dir)
        
        # Initialize HLSFactoryFlow
        design_hlsfactory_flow = HLSFactoryFlow(
            design_dir=output_design_dir,
            work_dir=work_dir,
            n_random_samples=64,
            random_sample_seed=64,
            n_jobs=self.n_jobs_hlsfactory,
            run_vivado_impl=self.run_vivado_impl,
        )
        return output_design_dir, work_dir, design_hlsfactory_flow, kernel_name

    def run_hlsfactory_flow(
        self,
        design_generated_dir: Path,
        eval_dir_top: Path,
        eval_id: str,
        eval_design_id: str,
        top_function_name: str,
    ) -> tuple[dict, Path]:
        """
        Runs full HLSFactory flow (sample + analyze) for a prepared design.
        Returns (opt_dsl_out_dict, output_design_dir).
        """
        with DesignEvaluator.FULL_FLOW_LOCK:
            output_design_dir, work_dir, design_hlsfactory_flow, _kernel = self._prepare_hlsfactory_inputs(
                design_generated_dir, eval_dir_top, eval_design_id, top_function_name
            )

            # Validate OptDSL
            opt_dsl_error, opt_dsl_error_message = design_hlsfactory_flow.opt_dsl_check()
            opt_dsl_out = {
                "return_code": 1 if opt_dsl_error else 0,
                "error": opt_dsl_error_message if opt_dsl_error else None,
                "pareto_scores": {}
            }
            if opt_dsl_error:
                shutil.rmtree(output_design_dir.parent, ignore_errors=True)
                shutil.rmtree(work_dir, ignore_errors=True)
                return opt_dsl_out, output_design_dir

            print(f"[{eval_id}] Running full HLSFactory flow...")
            design_hlsfactory_flow.run()
            design_hlsfactory_flow.analyze(
                design_generated_dir=design_generated_dir,
                design_dir=output_design_dir,
                output_dir=eval_dir_top / "zip_data" / eval_design_id,
            )

        try:
            pareto_scores_summary = json.loads((output_design_dir / "pareto_scores_summary.json").read_text())
            opt_dsl_out["pareto_scores"]["LUTs_vs_latency"] = pareto_scores_summary["LUTs_vs_latency"]["pareto_score"]
            opt_dsl_out["pareto_scores"]["FFs_vs_latency"]  = pareto_scores_summary["FFs_vs_latency"]["pareto_score"]
        except Exception as e:
            opt_dsl_out["error"] = f"Missing/invalid pareto_scores_summary.json: {e}"

        shutil.rmtree(work_dir, ignore_errors=True)
        return opt_dsl_out, output_design_dir

    def _opt_dsl_check_only(
        self,
        design_generated_dir: Path,
        eval_dir_top: Path,
        eval_id: str,
        top_function_name: str,
    ) -> dict:
        """
        Only prepares output_design_dir and validates OptDSL; no sampling.
        Useful during non-Pareto iterations to ensure template is consistent.
        """
        output_design_dir, work_dir, design_hlsfactory_flow, _kernel = self._prepare_hlsfactory_inputs(
            design_generated_dir, eval_dir_top, eval_id, top_function_name
        )
        opt_dsl_error, opt_dsl_error_message = design_hlsfactory_flow.opt_dsl_check()
        out = {
            "return_code": 1 if opt_dsl_error else 0,
            "error": opt_dsl_error_message if opt_dsl_error else None,
            "pareto_scores": {}
        }
        # Cleanup staged directories to keep iteration cheap
        shutil.rmtree(output_design_dir.parent, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        return out

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

        eval_id = f"{design_id}__{model_name_normalized}"
        eval_data["eval_type"] = "hls_gen_zero_shot"
        eval_data["eval_id"] = eval_id
        eval_data["status"] = "Fail"
        eval_data["error_message"] = None
        error_message = None
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
            eval_data["seed_design_tags"] = ["llm_gen"]
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
                    and "top.txt" in generated_code.keys()
                )
                assert(
                    len([k for k in generated_code.keys() if k.endswith(".md")]) == 1
                    and "kernel_description.md" in generated_code.keys()
                )
                assert (
                    len([k for k in generated_code.keys() if k.endswith(".tcl")]) == 1
                    and "opt_template.tcl" in generated_code.keys()
                )
                assert(
                    len([k for k in generated_code.keys() if k.endswith(".toml")]) == 1
                    and "hls_eval_config.toml" in generated_code.keys()
                )
            elif output_format == "OPTDSL":
                assert len(generated_code) == 1
                assert( 
                    len([k for k in generated_code.keys() if k.endswith(".tcl")]) == 1
                    and "opt_template.tcl" in generated_code.keys()
                )
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
        eval_data["opt_dsl_out"]["return_code"] = -1        
        eval_data["opt_dsl_out"]["error"] = None
        eval_data["opt_dsl_out"]["pareto_scores"] = {}
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
                    kernel_ast_analyzer.analyze_to_json()
                    print(f"[{eval_id}] Saving AST analysis results to eval data...")
                    eval_data["kernel_ast_out"] = json.loads((design_generated_dir / "call_graph.json").read_text())

            if full_flow:
                opt_out, _out_dir = self.run_hlsfactory_flow(
                    design_generated_dir=design_generated_dir,
                    eval_dir_top=eval_dir_top,
                    eval_id=eval_id,
                    eval_design_id=design_id,
                    top_function_name=top_function_name,
                )
                eval_data["opt_dsl_out"] = opt_out
            else:
                eval_data["opt_dsl_out"] = self._opt_dsl_check_only(
                    design_generated_dir=design_generated_dir,
                    eval_dir_top=eval_dir_top,
                    eval_id=eval_id,
                    top_function_name=top_function_name,
                )
        
        error_message = self._generate_error_message(eval_data)
        if error_message is None:
            eval_data["status"] = "Pass"
        eval_data["error_message"] = error_message
        self._serialize_eval_data(eval_id, eval_dir, eval_data)
        final_output_design = Design(design_generated_dir, name=f"{eval_id}")

        return error_message, final_output_design
