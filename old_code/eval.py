import itertools
import json
import logging
import shutil
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from joblib import Parallel, delayed
from llm import Response

from hls_eval.data import BenchmarkCase, copy_benchmark_list_to
from hls_eval.llms import Model, TAIPromptTooLong, TAITimeout, normalize_model_name
from hls_eval.prompting import approx_num_tokens, extract_code_xml_from_llm_output
from hls_eval.prompts import build_prompt_gen_zero_shot_single_input_with_opt, build_prompt_gen_zero_shot_no_input_with_opt, build_prompt_gen_feed_back, build_prompt_gen_optdsl_zero_shot
from hls_eval.rate_limit import RemoteLLMRateLimit
from hls_eval.tools import VitisHLSCSimTool, VitisHLSSynthTool
from hls_eval.hlsfactory_flow import HLSFactoryFlow


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

        self.llm_sema = BoundedSemaphore(n_jobs_pool_llm)

        self.llm_rate_limiter = RemoteLLMRateLimit(
            tokens_per_minute, requests_per_minute
        )

        self.pool_llm = ThreadPoolExecutor(max_workers=n_jobs_pool_llm)
        self.pool_csim = ThreadPoolExecutor(max_workers=n_jobs_pool_csim)
        self.pool_synth = ThreadPoolExecutor(max_workers=n_jobs_pool_synth)

    def shutdown(self):
        self.pool_llm.shutdown(wait=True)
        self.pool_csim.shutdown(wait=True)
        self.pool_synth.shutdown(wait=True)


class Evaluator(ABC):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        output_data_dir: Path,
    ) -> None:
        self.cpp_compiler_tool = vitis_hls_tool_csim
        self.vitis_hls_tool = vitis_hls_tool_synth
        self.output_data_dir = output_data_dir
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def evaluate_design(
        self,
        model: Model,
        pools: EvalThreadPools,
        benchmark_case: BenchmarkCase | None = None,        
        **kwargs,
    ) -> None:
        raise NotImplementedError

    def build_eval_combos(
        self,
        benchmark_cases: list[BenchmarkCase],
        models: list[Model],
    ) -> list[tuple[BenchmarkCase, Model]]:
        combos = list(itertools.product(benchmark_cases, models))
        combos = sorted(combos, key=lambda x: (x[0].name, x[1].name))
        return combos

    def evaluate_designs(
        self,
        models: list[Model],
        n_jobs: int = 1,
        n_jobs_pool_llm: int = 1,
        n_jobs_pool_csim: int = 1,
        n_jobs_pool_synth: int = 1,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        benchmark_cases: list[BenchmarkCase] | None = None,        
        **kwargs,
    ) -> None:
        pools = EvalThreadPools(
            n_jobs_pool_llm,
            n_jobs_pool_csim,
            n_jobs_pool_synth,
            tokens_per_minute,
            requests_per_minute,
        )
        if benchmark_cases is None:
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(self.evaluate_design)(model, pools, None)
                for model in models
            )
        else:
            combos: list[tuple[BenchmarkCase, Model]] = self.build_eval_combos(
                benchmark_cases, models
            )
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(self.evaluate_design)(model, pools, design)
                for design, model in combos
            )
        pools.shutdown()

    def evaluate_designs_multi_inputs(
        self,
        benchmark_cases: list[BenchmarkCase],
        models: list[Model],
        n_jobs: int = 1,
        n_jobs_pool_llm: int = 1,
        n_jobs_pool_csim: int = 1,
        n_jobs_pool_synth: int = 1,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        **kwargs,
    ) -> None:
        pools = EvalThreadPools(
            n_jobs_pool_llm,
            n_jobs_pool_csim,
            n_jobs_pool_synth,
            tokens_per_minute,
            requests_per_minute,
        )

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.evaluate_design)(benchmark_cases, model, pools)
            for model in models
        )
        pools.shutdown()

    def evaluate_designs_no_input(
        self,
        models: list[Model],
        n_jobs: int = 1,
        n_jobs_pool_llm: int = 1,
        n_jobs_pool_csim: int = 1,
        n_jobs_pool_synth: int = 1,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        **kwargs,
    ) -> None:
        pools = EvalThreadPools(
            n_jobs_pool_llm,
            n_jobs_pool_csim,
            n_jobs_pool_synth,
            tokens_per_minute,
            requests_per_minute,
        )
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.evaluate_design)(model, pools)
            for model in models
        )
        pools.shutdown()

    def evaluate_design_model_pairs(
        self,
        design_model_pairs: list[tuple[BenchmarkCase, Model]],
        n_jobs: int = 1,
        n_jobs_pool_llm: int = 1,
        n_jobs_pool_csim: int = 1,
        n_jobs_pool_synth: int = 1,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        **kwargs,
    ):
        pools = EvalThreadPools(
            n_jobs_pool_llm,
            n_jobs_pool_csim,
            n_jobs_pool_synth,
            tokens_per_minute,
            requests_per_minute,
        )
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.evaluate_design)(design, model, pools)
            for design, model in design_model_pairs
        )
        pools.shutdown()


def serialize_eval_data(eval_id: str, eval_output_dir: Path, single_eval_data: dict):
    print(f"[{eval_id}] Saving eval data to json...")
    single_eval_data_json = json.dumps(single_eval_data, indent=4)
    (eval_output_dir / "single_eval_data.json").write_text(str(single_eval_data_json))

class HLSGenerationZeroShotEvaluator(Evaluator):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        output_data_dir: Path,
        n_samples: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature

        super().__init__(vitis_hls_tool_csim, vitis_hls_tool_synth, output_data_dir)

    def evaluate_design(
        self,
        model: Model,
        pools: EvalThreadPools,
        benchmark_case: BenchmarkCase | None = None,        
        **kwargs,
    ) -> None:
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)
        benchmark_case_name = benchmark_case.name
        eval_id = f"{benchmark_case_name}__{model_name_normalized}"

        eval_dir_top = self.output_data_dir / eval_id
        if eval_dir_top.exists():
            self.logger.info(f"Removing existing top eval dir: {eval_dir_top}")
            shutil.rmtree(eval_dir_top)
        eval_dir_top.mkdir(parents=True)

        for sample_idx in range(self.n_samples):
            eval_data: dict[str, Any] = {}

            eval_data["eval_type"] = "hls_gen_zero_shot"
            eval_data["eval_id"] = eval_id
            eval_data["benchmark_case_name"] = benchmark_case_name
            eval_data["benchmark_case_tags"] = benchmark_case.tags_all
            eval_data["model_name"] = model_name
            eval_data["model_name_normalized"] = model_name_normalized

            eval_data["temperature"] = self.temperature
            eval_data["n_samples"] = self.n_samples

            self.logger.info(f"[{eval_id}] Running eval...")

            eval_dir = eval_dir_top / f"sample__{sample_idx}"
            if eval_dir.exists():
                self.logger.info(f"Removing existing sample eval dir: {eval_dir}")
                shutil.rmtree(eval_dir)
            eval_dir.mkdir(parents=True)

            design_dir = eval_dir / "design"
            benchmark_case = benchmark_case.copy_to(design_dir)

            assert len(benchmark_case.h_files) == 1
            design_header = benchmark_case.h_files[0]
            design_tb = benchmark_case.tb_file
            design_description = benchmark_case.kernel_description_fp
            design_kernel = benchmark_case.kernel_fp
            design_opt = benchmark_case.opt_fp
            design_pareto_score = benchmark_case.pareto_score_fp

            prompt = build_prompt_gen_zero_shot_single_input_with_opt(
                design_description,
                design_header,
                design_kernel,
                design_tb,
                design_opt,
                design_pareto_score,
            )
            eval_data["prompt"] = prompt
            (eval_dir / "raw_llm_prompt.txt").write_text(prompt)

            n_tokens_guess = approx_num_tokens(prompt)

            llm_pool = pools.pool_llm

            llm = model.llm

            t0 = time.monotonic()

            def call_model(
                prompt,
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
                # llm_rate_limiter.wait_for(n_tokens_guess)
                t_0 = time.monotonic()
                r: Response | None = None
                r_text: str | None = None
                r_json: dict | None = None
                model_timeout = False
                prompt_too_long = False
                try:
                    r = llm.prompt(
                        prompt=prompt,
                        stream=False,
                        temperature=self.temperature,
                    )
                    r._force()
                    r_json = r.json()
                    r_text = r.text()
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    model_timeout = False
                    prompt_too_long = False
                # except TAITimeout:
                except (TAITimeout, TAIPromptTooLong) as e:
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    # model_timeout = True
                    if isinstance(e, TAITimeout):
                        model_timeout = True
                        prompt_too_long = False
                    if isinstance(e, TAIPromptTooLong):
                        model_timeout = False
                        prompt_too_long = True

                return r, r_text, r_json, model_timeout, prompt_too_long, t_0, t1, dt

            future_llm = llm_pool.submit(call_model, prompt)
            r, r_text, r_json, model_timeout, prompt_too_long, t0, t1, dt = (
                future_llm.result()
            )

            eval_data["model_timeout"] = model_timeout
            eval_data["prompt_too_long"] = prompt_too_long
            eval_data["llm_execution_time"] = {"t0": t0, "t1": t1, "execution_time": dt}

            if model_timeout or prompt_too_long:
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            assert r is not None
            assert r_text is not None

            if r.response_json is not None:
                eval_data["response_json"] = r.response_json

            eval_data["raw_output"] = str(r_text)
            (eval_dir / "raw_llm_output.txt").write_text(data=r_text)

            print(f"[{eval_id}] Extracting code from output...")
            try:
                generated_code = extract_code_xml_from_llm_output(r_text)
                assert len(generated_code) == 6
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
                eval_data["generated_code"] = generated_code
                eval_data["can_parse_output"] = True
            except Exception:
                print(f"[{eval_id}] Error extracting code from LLM output")
                eval_data["can_parse_output"] = False
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            # make a design_generated dir
            design_generated_dir: Path = eval_dir / "design_generated"
            design_generated_dir.mkdir()

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

            serialize_eval_data(eval_id, eval_dir, eval_data)

        all_eval_data = {}
        for sample_idx in range(self.n_samples):
            sample_eval_data_fp = (
                eval_dir_top / f"sample__{sample_idx}" / "single_eval_data.json"
            )
            sample_eval_data = json.loads(sample_eval_data_fp.read_text())
            all_eval_data[sample_idx] = sample_eval_data
        all_eval_data_fp = eval_dir_top / "all_eval_data.json"
        all_eval_data_fp.write_text(json.dumps(all_eval_data, indent=4))

####################################### FEEDBACK EVALUATOR ###########################################
class HLSGenerationFeedbackEvaluator(Evaluator):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        output_data_dir: Path,
        n_samples: int = 1,
        n_feedback_iters: int = 3,
        n_versions: int = 3,
        temperature: float = 0.7,
        fix: bool = False,
        full_flow: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.n_versions = n_versions
        self.temperature = temperature
        self.n_feedback_iters = n_feedback_iters
        # fix: Whether to try fixing errors using feedback prompts.
        self.fix = fix
        self.full_flow = full_flow
        self.FULL_FLOW_LOCK = threading.Lock()

        super().__init__(vitis_hls_tool_csim, vitis_hls_tool_synth, output_data_dir)

    def _generate_error_message(self, prev_iter_eval_data: dict[str, Any]) -> str:
        prev_iter_syn_data = prev_iter_eval_data["vitis_hls_tool_out"]["data_execution"]
        if (
            "timeout" in prev_iter_syn_data
            and prev_iter_syn_data["timeout"] is True
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

        synth_log = prev_iter_syn_data["stdout"]
        error_lines = [
            line for line in synth_log.split("\n") if line.startswith("ERROR: ")
        ]
        return (
            "The generated code could not be synthesized with Vitis HLS. Please fix the issue and regenerate the corrected code. \nError Messages:\n"
            + "\n".join(error_lines)
        )

    def evaluate_design(
        self,
        model: Model,
        pools: EvalThreadPools,
        benchmark_case: BenchmarkCase | None = None,        
        **kwargs,
    ) -> None:
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)
        error_message = None
        if benchmark_case is None:
            eval_id = f"no_input__{model_name_normalized}"
        else:
            benchmark_case_name = benchmark_case.name
            eval_id = f"{benchmark_case_name}__{model_name_normalized}"

        eval_dir_top = self.output_data_dir / eval_id
        if eval_dir_top.exists():
            self.logger.info(f"Removing existing top eval dir: {eval_dir_top}")
            shutil.rmtree(eval_dir_top)
        eval_dir_top.mkdir(parents=True)

        for sample_idx in range(self.n_samples):
            eval_dir = eval_dir_top / f"sample__{sample_idx}"
            if eval_dir.exists():
                self.logger.info(f"Removing existing sample eval dir: {eval_dir}")
                shutil.rmtree(eval_dir)
            eval_dir.mkdir(parents=True)
            previous_case: BenchmarkCase = None
            initial_error = False

            for iter in range(0, self.n_feedback_iters + 1):
                version = 0
                # Try several versions till pass
                while version < self.n_versions:
                    iter_eval_dir = eval_dir / f"iter__{iter}_{version}"
                    if iter_eval_dir.exists():
                        self.logger.info(f"Removing existing iter eval dir: {iter_eval_dir}")
                        shutil.rmtree(iter_eval_dir)
                    iter_eval_dir.mkdir(parents=True)

                    iter_eval_data: dict[str, Any] = {}

                    iter_eval_data["eval_type"] = "hls_gen_zero_shot"
                    iter_eval_data["eval_id"] = eval_id
                    iter_eval_data["model_name"] = model_name
                    iter_eval_data["model_name_normalized"] = model_name_normalized

                    iter_eval_data["temperature"] = self.temperature
                    iter_eval_data["n_samples"] = self.n_samples
                    iter_eval_data["n_feedback_iters"] = self.n_feedback_iters

                    self.logger.info(f"[{eval_id}] Running eval...")

                    iter_design_dir = iter_eval_dir / "design"

                    if iter == 0:
                        if benchmark_case is None:
                            prompt = build_prompt_gen_zero_shot_no_input_with_opt()
                        else:
                            input_benchmark_case = benchmark_case.copy_to(iter_design_dir)

                            assert len(input_benchmark_case.h_files) == 1
                            design_header = input_benchmark_case.h_files[0]
                            design_tb = input_benchmark_case.tb_file
                            design_description = input_benchmark_case.kernel_description_fp
                            design_kernel = input_benchmark_case.kernel_fp
                            design_opt = input_benchmark_case.opt_fp
                            design_pareto_score = input_benchmark_case.pareto_score_fp

                            prompt = build_prompt_gen_zero_shot_single_input_with_opt(
                                design_description,
                                design_header,
                                design_kernel,
                                design_tb,
                                design_opt,
                                design_pareto_score,
                                )                   
                    else:
                        previous_case = previous_case.copy_to(iter_design_dir)
                        assert len(previous_case.h_files) == 1
                        design_header = previous_case.h_files[0]
                        design_tb = previous_case.tb_file
                        design_description = previous_case.kernel_description_fp
                        design_kernel = previous_case.kernel_fp
                        design_opt = previous_case.opt_fp
                        design_pareto_score = previous_case.pareto_score_fp

                        prompt = build_prompt_gen_feed_back(
                            design_description,
                            design_header,
                            design_kernel,
                            design_tb,
                            design_opt,
                            design_pareto_score,
                            error_message,
                            self.fix
                        )

                    iter_eval_data["prompt"] = prompt
                    (iter_eval_dir / "raw_llm_prompt.txt").write_text(prompt)

                    n_tokens_guess = approx_num_tokens(prompt)

                    llm_pool = pools.pool_llm

                    llm = model.llm

                    t0 = time.monotonic()

                    def call_model(
                        prompt,
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
                        # llm_rate_limiter.wait_for(n_tokens_guess)
                        t_0 = time.monotonic()
                        r: Response | None = None
                        r_text: str | None = None
                        r_json: dict | None = None
                        model_timeout = False
                        prompt_too_long = False
                        try:
                            r = llm.prompt(
                                prompt=prompt,
                                stream=False,
                                temperature=self.temperature,
                            )
                            r._force()
                            r_json = r.json()
                            r_text = r.text()
                            t1 = time.monotonic()
                            dt = t1 - t_0
                            model_timeout = False
                            prompt_too_long = False
                        # except TAITimeout:
                        except (TAITimeout, TAIPromptTooLong) as e:
                            t1 = time.monotonic()
                            dt = t1 - t_0
                            # model_timeout = True
                            if isinstance(e, TAITimeout):
                                model_timeout = True
                                prompt_too_long = False
                            if isinstance(e, TAIPromptTooLong):
                                model_timeout = False
                                prompt_too_long = True

                        return r, r_text, r_json, model_timeout, prompt_too_long, t_0, t1, dt

                    future_llm = llm_pool.submit(call_model, prompt)
                    r, r_text, r_json, model_timeout, prompt_too_long, t0, t1, dt = (
                        future_llm.result()
                    )

                    iter_eval_data["model_timeout"] = model_timeout
                    iter_eval_data["prompt_too_long"] = prompt_too_long
                    iter_eval_data["llm_execution_time"] = {"t0": t0, "t1": t1, "execution_time": dt}

                    if model_timeout or prompt_too_long:
                        serialize_eval_data(eval_id, iter_eval_dir, iter_eval_data)
                        continue

                    assert r is not None
                    assert r_text is not None

                    if r.response_json is not None:
                        iter_eval_data["response_json"] = r.response_json

                    iter_eval_data["raw_output"] = str(r_text)
                    (iter_eval_dir / "raw_llm_output.txt").write_text(data=r_text)

                    print(f"[{eval_id}] Extracting code from output...")
                    try:
                        generated_code = extract_code_xml_from_llm_output(r_text)
                        assert len(generated_code) == 6
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
                        iter_eval_data["generated_code"] = generated_code
                        iter_eval_data["can_parse_output"] = True
                    except Exception:
                        print(f"[{eval_id}] Error extracting code from LLM output")
                        iter_eval_data["can_parse_output"] = False
                        serialize_eval_data(eval_id, iter_eval_dir, iter_eval_data)
                        version += 1
                        continue

                    # make a design_generated dir
                    design_generated_dir: Path = iter_eval_dir / "design_generated"
                    design_generated_dir.mkdir()

                    # write the generated code to a file
                    for file_name, code in generated_code.items():
                        (design_generated_dir / f"{file_name}").write_text(code)

                    build_dir = iter_eval_dir / "build"
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

                    iter_eval_data["c_compile_out"] = {}
                    iter_eval_data["c_compile_out"]["data_execution"] = {
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
                        iter_eval_data["c_run_out"] = {}
                        iter_eval_data["c_run_out"]["data_execution"] = {
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

                    iter_eval_data["vitis_hls_tool_out"] = {}
                    iter_eval_data["vitis_hls_tool_out"]["data_execution"] = {
                        "return_code": vitis_hls_tool_output.data_execution.return_code,
                        "stdout": vitis_hls_tool_output.data_execution.stdout,
                        "stderr": vitis_hls_tool_output.data_execution.stderr,
                        "t0": vitis_hls_tool_output.data_execution.t0,
                        "t1": vitis_hls_tool_output.data_execution.t1,
                        "execution_time": vitis_hls_tool_output.data_execution.execution_time,
                        "timeout": vitis_hls_tool_output.data_execution.timeout,
                    }
                    iter_eval_data["vitis_hls_tool_out"]["data_tool"] = {}
                    if vitis_hls_tool_output.data_tool:
                        for k, v in vitis_hls_tool_output.data_tool.items():
                            iter_eval_data["vitis_hls_tool_out"]["data_tool"][k] = v
                    print(
                        f"[{eval_id}] Vitis HLS return code: {vitis_hls_tool_output.data_execution.return_code}"
                    )

                    serialize_eval_data(eval_id, iter_eval_dir, iter_eval_data)

                    prev_iter_eval_data = iter_eval_data.copy()

                    if vitis_hls_tool_output.data_execution.return_code != 0:
                        if iter == 0:
                            initial_error = True
                            break
                        else:
                            error_message = self._generate_error_message(prev_iter_eval_data)
                            if self.fix:
                                previous_case = BenchmarkCase(iter_generated_design_dir, name=f"{eval_id}__{iter}_{version}")
                            version += 1
                    else:
                        error_message = None
                        iter_generated_design_dir = iter_eval_dir / "design_generated"

                        kernel_name = (next(design_generated_dir.glob("*.h"), None)).stem
                        output_design_dir = eval_dir_top / "output_designs"/ f"sample__{sample_idx}"  / f"iter__{iter}_{version}" / kernel_name
                        src_dir = output_design_dir / "src"
                        tb_dir = output_design_dir / "tb"
                        output_design_dir.mkdir(parents=True, exist_ok=True)
                        src_dir.mkdir(parents=True, exist_ok=True)
                        tb_dir.mkdir(parents=True, exist_ok=True)
                        tcl_files_dir = self.output_data_dir.parent / "tcl_files"

                        for item in design_generated_dir.iterdir():
                            if item.is_file() and not item.name.endswith("pareto_score.txt"):
                                shutil.copy(item, output_design_dir)

                        for f in output_design_dir.glob("*.h"):
                            shutil.move(str(f), src_dir)
                        for f in output_design_dir.glob("*.cpp"):
                            if not f.name.endswith("_tb.cpp"):
                                shutil.move(str(f), src_dir)
                            else:
                                shutil.move(str(f), tb_dir)

                        for tcl in tcl_files_dir.iterdir():
                            if tcl.is_file():
                                shutil.copy(tcl, output_design_dir)

                        hls_template = output_design_dir / "hls_template.tcl"
                        content = hls_template.read_text()
                        content = content.replace("[top_function_name]", top_function_name)    
                        content = content.replace("[kernel_name]", kernel_name)
                        hls_template.write_text(content)

                        if self.full_flow:
                            work_dir = eval_dir_top / "raw_data" / f"sample__{sample_idx}"
                            design_hlsfactory_flow = HLSFactoryFlow(
                                design_dir = output_design_dir,
                                work_dir = work_dir,
                                n_random_samples = 64,
                                random_sample_seed = 64,
                                n_jobs = 8,
                            )
                            # Check if there are any inconsistent factor lists in the optdsl file before running the full flow.
                            if design_hlsfactory_flow.opt_dsl_check():
                                print("Inconsistent factor lists detected!")
                                version += 1
                            else:
                                with self.FULL_FLOW_LOCK:
                                    design_hlsfactory_flow.run()
                                    design_hlsfactory_flow.analyze(
                                        design_generated_dir = design_generated_dir, 
                                        design_dir = output_design_dir,
                                        output_dir = eval_dir_top / "zip_data" / f"sample__{sample_idx}",
                                    )
                                    previous_case = BenchmarkCase(iter_generated_design_dir, name=f"{eval_id}__{iter}_{version}")
                                    break
                        else:
                            previous_case = BenchmarkCase(iter_generated_design_dir, name=f"{eval_id}__{iter}_{version}")
                            break
                if initial_error or version >= self.n_versions:
                    break
                      

        all_eval_data = {}
        for sample_idx in range(self.n_samples):
            sample_all_eval_data = {}
            for iter_idx in range(0, self.n_feedback_iters + 1):
                for version in range(0, self.n_versions):
                    iter_eval_dir = eval_dir_top / f"sample__{sample_idx}" / f"iter__{iter_idx}_{version}"
                    sample_iter_eval_data_fp = iter_eval_dir / "single_eval_data.json"
                    if not sample_iter_eval_data_fp.exists():
                        break
                    else:
                        sample_iter_eval_data = json.loads(sample_iter_eval_data_fp.read_text())
                        all_eval_data[f"{sample_idx}__{iter_idx}_{version}"] = sample_iter_eval_data
                        if iter_idx not in sample_all_eval_data:
                            sample_all_eval_data[iter_idx] = {}
                        sample_all_eval_data[iter_idx][version] = sample_iter_eval_data
            sample_all_eval_data_fp = eval_dir_top / f"sample__{sample_idx}" / "single_eval_data.json"
            sample_all_eval_data_fp.write_text(json.dumps(sample_all_eval_data, indent=4))
        all_eval_data_fp = eval_dir_top / "all_eval_data.json"
        all_eval_data_fp.write_text(json.dumps(all_eval_data, indent=4))




class HLSGenerationZeroShotEvaluator_no_input(Evaluator):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        output_data_dir: Path,
        n_samples: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature

        super().__init__(vitis_hls_tool_csim, vitis_hls_tool_synth, output_data_dir)

    def evaluate_design(
        self,
        # benchmark_case: BenchmarkCase,
        model: Model,
        pools: EvalThreadPools,
        **kwargs,
    ) -> None:
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)
        # benchmark_case_name = benchmark_case.name
        # eval_id = f"{benchmark_case_name}__{model_name_normalized}"
        eval_id = f"{model_name_normalized}"

        eval_dir_top = self.output_data_dir / eval_id
        if eval_dir_top.exists():
            self.logger.info(f"Removing existing top eval dir: {eval_dir_top}")
            shutil.rmtree(eval_dir_top)
        eval_dir_top.mkdir(parents=True)

        for sample_idx in range(self.n_samples):
            eval_data: dict[str, Any] = {}

            eval_data["eval_type"] = "hls_gen_zero_shot"
            eval_data["eval_id"] = eval_id
            # eval_data["benchmark_case_name"] = benchmark_case_name
            # eval_data["benchmark_case_tags"] = benchmark_case.tags_all
            eval_data["model_name"] = model_name
            eval_data["model_name_normalized"] = model_name_normalized

            eval_data["temperature"] = self.temperature
            eval_data["n_samples"] = self.n_samples

            self.logger.info(f"[{eval_id}] Running eval...")

            eval_dir = eval_dir_top / f"sample__{sample_idx}"
            if eval_dir.exists():
                self.logger.info(f"Removing existing sample eval dir: {eval_dir}")
                shutil.rmtree(eval_dir)
            eval_dir.mkdir(parents=True)

            design_dir = eval_dir / "design"
            # benchmark_case = benchmark_case.copy_to(design_dir)

            # assert len(benchmark_case.h_files) == 1
            # design_header = benchmark_case.h_files[0]
            # design_tb = benchmark_case.tb_file
            # design_description = benchmark_case.kernel_description_fp
            # design_kernel = benchmark_case.kernel_fp

            # prompt = build_prompt_gen_2_zero_shot_no_input()
            prompt = build_prompt_gen_zero_shot_no_input_with_opt()
            eval_data["prompt"] = prompt
            (eval_dir / "raw_llm_prompt.txt").write_text(prompt)

            n_tokens_guess = approx_num_tokens(prompt)

            llm_pool = pools.pool_llm

            llm = model.llm

            t0 = time.monotonic()

            def call_model(
                prompt,
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
                # llm_rate_limiter.wait_for(n_tokens_guess)
                t_0 = time.monotonic()
                r: Response | None = None
                r_text: str | None = None
                r_json: dict | None = None
                model_timeout = False
                prompt_too_long = False
                try:
                    r = llm.prompt(
                        prompt=prompt,
                        stream=False,
                        temperature=self.temperature,
                    )
                    r._force()
                    r_json = r.json()
                    r_text = r.text()
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    model_timeout = False
                    prompt_too_long = False
                # except TAITimeout:
                except (TAITimeout, TAIPromptTooLong) as e:
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    # model_timeout = True
                    if isinstance(e, TAITimeout):
                        model_timeout = True
                        prompt_too_long = False
                    if isinstance(e, TAIPromptTooLong):
                        model_timeout = False
                        prompt_too_long = True

                return r, r_text, r_json, model_timeout, prompt_too_long, t_0, t1, dt

            future_llm = llm_pool.submit(call_model, prompt)
            r, r_text, r_json, model_timeout, prompt_too_long, t0, t1, dt = (
                future_llm.result()
            )

            eval_data["model_timeout"] = model_timeout
            eval_data["prompt_too_long"] = prompt_too_long
            eval_data["llm_execution_time"] = {"t0": t0, "t1": t1, "execution_time": dt}

            if model_timeout or prompt_too_long:
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            assert r is not None
            assert r_text is not None

            if r.response_json is not None:
                eval_data["response_json"] = r.response_json

            eval_data["raw_output"] = str(r_text)
            (eval_dir / "raw_llm_output.txt").write_text(data=r_text)

            print(f"[{eval_id}] Extracting code from output...")
            try:
                generated_code = extract_code_xml_from_llm_output(r_text)
                assert len(generated_code) == 6
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
                eval_data["generated_code"] = generated_code
                eval_data["can_parse_output"] = True
            except Exception:
                print(f"[{eval_id}] Error extracting code from LLM output")
                eval_data["can_parse_output"] = False
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            # make a design_generated dir
            design_generated_dir: Path = eval_dir / "design_generated"
            design_generated_dir.mkdir()

            # copy everything in design to design_generated recsursively
            # shutil.copytree(design_dir, design_generated_dir, dirs_exist_ok=True)
            # for f in design_generated_dir.glob("*"):
            #     if f.name == design_kernel.name:
            #         f.unlink()
            #     if f.name == design_tb.name:
            #         f.unlink()
            #     if f.name == design_header.name:
            #         f.unlink()

            # write the generated code to a file
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

            serialize_eval_data(eval_id, eval_dir, eval_data)

        all_eval_data = {}
        for sample_idx in range(self.n_samples):
            sample_eval_data_fp = (
                eval_dir_top / f"sample__{sample_idx}" / "single_eval_data.json"
            )
            sample_eval_data = json.loads(sample_eval_data_fp.read_text())
            all_eval_data[sample_idx] = sample_eval_data
        all_eval_data_fp = eval_dir_top / "all_eval_data.json"
        all_eval_data_fp.write_text(json.dumps(all_eval_data, indent=4))


class HLSGenerationZeroShotEvaluator_optdsl(Evaluator):
    def __init__(
        self,
        vitis_hls_tool_csim: VitisHLSCSimTool,
        vitis_hls_tool_synth: VitisHLSSynthTool,
        output_data_dir: Path,
        n_samples: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.n_samples = n_samples
        self.temperature = temperature

        super().__init__(vitis_hls_tool_csim, vitis_hls_tool_synth, output_data_dir)

    def evaluate_design(
        self,
        model: Model,
        pools: EvalThreadPools,
        benchmark_case: BenchmarkCase | None = None,        
        **kwargs,
    ) -> None:
        model_name: str = model.name
        model_name_normalized = normalize_model_name(model_name)
        benchmark_case_name = benchmark_case.name
        eval_id = f"{benchmark_case_name}__{model_name_normalized}"

        eval_dir_top = self.output_data_dir / eval_id
        if eval_dir_top.exists():
            self.logger.info(f"Removing existing top eval dir: {eval_dir_top}")
            shutil.rmtree(eval_dir_top)
        eval_dir_top.mkdir(parents=True)

        for sample_idx in range(self.n_samples):
            eval_data: dict[str, Any] = {}

            eval_data["eval_type"] = "hls_gen_zero_shot"
            eval_data["eval_id"] = eval_id
            eval_data["benchmark_case_name"] = benchmark_case_name
            eval_data["benchmark_case_tags"] = benchmark_case.tags_all
            eval_data["model_name"] = model_name
            eval_data["model_name_normalized"] = model_name_normalized

            eval_data["temperature"] = self.temperature
            eval_data["n_samples"] = self.n_samples

            self.logger.info(f"[{eval_id}] Running eval...")

            eval_dir = eval_dir_top / f"sample__{sample_idx}"
            if eval_dir.exists():
                self.logger.info(f"Removing existing sample eval dir: {eval_dir}")
                shutil.rmtree(eval_dir)
            eval_dir.mkdir(parents=True)

            design_dir = eval_dir / "design"
            benchmark_case = benchmark_case.copy_to(design_dir)

            assert len(benchmark_case.h_files) == 1
            design_header = benchmark_case.h_files[0]
            design_tb = benchmark_case.tb_file
            design_description = benchmark_case.kernel_description_fp
            design_kernel = benchmark_case.kernel_fp

            prompt = build_prompt_gen_optdsl_zero_shot(
                design_description,
                design_header,
                design_kernel,
                design_tb,
            )
            eval_data["prompt"] = prompt
            (eval_dir / "raw_llm_prompt.txt").write_text(prompt)

            n_tokens_guess = approx_num_tokens(prompt)

            llm_pool = pools.pool_llm

            llm = model.llm

            t0 = time.monotonic()

            def call_model(
                prompt,
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
                # llm_rate_limiter.wait_for(n_tokens_guess)
                t_0 = time.monotonic()
                r: Response | None = None
                r_text: str | None = None
                r_json: dict | None = None
                model_timeout = False
                prompt_too_long = False
                try:
                    r = llm.prompt(
                        prompt=prompt,
                        stream=False,
                        temperature=self.temperature,
                    )
                    r._force()
                    r_json = r.json()
                    r_text = r.text()
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    model_timeout = False
                    prompt_too_long = False
                # except TAITimeout:
                except (TAITimeout, TAIPromptTooLong) as e:
                    t1 = time.monotonic()
                    dt = t1 - t_0
                    # model_timeout = True
                    if isinstance(e, TAITimeout):
                        model_timeout = True
                        prompt_too_long = False
                    if isinstance(e, TAIPromptTooLong):
                        model_timeout = False
                        prompt_too_long = True

                return r, r_text, r_json, model_timeout, prompt_too_long, t_0, t1, dt

            future_llm = llm_pool.submit(call_model, prompt)
            r, r_text, r_json, model_timeout, prompt_too_long, t0, t1, dt = (
                future_llm.result()
            )

            eval_data["model_timeout"] = model_timeout
            eval_data["prompt_too_long"] = prompt_too_long
            eval_data["llm_execution_time"] = {"t0": t0, "t1": t1, "execution_time": dt}

            if model_timeout or prompt_too_long:
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            assert r is not None
            assert r_text is not None

            if r.response_json is not None:
                eval_data["response_json"] = r.response_json

            eval_data["raw_output"] = str(r_text)
            (eval_dir / "raw_llm_output.txt").write_text(data=r_text)

            print(f"[{eval_id}] Extracting code from output...")
            try:
                generated_code = extract_code_xml_from_llm_output(r_text)
                assert len(generated_code) == 1
                assert len([k for k in generated_code.keys() if k.endswith(".tcl")]) == 1
                eval_data["generated_code"] = generated_code
                eval_data["can_parse_output"] = True
            except Exception:
                print(f"[{eval_id}] Error extracting code from LLM output")
                eval_data["can_parse_output"] = False
                serialize_eval_data(eval_id, eval_dir, eval_data)
                continue

            # make a design_generated dir
            design_generated_dir: Path = eval_dir / "design_generated"
            design_generated_dir.mkdir()

            # copy everything in design to design_generated recsursively
            shutil.copytree(design_dir, design_generated_dir, dirs_exist_ok=True)
            # for f in design_generated_dir.glob("*"):
            #     if f.name == design_kernel.name:
            #         f.unlink()
            #     if f.name == design_tb.name:
            #         f.unlink()
            #     if f.name == design_header.name:
            #         f.unlink()

            # write the generated code to a file
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

            serialize_eval_data(eval_id, eval_dir, eval_data)

        all_eval_data = {}
        for sample_idx in range(self.n_samples):
            sample_eval_data_fp = (
                eval_dir_top / f"sample__{sample_idx}" / "single_eval_data.json"
            )
            sample_eval_data = json.loads(sample_eval_data_fp.read_text())
            all_eval_data[sample_idx] = sample_eval_data
        all_eval_data_fp = eval_dir_top / "all_eval_data.json"
        all_eval_data_fp.write_text(json.dumps(all_eval_data, indent=4))