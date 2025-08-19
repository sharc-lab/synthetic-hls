import re
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Any, Dict

from synthetic_hls.design import Design


def approx_num_tokens(text: str) -> int:
    """Approximate number of tokens in text using simple heuristic."""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def extract_code_xml_from_llm_output(output: str) -> Dict[str, str]:
    """Extract code blocks from LLM output in XML format."""
    pattern = r'<OUTPUT_CODE name="([^"]+)">\s*(.*?)\s*</OUTPUT_CODE>'
    matches = re.findall(pattern, output, re.DOTALL)

    if not matches:
        raise ValueError("No code blocks found in LLM output")

    return {filename: code.strip() for filename, code in matches}


def build_input_code_prompt_xml(code_inputs: Dict[str, str]) -> str:
    """Build XML formatted input code prompt."""
    xml_blocks = []
    for filename, content in code_inputs.items():
        xml_blocks.append(f'<INPUT_CODE name="{filename}">\n{content}\n</INPUT_CODE>')
    return "\n".join(xml_blocks)


# Core prompt templates
PROMPT_PRE = dedent("""
## Overview
You are a helpful expert hardware engineer and software developer who will assist the user with hardware design tasks for high-level synthesis.
The task will center around high-level synthesis (HLS) code written in C++ for a hardware design. The HLS design is written to target the latest Vitis HLS tool from Xilinx, which maps C++ code to a Verilog implementation for FPGAs.
""").strip()

PROMPT_GEN_SINGLE_INPUT_WITH_OPT = dedent("""
You are provided with a reference HLS benchmark containing:
- A C++ kernel implementation, header file, testbench, kernel description, and an OptDSLv2 optimization template file named `opt_template.tcl` defining its design space for optimization.
- A `pareto_score.txt` file containing two scalar metrics that quantify the scalability of the benchmark. Each metric is of the form:
    - `pareto_score_LUTs_vs_latency = <float>`
    - `pareto_score_FFs_vs_latency = <float>`
    These reflect how effectively the benchmark spans a tradeoff frontier between resource usage (LUTs, FFs) and performance (latency). 
    Smaller scores indicate better scalability.

### Task Description
Your task is to:
1. Design and implement a new benchmark as a complete high-complexity, multi-stage, application-level HLS-compatible C++ kernel with equal or better scalability (equal or smaller Pareto scores) than the reference benchmark.
2. Write a matching C++ header file for the design.
3. Create a testbench that can validate the functionality of the design.
4. Output the name of the top-level function in a file named `top.txt`.
5. Write a markdown file `kernel_description.md` that provides a concise, human-readable natural language description of the generated benchmark, explaining its functionality, purpose, inputs, and outputs.
6. Generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only.

### Important Constraints for the Benchmark Design:
- You must create an entirely new and original benchmark. DO NOT copy, approximate, re-implement, or repackage any given or pre-existing benchmarks.
- The functionality and algorithm of your benchmark must be unique and created from scratch.
- The total size of the design must be moderate and practical for synthesis. Typically, loop bounds should be smaller than 128.

### Structural Requirements:
- You must generate an entirely new and original benchmark that reflects a complete, multi-stage application accelerator.
- The benchmark should represent a complete real-world application case with at least 6 interconnected sub-functions.
- All for loops must be explicitly labeled using the syntax loop_label: for (...).
- DO NOT add any performance optimization pragmas in the kernel implementation file except `#pragma HLS top name=...`.

### Code Clarity Requirement
- DO NOT add any comments to the generated files.
- All generated code must be clean, syntactically correct, and self-contained.
""").strip()

PROMPT_GEN_NO_INPUT_WITH_OPT = dedent("""
## Task Description
Your task is to:
1. Design and implement a new benchmark as a well-scalable, high-complexity, multi-stage, application-level HLS-compatible C++ kernel.
2. Write a matching C++ header file for the design.
3. Create a testbench that can validate the functionality of the design.
4. Output the name of the top-level function in a file named `top.txt`.
5. Write a markdown file `kernel_description.md` that provides a concise description of the generated benchmark.
6. Generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration.

### Important Constraints:
- You must create an entirely new and original benchmark.
- The functionality and algorithm must be unique and created from scratch.
- The design must be moderate and practical for synthesis.
- All for loops must be labeled using the syntax `<label>: for (...)`.
- DO NOT add performance optimization pragmas except `#pragma HLS top name=...`.

### Code Clarity Requirement
- DO NOT add any comments to the generated files.
- All generated code must be clean and self-contained.
""").strip()

PROMPT_FEEDBACK_ITER_WITH_OPT = dedent("""
You are provided with a previously generated HLS benchmark that encountered synthesis errors.

Your task is to:
1. Fix the synthesis errors in the provided benchmark.
2. Regenerate all files with the corrections applied.
3. Ensure the design remains fully synthesizable by Vitis HLS.

### Error Information:
{error_message}

### Critical Constraints:
- You must ensure the redesigned benchmark remains fully synthesizable.
- All syntax, memory usage, control structures must comply with Vitis HLS compatibility.
- DO NOT add comments to the generated files.
- All generated code must be clean and syntactically correct.
""").strip()

# Output format templates
OUTPUT_FORMAT_GEN_WITH_OPT_XML = dedent("""
## Output Format
The generated HLS output code should be provided in the following format:
```
<OUTPUT_CODE name="kernel_name.h">
    ...
</OUTPUT_CODE>
<OUTPUT_CODE name="kernel_name.cpp">
    ...
</OUTPUT_CODE>
<OUTPUT_CODE name="kernel_name_tb.cpp">
    ...
</OUTPUT_CODE>
<OUTPUT_CODE name="top.txt">
    top_function_name
</OUTPUT_CODE>
<OUTPUT_CODE name="kernel_description.md">
    ...
</OUTPUT_CODE>
<OUTPUT_CODE name="opt_template.tcl">
    ...
</OUTPUT_CODE>
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.

You must output all six code blocks: the generated kernel code, header file, testbench code, top-level function name, kernel description, and OptDSLv2 template.
Make sure the testbench filename ends with `_tb.cpp`.

Only output the generated HLS code in the XML format and nothing else.
""").strip()


def build_prompt_gen_zero_shot_single_input_with_opt(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_opt: Path | None = None,
    design_pareto_score: Path | None = None,
) -> str:
    """Build prompt for generating new design based on reference design."""
    p = PROMPT_PRE
    p += "\n\n"
    p += PROMPT_GEN_SINGLE_INPUT_WITH_OPT
    p += "\n\n"
    p += OUTPUT_FORMAT_GEN_WITH_OPT_XML
    p += "\n\n"

    p += "## Task Inputs\n"
    p += "\n"

    code_inputs = {
        design_description_fp.name: design_description_fp.read_text(),
        design_h.name: design_h.read_text(),
        design_kernel.name: design_kernel.read_text(),
        design_tb.name: design_tb.read_text(),
    }

    if design_opt:
        code_inputs[design_opt.name] = design_opt.read_text()

    if design_pareto_score:
        code_inputs[design_pareto_score.name] = design_pareto_score.read_text()

    code = build_input_code_prompt_xml(code_inputs)
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p


def build_prompt_gen_zero_shot_no_input_with_opt() -> str:
    """Build prompt for generating design from scratch."""
    p = PROMPT_PRE
    p += "\n\n"
    p += PROMPT_GEN_NO_INPUT_WITH_OPT
    p += "\n\n"
    p += OUTPUT_FORMAT_GEN_WITH_OPT_XML
    p += "\n\n"
    p += "## Task Output\n"
    p += "\n"

    return p


def build_prompt_gen_feedback(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_opt: Path | None = None,
    design_pareto_score: Path | None = None,
    error_message: str | None = None,
) -> str:
    """Build prompt for fixing design based on feedback."""
    p = PROMPT_PRE
    p += "\n\n"

    if error_message:
        prompt_template = Template(PROMPT_FEEDBACK_ITER_WITH_OPT)
        p += prompt_template.substitute(error_message=error_message)
    else:
        p += PROMPT_FEEDBACK_ITER_WITH_OPT.replace("{error_message}", "")

    p += "\n\n"
    p += OUTPUT_FORMAT_GEN_WITH_OPT_XML
    p += "\n\n"

    p += "## Task Inputs\n"
    p += "\n"

    code_inputs = {
        design_description_fp.name: design_description_fp.read_text(),
        design_h.name: design_h.read_text(),
        design_kernel.name: design_kernel.read_text(),
        design_tb.name: design_tb.read_text(),
    }

    if design_opt:
        code_inputs[design_opt.name] = design_opt.read_text()

    if design_pareto_score:
        code_inputs[design_pareto_score.name] = design_pareto_score.read_text()

    code = build_input_code_prompt_xml(code_inputs)
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p


def build_prompt_fix(current_design: Design, design_error: str) -> str:
    """Build prompt for fixing synthesis errors."""
    return build_prompt_gen_feedback(
        current_design.kernel_description_fp,
        current_design.h_files[0],
        current_design.kernel_fp,
        current_design.tb_file,
        current_design.design_dir / "opt_template.tcl"
        if (current_design.design_dir / "opt_template.tcl").exists()
        else None,
        current_design.design_dir / "pareto_score.txt"
        if (current_design.design_dir / "pareto_score.txt").exists()
        else None,
        design_error,
    )


def build_prompt_mutate_target(current_design: Design, target: str) -> str:
    """Build prompt for targeted mutation/improvement."""
    # For now, use feedback prompt with target as error message
    # Can be enhanced later for specific mutation types
    return build_prompt_fix(current_design, f"Improve design to achieve: {target}")
