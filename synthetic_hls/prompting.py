import re
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Any, Dict

from synthetic_hls.design import Design
from synthetic_hls.prompts import (
    PROMPT_PRE,
    COMPLEXITY_TARGETS,
    PROMPT_GEN_OPTDSL_V2,
    PROMPT_GEN_NO_INPUT_WITH_OPT,
    PROMPT_GEN_SINGLE_INPUT_WITH_OPT,
    PROMPT_GEN_FIX_WITH_OPT,
    PROMPT_GEN_AST_FEEDBACK_WITH_OPT,
    PROMPT_GEN_SCORE_FEEDBACK_WITH_OPT
)


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
    

# Output format templates
OUTPUT_FORMAT_OPTDSL = dedent(
    text="""
## Output Format
The generated OptDSL `opt_template.tcl` file should be provided in the following XML format:
```
<OUTPUT_CODE name="opt_template.tcl">
    ...
</OUTPUT_CODE>
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.
Only output the generated TCL code in the XML format and nothing else.
"""
).strip()

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
<OUTPUT_CODE name="hls_eval_config.toml">
    tags = ["llm_gen"]
</OUTPUT_CODE>
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.

You must output all seven code blocks: the generated kernel code, header file, testbench code, top-level function name, kernel description, OptDSLv2 template, and HLS evaluation config.
Make sure the testbench filename ends with `_tb.cpp`.

Only output the generated HLS code in the XML format and nothing else.
""").strip()


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


def build_prompt_fix(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_opt: Path | None = None,
    error_message: str | None = None,
) -> str:
    """Build prompt for fixing design based on feedback."""
    p = PROMPT_PRE
    p += "\n\n"
    if error_message == "OptDSL error":
        p += PROMPT_GEN_OPTDSL_V2
        p += "\n\n"
        p += OUTPUT_FORMAT_OPTDSL
        p += "\n\n"
    else:
        prompt_template = Template(PROMPT_GEN_FIX_WITH_OPT)
        p += prompt_template.substitute(error_message=error_message)
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

    code = build_input_code_prompt_xml(code_inputs)
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p


def build_prompt_gen_feedback(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_call_graph: Path | None = None,
    design_opt: Path | None = None,
    design_pareto_score: Path | None = None,
    target: str | None = None,
) -> str:
    """Build prompt for improving design based on feedback."""
    p = PROMPT_PRE
    p += "\n\n"

    if target == "pareto_scores":
        p += PROMPT_GEN_SCORE_FEEDBACK_WITH_OPT
    else:
        if COMPLEXITY_TARGETS.get(target) is None:
            raise ValueError(f"Unknown complexity target: {target}")
        else:
            prompt_template = Template(PROMPT_GEN_AST_FEEDBACK_WITH_OPT)
            p += prompt_template.substitute(complexity_target=COMPLEXITY_TARGETS[target])

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

    if design_call_graph:
        code_inputs[design_call_graph.name] = design_call_graph.read_text()

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

def build_prompt_mutate_target(design: Design, fix: bool = False, error_message: str | None = None, target: str | None = None) -> str:
    """Build prompt for targeted mutation/improvement."""
    if fix == True and error_message is not None:
        return build_prompt_fix(
            design.kernel_description_fp,
            design.h_files[0],
            design.kernel_fp,
            design.tb_file,
            design.opt_fp,
            error_message=error_message,
        )
    else:
        return build_prompt_gen_feedback(
            design.kernel_description_fp,
            design.h_files[0],
            design.kernel_fp,
            design.tb_file,
            design.call_graph_fp,
            design.opt_fp,
            design.pareto_scores_fp,
            target=target,
        )
