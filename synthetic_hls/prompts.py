from pathlib import Path
from textwrap import dedent
from typing import Dict

### Prompt components for various tasks ###
PROMPT_PRE = dedent(
    """
## Overview
You are a helpful export hardware engineer and software developer who will assist the user with hardware design tasks for high-level synthesis.
The task will center around high-level synthesis (HLS) code written in C++ for a hardware design. The HLS design is written to target the latest Vitis HLS tool from Xilinx, which maps C++ code to a Verilog implementation for FPGAs.
"""
).strip()


PROMPT_SCALABILITY_EXPLANATION = dedent(
    """
 ### What "well-scalable" means:
The design should exhibit a rich and diverse performance-resource tradeoff space when synthesized under different pragma configurations. Specifically:
- It should be sensitive to HLS directives (i.e. `pipeline`, `unroll`, `array_partition`, etc.) in ways that yield a wide range of implementations with varying resource usage and latency.
- The generated design space should contain datapoints that span from low-resource/high-latency to high-resource/low-latency implementations, allowing for a meaningful Pareto frontier to be constructed.
- Avoid overly rigid or bottlenecked structures that limit the impact of directive combinations.
Latency sensitivity requirement: The design must show material latency variation across explored points (i.e. `max(latency)/min(latency) >= 1.5` and `latency_range = max-min >= 10%` of `max`). 
- If this is not met, redesign to remove bottlenecks:
    Bottleneck audit (design-time): Identify the loop(s)/stage(s) that dominate cycles and explain how `pipeline II`, `unroll`, memory parallelism (`array_partition`/tiling), and optional work paths will change the critical path cycles.
"""
).strip()


PROMPT_GENERAL_CONSTRAINTS = dedent(
    """
### Code Clarity Constraints:
- DO NOT add any comments to any of the generated files, including:
  - C++ kernel implementation files (`.cpp`)
  - Header files (`.h`)
  - Testbench files (`_tb.cpp`)
  - Optimization templates (`opt_template.tcl`)
- All generated code must be clean, syntactically correct, and self-contained without any inline or block comments.
- The only textual explanation should be in the `kernel_description.md` file.

### General Constraints:
- DO NOT simply wrap the previous benchmark in a wrapper or rename functions.
- DO NOT re-output the exact same kernel or trivial edits.
- You must ensure the design is synthesizable, well-scalable, and modular, supporting clean hierarchy, sub-functions, templates, or structs where appropriate.
- DO NOT add any performance optimization pragmas such as `pipeline`, `unroll`, `array_partition`, `inline`, or similar in the kernel implementation file. The only pragma you are allowed to use is `#pragma HLS top name=...` to define the kernel top function.
- Ensure the total design space defined by OptDSLv2 optimization template file includes a rich spread of latency vs. resource trade-offs.
- Do not omit any part. Do not output anything other than the required seven complete code files.
    """
).strip()

PROMPT_OPTDSL_V2_REQUIREMENTS = dedent(
    """
### OptDSLv2 Guidelines:
- The OptDSLv2 optimization template file named `opt_template.tcl` is to define design space and enable design space exploration for the kernel implementation file only.
- You should update the OptDSLv2 file to match the new kernel structure.
- The OptDSLv2 format replaces TCL directives with a structured Python-like DSL that expresses HLS directives such as `pipeline`, `unroll`, and `partition`.

#### OptDSL Format Compliance Requirement:
You must follow the format, directive structure, and example configurations exactly as provided below. 
- DO NOT invent new directive types, modify argument order, or change syntax.
- DO NOT reformat the grouping logic or factor list structure.
- Do NOT omit any bracket or comma. This is required syntax.
    - Correct: `partition("array_name", "kernel", "cyclic", [1, 2, 4, 8], 1, "group_name")`
    - Incorrect: `partition "array_name" "kernel" "cyclic" [1, 2, 4, 8] 1 "group_name"` or `partition("array_name", "kernel", "cyclic", 1 2 4 8, 1, "group_name")`

#### OptDSLv2 Semantics:
- This format uses a Python-like DSL to describe directive configurations for:
    - `pipeline(label: str, function: str, optional: bool = False)`
    - `unroll(label: str, function: str, factor: list[int], group: str | None = None)`
    - `partition(array_var: str, function: str, partition_type: str, factor: list[int], dim: int, group: str | None = None)`
        - DO NOT use dimension index 0 in any `partition()` directive. Vivado HLS indexing starts from 1, and `dim=0` is invalid. 
        - The `partition_type` is fixed as `cyclic` for all partitions.

#### OptDSL Output Requirements:
1. Resource Directives
   At the beginning of the file, use standard Vivado HLS directives for memory binding and inlining.
   For example:
    ```
    set_directive_resource -core RAM_1P "kernel_name" input_array_1_name
    set_directive_resource -core RAM_1P "kernel_name" input_array_2_name

    set_directive_inline sub_function_name
    ```

2. Grouped Optimizations (Zip-Combination by Factor)
   All `partition()` and `unroll()` calls that use the same `group` name will be zipped together by factor index. 
   For example:
    ```
    partition("input", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
    partition("output", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
    unroll("loop_i", "kernel", [1, 2, 4, 8], "group_1")

    ```
    or
    ```
    partition("A", "kernel", "cyclic", [1, 2, 4, 8], 2, "partition_group_1")
    partition("B", "kernel", "cyclic", [1, 2, 4, 8], 2, "partition_group_1")
    partition("C", "kernel", "cyclic", [1, 2, 4, 8], 2, "partition_group_1")    
 
    ```       
    This creates exactly 4 variants, each using a consistent factor across the grouped elements.

    Important Guidelines:
    - If you have multiple `partition()` directives with the same factor list, group them together using a common `group` name (like `partition_group_1`) to form a zipped block. This reduces redundancy and ensures efficient design space coverage.
    - A group must contain at least two directives. DO NOT create a group with only one directive—such grouping is unnecessary and should be avoided.
    - To balance design space size and coverage, DO NOT blindly group all unrolls. Choose some directives to group (especially ones with matching behavior or structure), and keep others ungrouped.

3. Ungrouped Optimizations (Full Cross Product)
   Ungrouped `unroll()` or `partition()` calls (i.e., with `group=None`) will be cross-producted with the grouped variants.
   Pipelines can be:
    - Always enabled: `pipeline("loop_label", "kernel")`
    - Optional (explored on/off): `pipeline("loop_label", "kernel", optional=True)`
   For example:
    ```
    pipeline("loop_j", "kernel")
    pipeline("loop_j", "kernel", optional=True)
    unroll("loop_j", "kernel", [1, 2, 4])
    unroll("loop_k", "kernel", [1, 2, 4, 8])
    partition("window", "kernel", "cyclic", [1, 2, 4], 1)

    ```

4. Factor Lists Guidelines
    The list of unroll or partition factors must be compatible with loop bounds or array sizes. 
    DO NOT use symbolic constants, variables, or macros (i.e. `MAX_SIZE`, `N`, `FEATURE_DIM`, etc.).  
    All elements in the factor list must be explicit, hardcoded integers.  
    For example:
        Incorrect usage (NOT allowed):
        - `unroll("loop_k", "kernel", [MAX_WAVELETS])`
        - `partition("buffer", "kernel", "cyclic", [1, 2, FEATURE_SIZE], 1)`

        Correct usage (allowed):
        - For loops with bound 3 → recommend `[3]`
        - For loops with bound 64 → recommend `[1, 2, 4, 8]`
        - For larger loops → consider `[2, 4, 8, 16]` if suitable
    If a loop processes an array along a specific dimension, the unroll factor list for that loop and the partition factor list on that array dimension should match to avoid banking conflicts.
    All directives in the same group MUST use the exact same factor list. Avoid mismatched factor list lengths or inconsistent values across grouped directives.
    For example, this is NOT allowed:
        ```
        partition("input", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
        partition("output", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
        unroll("loop_i", "kernel", [1, 2, 4, 8], "group_1")
        unroll("loop_j", "kernel", [1, 2, 4], "group_1")  # WRONG: Mismatched factor list
        ```
        Instead, all of them must use the same full list, like [1, 2, 4, 8].

The generated `opt_template.tcl` file should:
- Begin with all `set_directive_resource` and `set_directive_inline` lines.
- Use grouped and ungrouped `partition`, `unroll`, and `pipeline` directives appropriately.
  - Apply `partition()` to major I/O and intermediate arrays.
  - Apply `unroll()` and `pipeline()` to key loops, avoiding duplication.
- Provide a diverse, structured design space while avoiding redundancy and overgrowth.

#### Constraints for the OptDSLv2 Template File:
- DO NOT apply `pipeline` or `unroll` to the same loop in both grouped and ungrouped sections.
- DO NOT apply multiple directives of the same kind (i.e. two `pipeline()` calls with and without `optional=True`, or two `partition()` directives with different factors) to the same loop or array.
- Match all loop labels and array names exactly as used in the kernel code.
- Ensure the design space has a balanced spread of performance vs. resource trade-offs.
- Ensure the consistancy of factor lists across directives in the same group.
- Keep the total number of configurations but diverse enough for performance-resource tradeoff analysis. The total number of distinct directive combinations produced from the OptDSLv2 template file should exceed 64 but be less than 256. Choose the number of blocks and factors accordingly.

    """
).strip()


COMPLEXITY_TARGETS: Dict[str, str] = {
    "max_call_chain_depth": dedent("""
        Increase this complexity dimension: maximum call-chain depth.
        Target: increase `max_call_chain_depth` by >= 1 while maintaining or increasing `average_function_lines` (from `call_graph.json`).
        - Any added depth must result from meaningful intermediate subfunctions performing real computation/control (no wrappers).
        - Substance guard: every new/modified function should have LOC >= baseline `average_function_lines` and add non-trivial logic (loops, transformations, branching, or orchestration with in-between logic).
        - No filler: do not add meaningless lines, redundant code, or micro-splits that only inflate counts.
        - Keep the top-level kernel interface unchanged.
    """).strip(),

    "num_functions": dedent("""
        Increase this complexity dimension: number of functions.
        Target: add >= 2 meaningful subfunctions while maintaining or increasing `average_function_lines` (from `call_graph.json`).
        - Substance guard: each new function must have LOC >= baseline `average_function_lines` and encapsulate real computation, data movement, or control.
        - No filler: avoid wrappers, redundant lines, or trivial splits that reduce substance or only inflate counts.
        - Keep the top-level kernel interface unchanged.
    """).strip(),

    "average_function_lines": dedent("""
        Increase this complexity dimension: average function lines.
        Target: increase `average_function_lines` (from `call_graph.json`) by >= 0.2 relative to the current baseline
                (or by >= 5 LOC if the baseline < 25), without reducing synthesizability.
        - Strategy (substantive ways to add LOC):
          - Add meaningful computation blocks: tiled inner loops, reductions, windowed/stencil ops, prefix/suffix transforms.
          - Introduce mid-level orchestration: explicit buffer staging, boundary handling, loop-carried state, or reduction trees.
          - Replace opaque one-liners with explicit staged steps and intermediate values that HLS can analyze.
        - Substance guard: every modified function must include at least one non-trivial element (loop, branch, reduction,
          data reorganization) and should meet or exceed the baseline `average_function_lines`.
        - No filler: avoid dead code, no-op variables, redundant copies, excessive parameter padding, or comment-only inflation.
    """).strip(),
}


### Final combined prompts for generation tasks ###
PROMPT_GEN_OPTDSL_V2 = dedent(
    f"""
## Task Description
You are given an HLS design including its implementation file, header file, testbench and natural language description.

Your task is to generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only. 
The testbench file is not relevant for optimization and should not be considered. Do not modify any given files, only generate the new `opt_template.tcl` file.

{PROMPT_OPTDSL_V2_REQUIREMENTS}"""
).strip()


PROMPT_GEN_NO_INPUT_WITH_OPT = dedent(
    f"""
## Task Description
Your task is to:
1. Design and implement a new benchmark as a well-scalable, high-complexity, multi-stage, application-level HLS-compatible C++ kernel.
2. Write a matching C++ header file for the design.
3. Create a testbench that can validate the functionality of the design.
4. Output the name of the top-level function in a file named `top.txt`.
5. Write a markdown file `kernel_description.md` that provides a concise, human-readable natural language description of the generated benchmark, explaining its functionality, purpose, inputs, and outputs.
6. Generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only. The testbench file is not relevant for optimization and should not be considered.
7. Write a hls_eval_config.toml file tagging the design with:
    - `tags = ["llm_gen"]`

{PROMPT_SCALABILITY_EXPLANATION}

### Important Constraints for the Benchmark Design:
- You must create an entirely new and original benchmark. DO NOT copy, approximate, re-implement, or repackage any pre-existing benchmarks (i.e. PolyBench, CHStone, MachSuite, or other known datasets).
- The functionality and algorithm of your benchmark must be unique and created from scratch.
- If multiple benchmarks are requested, each must represent a different design concept and algorithmic structure. Avoid duplicating the same kernel logic across samples.
- The total size of the design must be moderate and practical for synthesis. Avoid extremely large loop bounds, excessive buffer sizes, or deeply nested control that could cause long compile times or unrealistic synthesis results. Typically, loop bounds should be smaller than 128.
- Anti-stall rule: Do not create a single outer loop that is always `pipeline II=1` with trivial inner work; such structures pin latency. Balance work across stages so changing `unroll`/`II` actually moves total cycles.

### Structural Requirements for the Benchmark Design:
- You must build a full, multi-stage application accelerator, not just a reusable module. It should involve multiple sub-functions or kernels, realistic data access and compute dependencies, and ideally feature both compute-intensive and logic-driven stages.
- Your generated benchmark must not be limited to simple filters, matrix operations, or textbook-style kernels.
- You are encouraged to model your generated benchmark as a complete pipeline, integrating at least 4 or 6 sub-functions or logical modules that reflect realistic data flow, memory usage, control logic, and computational diversity. Each benchmark should serve a clear, self-contained purpose within a real-world scenario. 

- Examples of stage types:
  - Sensor ingestion / buffering
  - Feature extraction or transformation
  - State update / memory interaction
  - Control decision / scheduling logic
  - Post-processing or response generation

- Example real-world application contexts include (but are not limited to):
- **Surgical Assistance System**: Real-time control of a robotic endoscope navigating a vascular model, integrating pressure sensor fusion, path planning based on local vessel curvature, and actuator signal sequencing with fault detection.
- **Autonomous Aerial Survey Platform**: An end-to-end pipeline for drone-based terrain analysis: onboard stereo image alignment, disparity-based depth estimation, obstacle detection, and real-time waypoint re-routing using GPS + IMU data fusion.
- **Industrial Predictive Maintenance Controller**: Edge-deployed accelerator managing rotating machinery telemetry. It performs time-domain signal buffering, FFT-based feature extraction, rolling anomaly scoring with historical context, and machinery state reporting for downstream scheduling systems.
- **Onboard Satellite Communication Scheduler**: Solves a constrained task assignment problem for satellite downlink jobs. The kernel handles task queuing, visibility window filtering, priority-aware slot assignment, and thermal/power constraint enforcement in a time-slotted framework.
- **Multi-Agent Swarm Coordination Kernel**: Simulates and controls agent motion in a distributed swarm system. Handles neighbor sensing, consensus update rules, collision avoidance logic, and group target convergence in bounded grid space.
- **Medical Diagnostic Assistant**: Accelerates diagnosis logic from biosensor streams. It performs real-time signal filtering, multi-parameter health scoring, diagnostic rule matching, and patient state alerting with adjustable thresholds.

- The design should be non-trivial, meaning it includes both compute-heavy operations and non-trivial data dependencies. The complexity should allow rich exploration under HLS directive tuning.
- The kernel must process structured data, contain multiple computation layers or phases, and represent a self-contained functional pipeline.
- All loops must have static bounds analyzable by synthesis tools. Avoid dynamic memory, recursion, or unbounded loops.
- All `for` loops in the kernel must be clearly labeled using the syntax `<label>: for (...)`, where the label is unique and descriptive. Do not leave any loop unlabeled.

{PROMPT_OPTDSL_V2_REQUIREMENTS}

{PROMPT_GENERAL_CONSTRAINTS}
"""
).strip()


PROMPT_GEN_SINGLE_INPUT_WITH_OPT = dedent(
     f"""
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
6. Generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only. The testbench file is not relevant for optimization and should not be considered.
7. Write a hls_eval_config.toml file tagging the design with:
    - `tags = ["llm_gen"]`

{PROMPT_SCALABILITY_EXPLANATION}

#### Measurement of Scalability:
Scalability is quantitatively evaluated using Pareto scores, which measure how effectively a benchmark design spans the tradeoff space between performance (latency) and hardware resource usage LUTs or FFs). 
Each score is computed from 64 synthesized design points generated using the corresponding opt_template.tcl file.
- Lower Pareto scores are better, indicating that the design supports a wide range of tunable tradeoffs and yields a smooth, continuous Pareto frontier.
- A score below 0.35 is considered excellent, reflecting strong directive sensitivity and a well-balanced optimization space.
- A score above 0.6 typically suggests poor scalability either due to rigid structures, design bottlenecks, or weak responsiveness to directive tuning.
- The Pareto score is computed using:
    pareto_score = (distance_to_corners + max_gap) / (distance_to_corners + total_curve_length)
    Where:
    - distance_to_corners = how close the Pareto frontier reaches the ideal tradeoff corners (low latency / low resource).
    - max_gap = the largest gap between neighboring Pareto-optimal points.
    - total_curve_length = total length of the frontier curve (to normalize for scale).
Goal: Your generated benchmark should aim to match or surpass the scalability of the reference benchmark, achieving comparable or smaller Pareto scores on both LUTs vs. latency and FFs vs. latency.

#### Guidance for Benchmark Design:
Use the provided reference benchmark and its associated scores as a guide:
- Your generated benchmark should aim to be longer, more comprehensive, and structurally richer than the reference benchmark wherever possible.
- Study which structures, memory organizations, and loop hierarchies lead to lower Pareto scores.
- Design your new benchmark to match or surpass the scalability and architectural diversity of the reference, aiming for comparable or smaller Pareto scores on both LUTs vs. latency and FFs vs. latency.
- Analyze patterns in the reference `opt_template.tcl` files and design stage breakdowns to guide your own benchmark decomposition.

### Important Constraints for the Benchmark Design:
- You must create an entirely new and original benchmark. DO NOT copy, approximate, re-implement, or repackage any given or pre-existing benchmarks (i.e. PolyBench, CHStone, MachSuite, or other known datasets).
- The functionality and algorithm of your benchmark must be unique and created from scratch.
- If multiple benchmarks are requested, each must represent a different design concept and algorithmic structure. Avoid duplicating the same kernel logic across samples.
- The total size of the design must be moderate and practical for synthesis. Avoid extremely large loop bounds, excessive buffer sizes, or deeply nested control that could cause long compile times or unrealistic synthesis results. Typically, loop bounds should be smaller than 128.

### Structural Requirements for the Benchmark Design:
- You must generate an entirely new and original benchmark that reflects a complete, multi-stage application accelerator, not a reusable or toy kernel.
- The benchmark should represent a complete real-world application case, such as audio processing, image processing, robotics, digital signal processing, or other domains involving structured data, control logic, and computational diversity.
- Designs must be functionally meaningful, integrating at least 6 interconnected sub-functions (i.e. filtering, modulation, decoding, transformation, scheduling, or state control).
- Benchmarks must include both compute-heavy and control-intensive stages with static loop bounds. Avoid trivial operations, isolated filters, or deeply nested control with impractical loop sizes.
- All for loops, including nested and inner loops, must be explicitly labeled using the syntax loop_label: for (...). This applies to every loop in the kernel, regardless of its depth or complexity. Loop bounds must be static and analyzable by HLS tools.

{PROMPT_OPTDSL_V2_REQUIREMENTS}

{PROMPT_GENERAL_CONSTRAINTS}
"""
).strip()


PROMPT_GEN_FIX_WITH_OPT = dedent(
    f"""
You are provided with a previously generated HLS benchmark that encountered synthesis errors.

Your task is to:
1. Fix the synthesis errors in the provided benchmark.
2. Regenerate all files with the corrections applied.
3. Ensure the design remains fully synthesizable by Vitis HLS.

### Error Information:
$error_message

### Critical Constraints:
- You must ensure the redesigned benchmark remains fully synthesizable.
- All syntax, memory usage, control structures must comply with Vitis HLS compatibility.
- DO NOT add comments to the generated files.
- All generated code must be clean and syntactically correct.

After fixing the code, regenerate the updated OptDSLv2 optimization template file `opt_template.tcl` file that matches the corrected kernel structure and defines the proper design space.
{PROMPT_OPTDSL_V2_REQUIREMENTS}

{PROMPT_GENERAL_CONSTRAINTS}
"""
).strip()


PROMPT_GEN_AST_FEEDBACK_WITH_OPT = dedent(
    f"""
You are provided with a previously generated HLS design, including:
- A C++ kernel implementation file describing a synthesizable, modular application-level accelerator.
- A matching C++ header file, testbench, top-level function name, markdown description, and OptDSLv2 optimization file.
- A `call_graph.json` that summarizes current function hierarchy and calls:
    This JSON provides a minimal view of the current hierarchy
    - `num_functions` (int): number of unique function names currently detected.
    - `max_call_chain_depth` (int): longest path length (in nodes) from the top-level caller to any leaf callee.
    - `functions` (list[str]): names of functions present in the current design.
    - `kernel_total_lines` (int): total lines of code in the kernel.
    - `function_line_counts` (dict[str, int]): per-function lines of code (LOC) measured from the AST.
    - `average_function_lines` (float): mean LOC across functions. Treat this as a primary, quantitative complexity KPI; the redesign must keep or improve it.
    - `edges` (list[{{"caller": str, "callee": str}}]): directed edges; each means `caller()` invokes `callee()`.

Your task is to:
1. You must meaningfully redesign and improve kernel complexity (structure, content, logic) along this single target dimension:
$complexity_target
while not regressing the other complexity metrics from `call_graph.json`. Specifically, when you improve one target, you must at least keep the other metrics like `num_functions`, `max_call_chain_depth`, `average_function_lines` etc. (use the provided JSON as the baseline)
- All improvements must be substantive and practically justified. No wrappers or cosmetic edits.
- Workflow compliance is MANDATORY: All redesign and code generation must adhere to the Design-first workflow below. Deviation is considered incorrect.
2. Fix or refine any unclear, redundant, trivial, or potentially invalid code and design structures if present in the input files.
3. Generate updated versions of all files (kernel, header, testbench, top.txt, kernel_description.md, OptDSLv2) reflecting the improved design.
4. Write a hls_eval_config.toml file tagging the design with:
    - `tags = ["llm_gen"]`

## Design-first workflow (MANDATORY)
### Step 1 - Read & revise the functional spec (Design-first, MANDATORY)
1. Carefully read the existing `kernel_description.md` to understand the current functionality, data model, pipeline, and constraints.  
2. Produce a Revised Kernel Description (R-KD) that refines and, where helpful, expands or rewrites the kernel's functionality within the same application domain to enable meaningful complexity growth while staying realistic for HLS.

The R-KD must include:
- **Goals & Target:** What you will change to improve `$complexity_target` and how you will keep or raise the other complexity metrics (i.e. `num_functions`, `max_call_chain_depth`, `average_function_lines`).
- **Functional Overview:** Clear end-to-end behavior and any functional extensions/refinements you introduce (brief rationale).
- **Dataflow & Staging:** Phases, fixed-size buffers/tiling, branching/reduction steps, and inter-stage interactions.
- **Function Inventory (proposed):** List of functions/modules with roles and non-trivial estimated logic (loops/branches). Avoid micro-splits; each function must perform substantive work.
- **Control/Datapath Structure:** Orchestration logic explaining how deeper call chains or larger function sets arise meaningfully.
- **Directive Sensitivity Plan:** Where/why `pipeline`, `unroll`, `array_partition`, etc., will induce a rich tradeoff space.
- **Interface & I/O Plan:** You may change the top-level interface and I/O semantics
  (number/order of ports, data types/bit-widths, shapes, streaming vs memory)
  to enable a better design, provided that:
    - All sizes are compile-time constants (no dynamic allocation).
    - The interface is HLS-compatible (i.e. static arrays, `hls::stream`, `ap_int`).
    - `top.txt`, header, and testbench are updated consistently
    - You include an I/O Migration Map detailing old→new mapping, rationale, and required test updates.
- **Synthesis Constraints:** Static loop bounds; no recursion or dynamic allocation; full Vitis HLS compatibility.
- **Practicality Justification:** For each added/modified function or structure, briefly state the practical benefit (i.e. improves locality/reuse, exposes tunable II/unroll, enables tiling), and indicate how it is expected to improve the target and kernel complexity.

The R-KD is the ground truth. Generate code only after finalizing the R-KD, and ensure the implementation faithfully matches it.
The final R-KD must be saved as `kernel_description.md`. Do not emit alternative filenames.

### Step 2 - Implement the revised design
Use the R-KD to regenerate all artifacts:
- Updated `kernel_description.md` reflecting your revised functionality.
- Kernel `.cpp`, header `.h`, testbench, `top.txt`, and OptDSLv2 file.
- Begin Step 2 only after Step 1 is complete. Implement the R-KD faithfully; if you must deviate, update the R-KD first and then proceed. Do not "add code in place" without the spec reflecting it.


## Substance/complexity rules:
- Increase the requested target meaningfully.
- Cross-metric non-regression: Regardless of target, do not reduce the other metrics from their baseline values in `call_graph.json`; increasing them is preferred.
- Each new function should have LOC >= baseline `average_function_lines`.
- No artificial LOC inflation: Do not increase `average_function_lines` via comments, whitespace, dead code, no-ops, or superficial scaffolding. Line growth must come from real computation/control.
    - **Explicit no-op ban:** No statements like `a = a + 0`, `x *= 1`, `y = y | 0`, dummy branches that always execute the same path, or dead stores/reads added only to raise LOC.
    - **Wrapper ban:** Do not add pass-through functions whose sole purpose is to call another function without in-between logic (buffering, indexing transforms, boundary handling, reduction staging, etc.).
    - **Meaningful structure only:** Added loops/branches must change dataflow, reuse, or timing (i.e. tiling for locality, partial partition for bandwidth, staged reductions) and be verifiable by changed HLS metrics for at least one OptDSL configuration.
- Non-Triviality Gate (new/split functions must satisfy >= 1):
    - Implements a distinct algorithmic sub-stage (i.e. multi-step transform, reduction/aggregation, windowed pass) with static bounds, or
    - Contains substantive control or dataflow (meaningful loop/branch that affects results), or
    - Orchestrates >= 2 non-trivial subcalls and performs in-between logic (buffering/indexing/combining).
- Encourage deep, meaningful structure: tiling/phasing, fixed-size buffering, staged transforms, deterministic branching; no superficial layering.

## Structural Requirements for the Benchmark Redesign:
- The design should be non-trivial, meaning it includes both compute-heavy operations and non-trivial data dependencies. The complexity should allow rich exploration under HLS directive tuning.
- The kernel must process structured data, contain multiple computation layers or phases, and represent a self-contained functional pipeline.
- All loops must have static bounds analyzable by synthesis tools. Avoid dynamic memory, recursion, or unbounded loops.
- All `for` loops in the kernel must be clearly labeled using the syntax `<label>: for (...)`, where the label is unique and descriptive. Do not leave any loop unlabeled.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.

## Critical Constraints:
- Only include comments in the generated C++ code and `opt_template.tcl` where they are essential for understanding complex behavior or assumptions. Avoid excessive, obvious, or redundant comments. Keep all code and directives clean and focused.
- You must ensure that the redesigned benchmark remains fully synthesizable by Vitis HLS.  
- All syntax, memory usage, control structures, and function constructs must comply with Vitis HLS compatibility. Designs that fail synthesis due to invalid constructs are not acceptable.
- Every change must add real algorithmic/architectural value (see Complexity principles above).
- Workflow compliance is mandatory: Skipping Step 1, mixing spec and code, or emitting code before the R-KD section is considered invalid output.
- Complexity KPI: Treat `average_function_lines` as a direct complexity measure to be kept or improved; any increase must result from meaningful algorithmic content, not padding.


{PROMPT_OPTDSL_V2_REQUIREMENTS}

{PROMPT_GENERAL_CONSTRAINTS}
"""
).strip()


PROMPT_GEN_SCORE_FEEDBACK_WITH_OPT = dedent(
f"""
You are provided with:
- A previously generated HLS benchmark, including kernel (.cpp), header (.h), testbench, top function name (top.txt), markdown description, and an OptDSLv2 optimization file (opt_template.tcl).
- A JSON feedback report named `pareto_scores_summary.json` (authoritative), which contains detailed Pareto analysis for LUTs-vs-latency and FFs-vs-latency:
  {{
    "LUTs_vs_latency": {{
      "pareto_score": <float or null>, may be null (None) if the Pareto frontier has < 2 points
      "n_points": <int>,
      "n_pareto_frontier_points": <int>,
      "resource_range": [min, max],
      "latency_range": [min, max],
      "start_point_to_corner": <float>,
      "end_point_corner": <float>,
      "max_gap/curve_length": <float>,
      "max_gap_points": [[r1,l1], [r2,l2]]
    }},
    "FFs_vs_latency": {{ ... same fields ... }}
  }}

Goal:
Redesign and regenerate the benchmark so that BOTH Pareto scores in `pareto_scores_summary.json` become numerically LOWER than before (e.g., if 0.52, target < 0.52; ideally ≤ 0.35), while keeping the design fully synthesizable in Vitis HLS.
- Use ONLY `pareto_scores_summary.json` for quantitative guidance. If any other score files exist (e.g., `pareto_scores.txt`), ignore them.

Your tasks:
1. Improve scalability so both reported scores in the next `pareto_scores_summary.json` decrease.
2. Maintain functional correctness and synthesizability.
3. Address the specific issues implied by `pareto_scores_summary.json`:
   - If `n_pareto_frontier_points` is small (< 10) → enrich design space to create more tradeoff points.
   - If `max_gap/curve_length` is large (> 0.2)→ reduce large gaps so the frontier is smoother.
   - If `resource_range` or `latency_range` is narrow → widen tunability via loop tiling/unrolling/partitioning and pipeline placement.
   - If `start_point_to_corner` or `end_point_corner` is large (> 0.2) → push the frontier closer to ideal corners (low-lat/low-res and vice versa).
4. Regenerate ALL files (kernel, header, testbench, top.txt, kernel description, OptDSLv2).
5. Write `hls_eval_config.toml` with: tags = ["llm_gen"].

{PROMPT_SCALABILITY_EXPLANATION}

#### Measurement of Scalability:
Scalability is quantitatively evaluated using Pareto scores, which measure how effectively a benchmark design spans the tradeoff space between performance (latency) and hardware resource usage LUTs or FFs). 
Each score is computed from 64 synthesized design points generated using the corresponding opt_template.tcl file.
Numerically lower Pareto scores are better, indicating that the design supports a wide range of tunable tradeoffs and yields a smooth, continuous Pareto frontier.
- The Pareto score is computed using:
    pareto_score = ((start_point_to_corner + end_point_corner) + max_gap) / ((start_point_to_corner + end_point_corner) + curve_length)
- A score below 0.35 is considered excellent, reflecting strong directive sensitivity and a well-balanced optimization space.
- A score above 0.6 typically suggests poor scalability either due to rigid structures, design bottlenecks, or weak responsiveness to directive tuning.

### Redesign Guidance (tie your edits to `pareto_score_summary.json`)
If score is None → must prioritize generating valid points(n_pareto_frontier_points) first before minimizing the score.
- Increase frontier coverage and density (more points on the Pareto frontier).
- Smooth the frontier (reduce `max_gap/curve_length`).
- Improve directive sensitivity (loop labels, partitioned arrays, controlled pipelining/unrolling).
- Keep loops statically bounded and the design modular and non-trivial

### Structural Requirements for the Benchmark Redesign:
- The updated design should be non-trivial, meaning it includes both compute-heavy operations and non-trivial data dependencies. The complexity should allow rich exploration under HLS directive tuning.
- The kernel must process structured data, contain multiple computation layers or phases, and represent a self-contained functional pipeline.
- All loops must have static bounds analyzable by synthesis tools. Avoid dynamic memory, recursion, or unbounded loops.
- All `for` loops in the kernel must be clearly labeled using the syntax `<label>: for (...)`, where the label is unique and descriptive. Do not leave any loop unlabeled.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.

{PROMPT_OPTDSL_V2_REQUIREMENTS}

{PROMPT_GENERAL_CONSTRAINTS}
"""
).strip()