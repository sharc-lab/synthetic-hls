from pathlib import Path
from textwrap import dedent

from hls_eval.data import BenchmarkCase
from hls_eval.prompting import build_input_code_prompt_xml

prompt_synth_prep = dedent("""
## Task Description
Your task is to edit the given user's HLS code to prepare it for synthesis using Vitis HLS.
The code should be modified to ensure that it is synthesizable by the Vitis HLS tool.
                           
All C-style arrays need and array function arguments need to be declared and passed as fixed sized arrays.
For example `int* arr` should be converted to `int arr[SIZE]` and any functions that take `int* arr` as an argument should be changed to take `int arr[SIZE]` as an argument.
This also needs to be adapted for multi-dimensional arrays.
If any array dimension size parameters are needed for the fixed-size arrays, they should be defined as constants at the top of the file.
All arrays needed to have explicit sizes. For example, `int arr[]` should not be used; instead, use `int arr[SIZE]`.

There should be no recursion in the code; if so, refactor the code to remove recursion.
There should be no printf or sprintf statements in the code; if so, comment them out.
There should be no dynamic memory allocation in the code; if so, refactor the code to remove dynamic memory allocation and use fixed-size arrays.
There should be no usage of pointers, pointer dereferencing, or pointer arithmetic in the code; if so, refactor the code to remove pointers.

Floating-point data types and operations should be converted to fixed-point data types and operations when appropriate.
Please only do this conversion for `float` and double `types`.

All loops should have loop labels added to them using the `label: statement` syntax.
                           
Inserting Vitis HLS pragmas as needed (i.e. using `#pragma HLS ...`).

Please complete all the above steps to prepare the code for synthesis.
Do not over-optimize the code; the goal is to make the code synthesizable and slightly optimized as a starting point for further optimization.

You do not need to include any testbench code modifications in the output; these file (`*_tb.cpp`) should not be listed in the output.
""").strip()


prompt_pre = dedent(
    """
## Overview
You are a helpful export hardware engineer and software developer who will assist the user with hardware design tasks for high-level synthesis.
The task will center around high-level synthesis (HLS) code written in C++ for a hardware design. The HLS design is written to target the latest Vitis HLS tool from Xilinx, which maps C++ code to a Verilog implementation for FPGAs.
"""
).strip()

prompt_gen = dedent(
    """
## Task Description
Given a natural language description of an HLS design, a pre-written C++ design header, and a pre-written C++ testbench, generate the C++ implementation of the HLS design that aligns with the natural language description.

It should be functionally equivalent to the natural language description, be consistent with the provided header file, and pass the testbench. The design should also be synthesizable by the HLS tool.

Only generate the code for the design; do not modify the header file or the testbench. Make sure to import the header file as well.

Provide the complete design code in the single output; do not omit anything or leave placeholders.

Hierarchical design, sub-functions, template functions, structs, typedefs, and define statements are allowed but should be used only if appropriate.
"""
).strip()

prompt_optdsl_v2_requirements = dedent(
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

prompt_gen_single_input_with_opt = dedent(
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

### What "well-scalable" means:
The design should exhibit a rich and diverse performance-resource tradeoff space when synthesized under different pragma configurations. Specifically:
- It should be sensitive to HLS directives (i.e. `pipeline`, `unroll`, `array_partition`, etc.) in ways that yield a wide range of implementations with varying resource usage and latency.
- The generated design space should contain datapoints that span from low-resource/high-latency to high-resource/low-latency implementations, allowing for a meaningful Pareto frontier to be constructed.
- Avoid overly rigid or bottlenecked structures that limit the impact of directive combinations.

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

{prompt_optdsl_v2_requirements}

### Code Clarity Requirement
- DO NOT add any comments to any of the generated files, including:
  - C++ kernel implementation files (`.cpp`)
  - Header files (`.h`)
  - Testbench files (`_tb.cpp`)
  - Optimization templates (`opt_template.tcl`)
- All generated code must be clean, syntactically correct, and self-contained without any inline or block comments.
- The only textual explanation should be in the `kernel_description.md` file.

### General Constraints:
- You must ensure the design is synthesizable, well-scalable, and modular, supporting clean hierarchy, sub-functions, templates, or structs where appropriate.
- You must ensure the total design space defined by OptDSLv2 optimization template file includes a rich spread of latency vs. resource trade-offs.
- DO NOT add any performance optimization pragmas such as `pipeline`, `unroll`, `array_partition`, `inline`, or similar in the kernel implementation file. The only pragma you are allowed to use is `#pragma HLS top name=...` to define the kernel top function.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.
- Hierarchical design, sub-functions, template functions, structs, typedefs, and define statements are allowed but should be used only if appropriate.
- Do not omit any part. Do not output anything other than the required five complete code files.
"""
).strip()

prompt_gen_no_input_with_opt = dedent(
    f"""
## Task Description
Your task is to:
1. Design and implement a new benchmark as a well-scalable, high-complexity, multi-stage, application-level HLS-compatible C++ kernel.
2. Write a matching C++ header file for the design.
3. Create a testbench that can validate the functionality of the design.
4. Output the name of the top-level function in a file named `top.txt`.
5. Write a markdown file `kernel_description.md` that provides a concise, human-readable natural language description of the generated benchmark, explaining its functionality, purpose, inputs, and outputs.
6. Generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only. The testbench file is not relevant for optimization and should not be considered.

### What "well-scalable" means:
The design should exhibit a rich and diverse performance-resource tradeoff space when synthesized under different pragma configurations. Specifically:
- It should be sensitive to HLS directives (i.e. `pipeline`, `unroll`, `array_partition`, etc.) in ways that yield a wide range of implementations with varying resource usage and latency.
- The generated design space should contain datapoints that span from low-resource/high-latency to high-resource/low-latency implementations, allowing for a meaningful Pareto frontier to be constructed.
- Avoid overly rigid or bottlenecked structures that limit the impact of directive combinations.

### Important Constraints for the Benchmark Design:
- You must create an entirely new and original benchmark. DO NOT copy, approximate, re-implement, or repackage any pre-existing benchmarks (i.e. PolyBench, CHStone, MachSuite, or other known datasets).
- The functionality and algorithm of your benchmark must be unique and created from scratch.
- If multiple benchmarks are requested, each must represent a different design concept and algorithmic structure. Avoid duplicating the same kernel logic across samples.
- The total size of the design must be moderate and practical for synthesis. Avoid extremely large loop bounds, excessive buffer sizes, or deeply nested control that could cause long compile times or unrealistic synthesis results. Typically, loop bounds should be smaller than 128.

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

{prompt_optdsl_v2_requirements}

### Code Clarity Requirement
- In the generated C++ and header code: only include comments where essential for understanding complex logic or assumptions. Avoid excessive or trivial comments.
- DO NOT add any comments to the `opt_template.tcl` file. The file must only contain valid OptDSLv2 directives without any inline or block comments.

### General Constraints:
- You must ensure the design is synthesizable, well-scalable, and modular, supporting clean hierarchy, sub-functions, templates, or structs where appropriate.
- DO NOT add any performance optimization pragmas such as `pipeline`, `unroll`, `array_partition`, `inline`, or similar in the kernel implementation file. The only pragma you are allowed to use is `#pragma HLS top name=...` to define the kernel top function.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.
- Hierarchical design, sub-functions, template functions, structs, typedefs, and define statements are allowed but should be used only if appropriate.
- Do not omit any part. Do not output anything other than the required six complete code files.
"""
).strip()

########################################## FEEDBACK ###########################################
prompt_gen_feedback_iter_with_opt = dedent(
    f"""
You are provided with a previously generated HLS benchmark, including:
- A C++ kernel implementation file describing a synthesizable, modular application-level accelerator.
- A matching C++ header file, testbench, top-level function name, markdown description, and OptDSLv2 optimization file.

Your task is to:
1. Redesign and regenerate the given benchmark with significantly increased length, structural depth, real-world relevance, and application completeness: making it longer, more complex, and more meaningful in reflecting real-world use cases.
2. Preserve its well-scalable directive sensitivity and overall synthesis compatibility, while expanding it meaningfully.
3. Fix or refine any unclear, redundant, trivial, or potentially invalid code and design structures if present in the input files.
4. Generate updated versions of all files (kernel, header, testbench, top.txt, kernel description, OptDSLv2) reflecting the improved design.

### Critical Constraint:
- Only include comments in the generated C++ code and `opt_template.tcl` where they are essential for understanding complex behavior or assumptions. Avoid excessive, obvious, or redundant comments. Keep all code and directives clean and focused.
- You must ensure that the redesigned benchmark remains fully synthesizable by Vitis HLS.  
- All syntax, memory usage, control structures, and function constructs must comply with Vitis HLS compatibility. Designs that fail synthesis due to invalid constructs are not acceptable.

### What "well-scalable" means:
The design should exhibit a rich and diverse performance-resource tradeoff space when synthesized under different pragma configurations. Specifically:
- It should be sensitive to HLS directives (i.e. `pipeline`, `unroll`, `array_partition`, etc.) in ways that yield a wide range of implementations with varying resource usage and latency.
- The generated design space should contain datapoints that span from low-resource/high-latency to high-resource/low-latency implementations, allowing for a meaningful Pareto frontier to be constructed.
- Avoid overly rigid or bottlenecked structures that limit the impact of directive combinations.

### Redesign Instructions:
- Build upon the previous kernel as a foundation. Keep useful structures, reuse well-designed modules, and extend with new functional stages.
- Introduce additional sub-modules, loop structures, data dependencies, or pipeline stages that reflect real-world applications.
- The updated design must be substantially longer and more complex than the input, introducing meaningful functionality and richer behavior.
- Do not simply inflate the design with shallow or artificial operations. All additions must serve a clear purpose in a realistic application context.
- Do not merely insert dummy code. All additions should be coherent, functional, and realistic for HLS-based synthesis.
- Ensure the new design adheres to constraints for synthesis feasibility, while being non-trivial and multi-phase.

### Structural Requirements for the Benchmark Redesign:
- The design should be non-trivial, meaning it includes both compute-heavy operations and non-trivial data dependencies. The complexity should allow rich exploration under HLS directive tuning.
- The kernel must process structured data, contain multiple computation layers or phases, and represent a self-contained functional pipeline.
- All loops must have static bounds analyzable by synthesis tools. Avoid dynamic memory, recursion, or unbounded loops.
- All `for` loops in the kernel must be clearly labeled using the syntax `<label>: for (...)`, where the label is unique and descriptive. Do not leave any loop unlabeled.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.

{prompt_optdsl_v2_requirements}

### Code Clarity Requirement
- DO NOT add any comments to any of the generated files, including:
  - C++ kernel implementation files (`.cpp`)
  - Header files (`.h`)
  - Testbench files (`_tb.cpp`)
  - Optimization templates (`opt_template.tcl`)
- All generated code must be clean, syntactically correct, and self-contained without any inline or block comments.
- The only textual explanation should be in the `kernel_description.md` file.

General Constraints:
- DO NOT simply wrap the previous benchmark in a wrapper or rename functions.
- DO NOT re-output the exact same kernel or trivial edits.
- The new kernel must represent a meaningful functional and structural improvement over the previous.
- You must ensure the design is synthesizable, well-scalable, and modular, supporting clean hierarchy, sub-functions, templates, or structs where appropriate.
- DO NOT add any performance optimization pragmas such as `pipeline`, `unroll`, `array_partition`, `inline`, or similar in the kernel implementation file. The only pragma you are allowed to use is `#pragma HLS top name=...` to define the kernel top function.
- Ensure the total design space defined by OptDSLv2 optimization template file includes a rich spread of latency vs. resource trade-offs.
- Do not omit any part. Do not output anything other than the required six complete code files.
"""
).strip()

prompt_gen_score_feedback_iter_with_opt = dedent(
    f"""
You are provided with:
- A previously generated HLS benchmark, including:
  - A C++ kernel implementation file describing a synthesizable, modular application-level accelerator.
  - A matching C++ header file, testbench, top-level function name, markdown description, and OptDSLv2 optimization file.
- A `pareto_score.txt` file containing two scalar metrics that quantify the scalability of the benchmark. Each metric is of the form:
    - `pareto_score_LUTs_vs_latency = <float>`
    - `pareto_score_FFs_vs_latency = <float>`
    These reflect how effectively the benchmark spans a tradeoff frontier between resource usage (LUTs, FFs) and performance (latency). 
    Numerically smaller/lower scores indicate better scalability.

Your goal is to update the design with better scalability, so that both Pareto scores in the new design are numerically lower than those in the provided pareto_score.txt (e.g, if the original score is 0.52, the new score should be < 0.52), ideally approaching or below 0.35.

Your task is to:
1. Redesign and regenerate the given benchmark so that it achieves numerically lower/smaller Pareto scores than those in the provided `pareto_score.txt`, for both LUTs vs. latency and FFs vs. latency.
2. Preserve synthesis compatibility and ensure that the design remains fully functional and synthesizable in Vitis HLS.
3. Fix any issues that limit scalability, such as:
   - Overly rigid structures that reduce sensitivity to `pipeline`, `unroll`, or `partition` directives.
   - Design bottlenecks that prevent latency-resource tradeoff variations.
4. Generate updated versions of all files (kernel, header, testbench, top.txt, kernel description, OptDSLv2) reflecting the improved scalability.

### What "well-scalable" means:
The design should exhibit a rich and diverse performance-resource tradeoff space when synthesized under different pragma configurations. Specifically:
- It should be sensitive to HLS directives (i.e. `pipeline`, `unroll`, `array_partition`, etc.) in ways that yield a wide range of implementations with varying resource usage and latency.
- The generated design space should contain datapoints that span from low-resource/high-latency to high-resource/low-latency implementations, allowing for a meaningful Pareto frontier to be constructed.
- Avoid overly rigid or bottlenecked structures that limit the impact of directive combinations.

#### Measurement of Scalability:
Scalability is quantitatively evaluated using Pareto scores, which measure how effectively a benchmark design spans the tradeoff space between performance (latency) and hardware resource usage LUTs or FFs). 
Each score is computed from 64 synthesized design points generated using the corresponding opt_template.tcl file.
Numerically lower Pareto scores are better, indicating that the design supports a wide range of tunable tradeoffs and yields a smooth, continuous Pareto frontier.
- A score below 0.35 is considered excellent, reflecting strong directive sensitivity and a well-balanced optimization space.
- A score above 0.6 typically suggests poor scalability either due to rigid structures, design bottlenecks, or weak responsiveness to directive tuning.
- The Pareto score is computed using:
    pareto_score = (distance_to_corners + max_gap) / (distance_to_corners + total_curve_length)
    Where:
    - distance_to_corners = how close the Pareto frontier reaches the ideal tradeoff corners (low latency / low resource).
    - max_gap = the largest gap between neighboring Pareto-optimal points.
    - total_curve_length = total length of the frontier curve (to normalize for scale).
Goal: Ensure both LUTs-vs-latency and FFs-vs-latency Pareto scores in the updated design are numerically lower than in the original design.

### Redesign Instructions for Lower Pareto Scores:
- Increase the number of design points lying on the Pareto frontier.
- Reduce large gaps in the frontier by ensuring directive variations produce incremental performance/resource tradeoffs.
- Adjust loop structures, array partitions, and pipelining opportunities to improve directive sensitivity without breaking functionality.

### Structural Requirements for the Benchmark Redesign:
- The updated design should be non-trivial, meaning it includes both compute-heavy operations and non-trivial data dependencies. The complexity should allow rich exploration under HLS directive tuning.
- The kernel must process structured data, contain multiple computation layers or phases, and represent a self-contained functional pipeline.
- All loops must have static bounds analyzable by synthesis tools. Avoid dynamic memory, recursion, or unbounded loops.
- All `for` loops in the kernel must be clearly labeled using the syntax `<label>: for (...)`, where the label is unique and descriptive. Do not leave any loop unlabeled.
- The header must declare all interfaces and top-level functions clearly.
- The testbench must initialize inputs, invoke the top-level function, and validate outputs with representative test cases.

{prompt_optdsl_v2_requirements}

### Code Clarity Requirement
- DO NOT add any comments to any of the generated files, including:
  - C++ kernel implementation files (`.cpp`)
  - Header files (`.h`)
  - Testbench files (`_tb.cpp`)
  - Optimization templates (`opt_template.tcl`)
- All generated code must be clean, syntactically correct, and self-contained without any inline or block comments.
- The only textual explanation should be in the `kernel_description.md` file.

General Constraints:
- Match all loop labels and array names exactly as used in the kernel code.
- Ensure the design space defined in the OptDSLv2 template produces a rich spread of latency vs. resource trade-offs.
- DO NOT add performance pragmas directly to the kernel code except for `#pragma HLS top name=...`.
- Do not omit any part. Output all six required code files.
"""
).strip()
##################################################################################

prompt_gen_optdsl_v2 = dedent(
    """
## Task Description
You are given an HLS design including its implementation file, header file, testbench and natural language description.

Your task is to generate an OptDSLv2 optimization template file named `opt_template.tcl` to enable design space exploration for the kernel implementation file only. 
The testbench file is not relevant for optimization and should not be considered.

This format replaces TCL directives with a structured Python-like DSL that expresses HLS directives such as `pipeline`, `unroll`, and `partition`.

### OptDSLv2 Semantics:
- This format uses a Python-like DSL to describe directive configurations for:
    - `pipeline(label: str, function: str, optional: bool = False)`
    - `unroll(label: str, function: str, factor: list[int], group: str | None = None)`
    - `partition(array_var: str, function: str, partition_type: str, factor: list[int], dim: int, group: str | None = None)`
        - DO NOT use dimension index 0 in any `partition()` directive. Vivado HLS indexing starts from 1, and `dim=0` is invalid. 
        - The `partition_type` is fixed as `cyclic` for all partitions.

### OptDSL Output Requirements:
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
    - The partition type is fixed as `cyclic` for all partitions.
    - A group must contain at least two directives. DO NOT create a group with only one directive—such grouping is unnecessary and should be avoided.
      - All directives in the same group MUST use the exact same factor list. Avoid mismatched factor list lengths or inconsistent values across grouped directives.
        For example, this is NOT allowed:
        ```
        partition("input", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
        partition("output", "kernel", "cyclic", [1, 2, 4, 8], 2, "group_1")
        unroll("loop_i", "kernel", [1, 2, 4, 8], "group_1")
        unroll("loop_j", "kernel", [1, 2, 4], "group_1")  # Mismatched factor list
        ```
        Instead, all of them must use the same full list, like [1, 2, 4, 8].
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
    DO NOT use symbolic constants, variables, or macros. All elements in the factor list must be explicit integers.
    For example:
    - For loops with bound 3 → recommend `[3]`
    - For loops with bound 64 → recommend `[1, 2, 4, 8]`
    - For larger loops → consider `[2, 4, 8, 16]` if suitable
    If a loop processes an array along a specific dimension, the unroll factor list for that loop and the partition factor list on that array dimension should match to avoid banking conflicts.


The generated `opt_template.tcl` file should:
- Begin with all `set_directive_resource` and `set_directive_inline` lines.
- Use grouped and ungrouped `partition`, `unroll`, and `pipeline` directives appropriately.
  - Apply `partition()` to major I/O and intermediate arrays.
  - Apply `unroll()` and `pipeline()` to key loops, avoiding duplication.
- Provide a diverse, structured design space while avoiding redundancy and overgrowth.

### Constraints
- DO NOT apply `pipeline` or `unroll` to the same loop in both grouped and ungrouped sections.
- DO NOT apply multiple directives of the same kind (e.g., two `pipeline()` calls with and without `optional=True`, or two `partition()` directives with different factors) to the same loop or array.
- Match all loop labels and array names exactly as used in the kernel code.
- Ensure the design space has a balanced spread of performance vs. resource trade-offs.
- Keep the total number of configurations but diverse enough for performance-resource tradeoff analysis.

Ensure the total design space includes a rich spread of latency vs. resource trade-offs.

Your final output must be a complete and syntactically correct OptDSLv2 optimization template file named `opt_template.tcl` that defines a diverse and practical optimization space for the design.
"""
).strip()

prompt_feedback_optdsl_v2 = dedent(
    f"""
After fixing the code, regenerate the updated OptDSLv2 optimization template file `opt_template.tcl` file that matches the corrected kernel structure and defines the proper design space.

{prompt_optdsl_v2_requirements}

Ensure the total design space includes a rich spread of latency vs. resource trade-offs.

The output must be a complete and syntactically correct OptDSLv2 optimization template file named `opt_template.tcl` that defines a diverse and practical optimization space for the design.
"""
).strip()


prompt_output_format_xml = dedent(
    text="""
## Output Format
The generated HLS output code should be provided in the following format:
```
<OUTPUT_CODE name="kernel_name.cpp">
    ...
</OUTPUT_CODE>
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.
Only output the generated HLS code in the XML format and nothing else.
"""
).strip()


prompt_output_format_gen_xml = dedent(
    text="""
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
    kernel_name
</OUTPUT_CODE>
<OUTPUT_CODE name="kernel_description.md">
    ...
</OUTPUT_CODE>
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.

You must output all five code blocks: the generated kernel code, the generated header file, the generated testbench code, the top-level function name in a file named `top.txt` (no extra symbols or text), and the natural language description (`kernel_description.md`).
In the example above `kernel_name` should be replaced with the generated name of the kernel.
Make sure the testbench filename ends with `_tb.cpp`.

Only output the generated HLS code in the XML format and nothing else.
"""
).strip()

prompt_output_format_gen_with_opt_xml = dedent(
    text="""
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

You must output all six code blocks: the generated kernel code, the generated header file, the generated testbench code, the top-level function name in a file named `top.txt` (no extra symbols or text), and the natural language description (`kernel_description.md`), the generated OptDSLv2 `opt_template.tcl` file.
In the example above `kernel_name` should be replaced with the generated name of the kernel.
Make sure the testbench filename ends with `_tb.cpp`.

Only output the generated HLS code in the XML format and nothing else.
"""
).strip()

prompt_output_format_optdsl = dedent(
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


def build_prompt_gen_zero_shot(
    design_description_fp: Path, design_tb: Path, design_h: Path
) -> str:
    p = prompt_pre
    p += "\n\n"
    p += prompt_gen
    p += "\n\n"
    p += prompt_output_format_xml
    p += "\n\n"

    p += "## Task Inputs\n"
    p += "\n"
    code = build_input_code_prompt_xml(
        {
            design_description_fp.name: design_description_fp.read_text(),
            design_tb.name: design_tb.read_text(),
            design_h.name: design_h.read_text(),
        }
    )
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p

##################################single input########################################
def build_prompt_gen_zero_shot_single_input_with_opt(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_opt: Path | None = None,
    design_pareto_score: Path | None = None,
) -> str:
    p = prompt_pre
    p += "\n\n"
    p += prompt_gen_single_input_with_opt
    p += "\n\n"
    p += prompt_output_format_gen_with_opt_xml
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

def build_prompt_gen_zero_shot_no_input_with_opt(

) -> str:
    p = prompt_pre
    p += "\n\n"
    p += prompt_gen_no_input_with_opt
    p += "\n\n"
    p += prompt_output_format_gen_with_opt_xml
    p += "\n\n"
    p += "## Task Output\n"
    p += "\n"

    return p

def build_prompt_gen_feed_back(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
    design_opt: Path | None = None,
    design_pareto_score: Path | None = None,
    error_message: str | None = None,
) -> str:
    p = prompt_pre
    p += "\n\n"
    if error_message:
        p += error_message
        p += "\n\n"
        p += prompt_feedback_optdsl_v2
    else:
        if design_pareto_score:
            p += prompt_gen_score_feedback_iter_with_opt
        else:
            p += prompt_gen_feedback_iter_with_opt
    p += "\n\n"
    p += prompt_output_format_gen_with_opt_xml
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

def build_prompt_gen_optdsl_zero_shot(
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
) -> str:
    p = prompt_pre
    p += "\n\n"
    p += prompt_gen_optdsl_v2
    p += "\n\n"
    p += prompt_output_format_optdsl
    p += "\n\n"

    p += "## Task Inputs\n"
    p += "\n"
    code = build_input_code_prompt_xml(
        {
            design_description_fp.name: design_description_fp.read_text(),
            design_h.name: design_h.read_text(),
            design_kernel.name: design_kernel.read_text(),
            design_tb.name: design_tb.read_text(),
        }
    )
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p

prompt_edit = dedent(
    """
## Task Description
Given a complete implementation of an HLS kernel in C++, a pre-written C++ design header, a pre-written C++ testbench, and a natural language description of the HLS design, generate the edited C++ code of the kernel (and possibly header and testbench) to perform the specific editing task outlined below.

When editing the provided HLS code, do not make new versions of the HLS kernels; edit the provided HLS kernels directly.
Edits should maintain consistency across the kernel, header, and testbench files. For example, changing function signatures and types in the kernel should be reflected in all places relevant in the code.

Make sure the resulting edited code is still correct syntactically and functionally so that it will pass compilation, testbench execution, and HLS synthesis.
"""
).strip()


prompt_loop_labels = dedent("""
### Editing Task - Loop Labels
Your task is to modify the given user's code to insert loop labels into the user's kernel (including in the kernel's top function, and any kernel subfunctions).
Only insert the loop labels; don't modify the actual loop code or insert any other pragmas.
Use the "labeled statement" C++ syntax as `label: statement` to label the loops.
If there are no loops in code, leave the code unchanged.
""").strip()

prompt_fpx = dedent(
    """
## Editing Task - Arbitrary Precision and Fixed-Point Types
Your task is to modify the given user's code to convert the usage of int and uint types to arbitrary precision HLS types, `ap_int`, `ap_uint`, as well as convert float and double types to fixed-point HLS types, `ap_fixed`, provided by Vitis HLS.

- int and uint types should be converted to the appropriate arbitrary precision types, `ap_int` and `ap_uint`.
- float and double types should be converted to the appropriate fixed-point types, `ap_fixed`.

The integer types are defined as follows:
    - `ap_int<W>`: Signed integer type with `W` bits
    - `ap_uint<W>`: Unsigned integer type with `W` bits
In order to use ap_(u)int types, the user needs to include the "ap_int.h" header file.
The individual bits in the ap_(u)int types can be indexed using the [] operator.
    You can also set and clear bits at specific indexes in the ap_(u)int types using the set and clear methods:
        - void ap_(u)int::set (unsigned i)
        - void ap_(u)int::clear (unsigned i)

The fixed point type is defined as follows:
    `ap_fixed<W, I>`
where:
    - `W`: Word length in bits
    - `I`: The number of bits used to represent the integer value, that is, the number of integer bits to the left of the binary point, including the sign bit.
In order to use fixed point types, the user needs to include the "ap_fixed.h" header file.
The fixed point type can handle most C++ arithmetic operations (addition, subtraction, multiplication, division, etc.) and can be used in most C++ expressions.

If the user is also doing `cmath` operations on the original datatype numbers, these operations should be modified to use the HLS math library.
The HLS math library has the namespace `hls::*` and can be included with the following "hls_math.h" file. It supports most of the same math operators under the std::* namespace.

Typedefs for these new types are encouraged to make the code more readable.
Ideally, `typedef` statements should be placed in a header file so they can be reached by all source files.

The resulting code should maintain the same functionality as the original code but should convert all variants of int, uint, float, and double types to the appropriate arbitrary precision types.
"""
).strip()


prompt_dataflow = dedent(
    """
## Editing Task - Dataflow Semantics
Your task is to modify the given user's code to use "dataflow" semantics in the HLS kernel using the `#pragma HLS DATAFLOW` pragma.
To use this pragma effectively, the code will need to be refactored into different subfunctions for computation tasks with intermediate producers and consumer variables.
Effectively, a dataflow function only contains calls to subfunctions as well as intermediate variables passed between the subfunctions.
The subfunctions must follow single-producer single-consumer rules, meaning that a variable / buffer can only be written to by one function and then only be read by one other function.
If data is needed for two subfunctions it needs to be duplicated.

The edited kernels must not have any of the following coding styles present in order to use dataflow semantics:
- Single-producer-consumer violations
- Feedback between tasks
- Conditional execution of tasks
- Loops with multiple exit conditions

In this case, if data needs to be buffered between tasks in a dataflow region, you may consider using fixed-sized arrays as buffers.

The resulting code should maintain the same functionality as the original code but should refactor the components into necessary dataflow subfunctions and use the DATAFLOW pragma in a manner which does not cause HLS synthesis to raise any dataflow violations.
"""
).strip()

prompt_loop_tiling = dedent(
    """
## Editing Task - Loop Tiling

Your task is to refactor the user's provided HLS kernel code by applying manual loop tiling source code transformations and inserting appropriate loop unrolling and array partitioning directives to optimize parallelism and memory access efficiency.

Specifically, you should:

- Manually tile loops into smaller blocks (tiles) with constant block sizes defined in the code to improve data locality and reduce memory bandwidth bottlenecks.

For a 2D example:

```
for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
        data[i][j] = ...;
        // loop body
    }
}
```

Should be transformed into:

```
const int N_TILE = 16;
const int M_TILE = 8;

#pragma HLS array_partition variable=data cyclic factor=N_TILE dim=1
#pragma HLS array_partition variable=data cyclic factor=M_TILE dim=2
<any more array partition pragmas needed for other variables>...

for(int i = 0; i < N; i += N_TILE) {
    for(int j = 0; j < M; j += M_TILE) {
        for(int ii = 0; ii < N_TILE; ii++) {
            #pragma HLS UNROLL
            for(int jj = 0; jj < M_TILE; jj++) {
                #pragma HLS UNROLL

                data[i + ii][j + jj] = ...;
                // loop body
            }
        }
    }
}
```

The same applies to 1D loops and n-D loops. Not all dimensions need to be tiled in every application.

- All loops must have fixed bounds and be perfect loops. Any other loop type is not allowed for tiling.
- The `#pragma HLS array_partition` pragma must be used if any arrays inside the loop are accessed using the tile indexes.
- Tiling with dependec on the outer loop is NOT allowed (ex. (ii = i; ii < i + N_TILE; ii++) is not allowed)
- Insert loop unrolling pragmas (`#pragma HLS UNROLL`) in the block loop of the loop tiling to minimize loop control overhead and maximize instruction-level parallelism.
- Be sure that the tiling factor is a constant value that is a factor of the loop trip count.
- If arrays are accessed using tile indexes, you must array partitioning pragmas (`#pragma HLS ARRAY_PARTITION <type> factor=... dim=...`) to allow parallel access to array elements inside unrolled loop tiles.
    - <type> can be `block` or `cyclic`, `factor` is the partitioning factor, and `dim` is the dimension of the array to partition (indexing starting at 1).
    - If multiple dims are partitioned, each dim of the array needs a separate pragma statement.
    - `factor` can be set to a const or define value in the code.
- Ensure the transformed loops maintain the same functionality and do not introduce loop-carried dependencies or initiation interval (II) violations.

If loop tiling and unrolling optimizations are not applicable to the provided kernel code, leave the code unchanged.
"""
)

prompt_tiles = dedent("""
## Task Description
Your task is to modify the given user's HLS C++ code to apply 'loop tiling' (also known as blocking) optimization where appropriate. 

You should:
- Identify loops with large iteration ranges or memory accesses where tiling would help improve performance.
- Introduce new inner and outer loops by splitting the iteration space into tiles. 

For example:
```cpp
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    ...
  }
}
```
can be transformed into:
```cpp
for (int tile_i = 0; tile_i < N; tile_i += TILE_SIZE_I) {
  for (int tile_j = 0; tile_j < M; tile_j += TILE_SIZE_J) {
    for (int i = 0; i < TILE_SIZE_I; i++) {
      for (int j = 0; j < TILE_SIZE_J; j++) {
        int ii = tile_i + i;
        int jj = tile_j + j;
        if (ii < N && jj < M) {
          ...
        }
      }
    }
  }
}
```
Always use fixed-size inner loops and compute the actual index using `int ii = tile_i + i;`, followed by `if (ii < N)` inside the loop body.
Do not use std::min(...), runtime-determined or conditional loop bounds directly in loop headers.

You must preserve the exact functionality of the original code; the output produced by the tiled kernel must match the original output bit-for-bit.
Define constants like TILE_SIZE at the top of the file or using #define.
You do not need to add any new pragmas or optimizations; just focus on the tiling.

Make sure to tile only the loops where it makes sense and where loop-carried dependencies allow it. 
Do not blindly apply tiling to all loops.

Only apply tiling to the kernel implementation file (i.e. kernel.cpp).
Do not modify any testbench files (i.e. files with names ending in _tb.cpp).
""").strip()

prompt_tiles_unroll = dedent("""
## Task Description
Your task is to modify the given user's HLS C++ code to apply 'loop tiling' (also known as blocking) optimization where appropriate. 

You should:
- Identify loops with large iteration ranges or memory accesses where tiling would help improve performance.
- Introduce new inner and outer loops by splitting the iteration space into tiles. 
- Apply loop unrolling pragmas (`#pragma HLS UNROLL`) only to appropriate inner block loops (i.e., the fixed-size inner loop introduced by tiling, which does not involve memory-dependent or control-heavy outer logic).

For example:
```cpp
for (int i = 0; i < N; i++) {
  ...
}
```
can be transformed into:
```cpp
for (int tile_i = 0; tile_i < N; tile_i += TILE_SIZE) {
  for (int i = 0; i < TILE_SIZE; i++) {
    #pragma HLS UNROLL
    int ii = tile_i + i;
    if (ii < N) {
      ...
    }
  }
}
```
Always use fixed-size inner loops and compute the actual index using `int ii = tile_i + i;`, followed by `if (ii < N)` inside the loop body.
Do not use std::min(...), runtime-determined or conditional loop bounds directly in loop headers.

You must preserve the exact functionality of the original code; the output produced by the tiled kernel must match the original output bit-for-bit.
Define constants like TILE_SIZE at the top of the file or using #define.

Make sure to tile only the loops where it makes sense and where loop-carried dependencies allow it. 
Do not blindly apply tiling to all loops.

Only apply tiling to the kernel implementation file (i.e. kernel.cpp).
Do not modify any testbench files (i.e. files with names ending in _tb.cpp).
""").strip()

prompt_streaming = dedent("""
## Editing Task - Streaming

Your task is to modify the given user's code to use `hls::stream<>` types for data movement in the HLS kernel.
Refactor to use streams where appropriate; this may require some functions to be refactored or the dataflow to be restructured.
                        
Streams can be included from the Vitis HLS "hls_stream.h" library.
                        
Given a stream specified as hls::stream<T>, the type T can be:
- Any C++ native data type
- A Vitis HLS arbitrary precision type (for example, ap_int<>, ap_ufixed<>)
- A user-defined struct containing either of the above types
                        
Streams can be un-named or named as follows:
```cpp
hls::stream<T> my_stream;
hls::stream<T> my_stream("stream_name");
```
                        
When streams are passed into and out of functions, they must be passed-by-reference as in the following example:
```cpp
void stream_function (
    hls::stream<uint8_t> &strm_out,
    hls::stream<uint8_t> &strm_in,
    uint16_t strm_len
)                 
```

Streams also have a blocking API to read and write data.
Read: `.read()` or `>>`
Write: `.write(data)` or `<<`

Read Example:
```cpp
hls::stream<int> my_stream;
int dst_var = my_stream.read();
// or
// int dst_var;
// my_stream >> dst_var;
```

Write Example:
```cpp
hls::stream<int> my_stream;
int src_var = 10;
my_stream.write(src_var);
// or
// my_stream << src_var;
```

Streams also have a non-blocking API:
- `.empty()`: Returns true if the stream is empty
- `.full()`: Returns true if the stream is full
- `.capacity()`: Returns the total capacity of the stream
- `.size()`: Returns the current number of elements in the stream

The default FIFO stream size/depth is 2.
You can change the depth of a stream by applying a STREAM pragma in the same scope the stream is declared.
```cpp
#pragma HLS stream variable=<stream_variable> depth=<int>
```
Be sure to set a stream depth appropriate for the deisgn / compute / algorithm requirements so that the design won't deadlock and optmize the deisgn for performance (i.e. considering stalls and latency). 
""").strip()

prompt_output_format_editing_xml = dedent(
    text="""
## Output Format
The generated HLS edited output code should be provided in the following format:
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
```
Please use this XML format and do not use other formats like markdown code blocks or plain text.

You must output all three code blocks: the edited kernel code, the edited header file, and the edited testbench code.
If one of the files is not edited, you still need output the code block with the original code.
In the example above `kernel_name` should be replaced with the original name of the kernel.
Make sure the testbench filename ends with `_tb.cpp`.

Only output the generated HLS code in the XML format and nothing else.
"""
).strip()


def build_prompt_edit_zero_shot(
    prompt_task: str,
    design_description_fp: Path,
    design_h: Path,
    design_kernel: Path,
    design_tb: Path,
) -> str:
    p = prompt_pre
    p += "\n\n"
    p += prompt_edit
    p += "\n\n"
    p += prompt_task
    p += "\n\n"
    p += prompt_output_format_editing_xml
    p += "\n\n"

    p += "## Task Inputs\n"
    p += "\n"
    code = build_input_code_prompt_xml(
        {
            design_description_fp.name: design_description_fp.read_text(),
            design_h.name: design_h.read_text(),
            design_kernel.name: design_kernel.read_text(),
            design_tb.name: design_tb.read_text(),
        }
    )
    p += code
    p += "\n\n"

    p += "## Task Output\n"
    p += "\n"

    return p
