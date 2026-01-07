# SyntheticHLS: Building Complex Application-Level Synthetic HLS Design Datasets using LLMs

## Setup
### HLSFactory Version
SyntheticHLS **requires an HLSFactory version that fully supports OptDSLv2**.  
Please ensure the installed HLSFactory includes:
- OptDSLv2 frontend
- **OptDSLv2 validation logic** (currently available in the `mzhou_OptDSLv2` branch of HLSFactory, not yet merged into the main branch)
### Environment Variables
SyntheticHLS needs environment variables for tool paths and model access.  
Please refer to **template.env** under example experiment folders for a complete example configuration.
### Vitis HLS Version
SyntheticHLS has been tested with **Vitis HLS 2023.1**. Other versions may work but are still under test.
### Experiments
Example experiment scripts are provided under experiments/, including single-target and multi-targets experiments. The analysis and visualization example scripts are included under the same directory.
Each experiment produces a self-contained workspace with intermediate artifacts and summary JSON file.

## About
This repository contains the work-in-progress code, dataset, and evaluation results for SyntheticHLS, a framework for generating diverse, synthesizable synthetic HLS datasets using Large Language Models (LLMs).

Key contributions:

- **Design Diversity Metrics**: Structural complexity, design space size, and dataflow diversity.
- **Parameterized Designs**: Generate HLS source + design space specs to expand into multiple implementations.
- **Iterative Feedback Loop**: Incrementally transform seed designs into complex, application-level datasets.


If you use this work, please cite:

```txt
@inproceedings{}
```
