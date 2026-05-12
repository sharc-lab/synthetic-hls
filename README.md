# SyntheticHLS: Building Complex Application-Level Synthetic HLS Design Datasets using LLMs

## Setup

For the GNN-DSE flow, `torch-scatter` is required but is not included in the TOML dependencies because it is a compiled PyTorch extension and its pip installation can fail due to wheel/CUDA/PyTorch compatibility issues. It is recommended to install it separately with conda:

```
conda install -c conda-forge torch-scatter
```

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


## GNN-DSE Evaluation

### 1. Generate Graphs

First run `/GNN-DSE/graph_gen/dataset_build.py` to extract designs from SyntheticHLS design-generation runs and organize them into the input format required by the graph-generation flow.

Then run `/GNN-DSE/graph_gen/graph_gen.py` to generate graph representations for the extracted HLS designs.

After graph generation, run `/GNN-DSE/graph_gen/data_agg.py` to aggregate the generated graphs and QoR metadata into dataset zip files. If only HLS synthesis data are needed, use `/GNN-DSE/graph_gen/data_agg_hls.py` instead. The resulting zip files are used as input to the dataset build stage.

### 2. Build Dataset

Build a PyTorch Geometric dataset from the zipped graph/data files:

```bash
python main.py \
  --subtask build_dataset \
  --dataset [dataset_name] \
  --dataset_zips_glob "[dataset_zip_path]" \
  --force_regen True
```

Use different `--dataset` names for different datasets. Include `--encoder_path` when building a new dataset with an existing encoder to keep feature dimensions compatible.

### 3. Train

Train a GNN model on a built dataset:

```bash
python main.py \
  --subtask train \
  --dataset [dataset_name]
```

You can set `--batch_size`, `--epochs`, `--test_ratio`, and other configs for training.

### 4. Inference

Run inference using a trained checkpoint:

```bash
python main.py \
  --subtask inference \
  --dataset [dataset_name] \
  --model_path "[train_model_path]"
```

For cross-dataset evaluation, set `--dataset` to the target dataset and `--model_path` to the checkpoint trained on the source dataset. Make sure the target dataset was built with compatible encoders.


If you use this work, please cite:

```txt
@inproceedings{}
```
