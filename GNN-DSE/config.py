# config.py
from __future__ import annotations

import argparse
import ast as _ast
import torch

from utils import get_user, get_host

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def _maybe_literal_list(v):
    if isinstance(v, list):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return _ast.literal_eval(s)
            except Exception:
                return v
    return v

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="regression")
parser.add_argument("--subtask", default="train")
parser.add_argument("--plot_dse", type=_str2bool, default=False)

parser.add_argument("--dataset_zips_glob", default=None)
parser.add_argument("--graph_in_artifacts_subpath", default="artifacts/graph")

TARGETS = [
    "perf",
    "syn-BRAM", "syn-DSP", "syn-LUT", "syn-FF",
    "impl-BRAM", "impl-DSP", "impl-LUT", "impl-FF",
]
parser.add_argument("--targets", default=TARGETS)
parser.add_argument("--target", default=None)
parser.add_argument("--min_allowed_latency", type=float, default=100.0)
parser.add_argument("--epsilon", type=float, default=1e-12)
parser.add_argument("--normalizer", type=float, default=1e7)
parser.add_argument("--util_normalizer", type=float, default=1.0)
parser.add_argument("--max_number", type=float, default=1e10)

# Options: gnndse, logmse
parser.add_argument("--norm_method", default="gnndse")
parser.add_argument("--encode_log", type=_str2bool, default=False)
parser.add_argument("--invalid", type=_str2bool, default=False)

parser.add_argument("--graph_type", default="extended-pseudo-block")
parser.add_argument("--encode_edge_position", type=_str2bool, default=True)
parser.add_argument("--encoder_path", default=None)

parser.add_argument("--num_layers", type=int, default=6)
parser.add_argument("--D", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--jkn_mode", type=str, default="max")
parser.add_argument("--jkn_enable", type=_str2bool, default=True)

parser.add_argument("--node_attention", type=_str2bool, default=True)
parser.add_argument("--node_attention_MLP", type=_str2bool, default=False)

parser.add_argument("--separate_P", type=_str2bool, default=True) 
parser.add_argument("--separate_T", type=_str2bool, default=False)
parser.add_argument("--separate_pseudo", type=_str2bool, default=True)
parser.add_argument("--separate_icmp", type=_str2bool, default=False)
parser.add_argument("--P_use_all_nodes", type=_str2bool, default=True)

parser.add_argument("--gae_T", type=_str2bool, default=False)
parser.add_argument("--gae_P", type=_str2bool, default=False)
parser.add_argument("--input_encode", type=_str2bool, default=False)
parser.add_argument("--decoder_type", type=str, default="type2")

parser.add_argument("--pragma_as_MLP", type=_str2bool, default=True)
parser.add_argument("--pragma_as_MLP_list", default=["tile", "pipeline", "parallel"])
parser.add_argument("--pragma_scope", default="block")
parser.add_argument("--keep_pragma_attribute", type=_str2bool, default=False)
parser.add_argument("--pragma_order", default="parallel_and_merge")
parser.add_argument("--pragma_MLP_hidden_channels", default="[in_D // 2]")
parser.add_argument("--merge_MLP_hidden_channels", default="[in_D // 2]")
parser.add_argument("--gnn_layer_after_MLP", type=int, default=1)

parser.add_argument("--MLP_common_lyr", type=int, default=0)

parser.add_argument("--save_model", type=_str2bool, default=True)
parser.add_argument("--test_ratio", type=float, default=0.3)
parser.add_argument("--val_ratio", type=float, default=0)
parser.add_argument("--is_train_set", type=_str2bool, default=False)
parser.add_argument("--resample", type=_str2bool, default=False)
parser.add_argument("--activation", default="elu")
parser.add_argument("--activation_type", default="elu")

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--scheduler", default=None)
parser.add_argument("--warmup", default=None)
parser.add_argument("--random_seed", type=int, default=456)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--loss", type=str, default="MSE")
parser.add_argument("--epoch_num", type=int, default=300)

parser.add_argument("--feature_extract", type=_str2bool, default=False)
parser.add_argument("--fix_gnn_layer", type=int, default=0)
parser.add_argument("--random_MLP", type=_str2bool, default=False)
parser.add_argument("--plot_pred_points", type=_str2bool, default=False)
parser.add_argument("--val_debug", type=_str2bool, default=False)

parser.add_argument("--all_kernels", type=_str2bool, default=True)
parser.add_argument("--test_kernels", default=None)

parser.add_argument("--model_path", default=None)
parser.add_argument("--topk", type=int, default=50)

parser.add_argument("--dataset", default="polybench_mini-base")
parser.add_argument("--tag", default="gnndse")
parser.add_argument("--model_tag", default=None)

parser.add_argument("--force_regen", type=_str2bool, default=False)

gpu = 0
device = str(f"cuda:{gpu}" if torch.cuda.is_available() and gpu != -1 else "cpu")
parser.add_argument("--device", default=device)

parser.add_argument("--user", default=get_user())
parser.add_argument("--hostname", default=get_host())

FLAGS = parser.parse_args()

FLAGS.targets = _maybe_literal_list(FLAGS.targets)
FLAGS.pragma_as_MLP_list = _maybe_literal_list(FLAGS.pragma_as_MLP_list)

if FLAGS.model_tag is None:
    FLAGS.model_tag = FLAGS.tag

if FLAGS.target is not None:
    FLAGS.target = _maybe_literal_list(FLAGS.target)
else:
    FLAGS.target = FLAGS.targets

FLAGS.activation_type = FLAGS.activation
