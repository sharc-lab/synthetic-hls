# data.py
from __future__ import annotations

import ast
import io
import tempfile
import random
import time
import gc
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from os.path import join, basename, exists
from zipfile import ZipFile

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, coo_matrix, csr_matrix
import os.path as osp

from config import FLAGS
from saver import saver
from utils import (
    get_save_path,
    create_dir_if_not_exists,
    print_stats,
    load,
    save,
)

SAVE_DIR = join(
    get_save_path(),
    FLAGS.dataset,
    f"optdslv2_{FLAGS.graph_type}_{FLAGS.task}_tag_{FLAGS.tag}",
)
ENCODER_PATH = join(SAVE_DIR, "encoders")
create_dir_if_not_exists(SAVE_DIR)

FPGA_TOTAL_RESOURCES = {
    "LUT": 274080,
    "FF": 548160,
    "DSP": 2520,
    "BRAM": 1824,
}


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m{rem:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m{rem:04.1f}s"


def _progress(msg: str):
    full = f"[BUILD PROGRESS] {msg}"
    try:
        saver.log_info(full)
    except Exception:
        print(full, flush=True)


def _choose_progress_every(total: int) -> int:
    if total <= 20:
        return 1
    if total <= 200:
        return 10
    if total <= 1000:
        return 25
    if total <= 5000:
        return 100
    return 250


class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None, data_files=None):
        super(MyOwnDataset, self).__init__(SAVE_DIR, transform, pre_transform)
        if data_files is not None:
            self.data_files = data_files

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if hasattr(self, "data_files"):
            return self.data_files
        return sorted(glob(join(SAVE_DIR, "*.pt")))

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def __len__(self):
        return self.len()

    def get_file_path(self, idx):
        if hasattr(self, "data_files"):
            return self.data_files[idx]
        return osp.join(SAVE_DIR, f"data_{idx}.pt")

    def get(self, idx):
        fn = self.get_file_path(idx)
        return _torch_load_pt(fn)

def _sanitize_loaded_data(data):
    def _ensure_tensor_1d_len(val, target_len=None, dtype=torch.float32):
        if not torch.is_tensor(val):
            val = torch.tensor(val, dtype=dtype)
        val = val.reshape(-1).to(dtype)
        if target_len is not None:
            cur = val.numel()
            if cur < target_len:
                pad = torch.zeros(target_len - cur, dtype=dtype)
                val = torch.cat([val, pad], dim=0)
            elif cur > target_len:
                val = val[:target_len]
        return val

    def _ensure_tensor_2d_width(val, width, dtype=torch.float32):
        if not torch.is_tensor(val):
            val = torch.tensor(val, dtype=dtype)
        if val.dim() == 1:
            val = val.reshape(-1, 1)
        val = val.to(dtype)
        cur_w = val.shape[1]
        if cur_w < width:
            pad = torch.zeros((val.shape[0], width - cur_w), dtype=dtype)
            val = torch.cat([val, pad], dim=1)
        elif cur_w > width:
            val = val[:, :width]
        return val

    if hasattr(data, "edge_index") and torch.is_tensor(data.edge_index):
        ei = data.edge_index
        if ei.dim() != 2:
            ei = ei.reshape(2, -1) if ei.numel() % 2 == 0 else ei
        if ei.dim() == 2 and ei.shape[0] > 2:
            ei = ei[:2, :]
        data.edge_index = ei.long().contiguous()

    num_nodes = None
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() >= 1:
        num_nodes = data.x.shape[0]

    one_d_node_masks = [
        "X_contextnids",
        "X_pragmanids",
        "X_pragmascopenids",
        "X_pseudonids",
        "X_icmpnids",
    ]
    for key in one_d_node_masks:
        if hasattr(data, key):
            val = getattr(data, key)
            setattr(data, key, _ensure_tensor_1d_len(val, target_len=num_nodes, dtype=torch.float32))

    if hasattr(data, "X_pragma_per_node"):
        val = getattr(data, "X_pragma_per_node")
        val = _ensure_tensor_2d_width(val, width=4, dtype=torch.float32)
        if num_nodes is not None:
            if val.shape[0] < num_nodes:
                pad = torch.zeros((num_nodes - val.shape[0], val.shape[1]), dtype=torch.float32)
                val = torch.cat([val, pad], dim=0)
            elif val.shape[0] > num_nodes:
                val = val[:num_nodes, :]
        data.X_pragma_per_node = val

    if hasattr(data, "pragmas"):
        val = getattr(data, "pragmas")
        if not torch.is_tensor(val):
            val = torch.tensor(val, dtype=torch.float32)
        if val.dim() == 0:
            val = val.reshape(1, 1)
        elif val.dim() == 1:
            val = val.reshape(1, -1)
        if val.shape[1] < 1:
            pad = torch.zeros((val.shape[0], 1 - val.shape[1]), dtype=torch.float32)
            val = torch.cat([val, pad], dim=1)
        elif val.shape[1] > 1:
            val = val[:, :1]
        if val.shape[0] > 1:
            val = val[:1, :]
        data.pragmas = val.float()

    scalar_targets = [
        "perf", "actual_perf", "kernel_speedup", "quality",
        "syn_BRAM", "syn_DSP", "syn_LUT", "syn_FF",
        "impl_BRAM", "impl_DSP", "impl_LUT", "impl_FF",
        "total_BRAM", "total_DSP", "total_LUT", "total_FF",
    ]
    for key in scalar_targets:
        if hasattr(data, key):
            val = getattr(data, key)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32)
            val = val.reshape(-1).float()
            if val.numel() == 0:
                val = torch.zeros(1, dtype=torch.float32)
            elif val.numel() > 1:
                val = val[:1]
            setattr(data, key, val)

    return data

def _torch_load_pt(path: str):
    try:
        data = torch.load(path, weights_only=False)
    except TypeError:
        data = torch.load(path)
    return _sanitize_loaded_data(data)

def split_dataset(dataset, train, val, dataset_test=None):
    file_li = dataset.processed_file_names
    li = random_split(
        file_li,
        [train, val, len(dataset) - train - val],
        generator=torch.Generator().manual_seed(FLAGS.random_seed),
    )
    if dataset_test is None:
        dataset_test = li[2]
    saver.log_info(
        f"{len(file_li)} graphs in total: {len(li[0])} train {len(li[1])} val {len(dataset_test)} test"
    )
    return [MyOwnDataset(data_files=li[0]), MyOwnDataset(data_files=li[1]), MyOwnDataset(data_files=dataset_test)]


def split_dataset_resample(dataset, train, val, test, test_id=0):
    file_li = dataset.processed_file_names
    num_batch = int(1 / test)
    splits_ratio = [int(len(dataset) * test)] * num_batch
    splits_ratio[-1] = len(dataset) - int(len(dataset) * test * (num_batch - 1))
    splits_ = random_split(file_li, splits_ratio, generator=torch.Generator().manual_seed(100))
    test_split = splits_[test_id]
    train_val_data = []
    for i in range(num_batch):
        if i != test_id:
            train_val_data.extend(splits_[i])

    new_train = int(len(train_val_data) * train / (train + val))
    new_val = len(train_val_data) - new_train
    li = random_split(train_val_data, [new_train, new_val], generator=torch.Generator().manual_seed(100))

    saver.log_info(
        f"{len(file_li)} graphs in total: {len(li[0])} train {len(li[1])} val {len(test_split)} test"
    )
    return MyOwnDataset(data_files=li[0]), MyOwnDataset(data_files=li[1]), MyOwnDataset(data_files=test_split)


def _parse_kernel_list(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    s = str(val).strip()
    if not s:
        return None
    # list literal
    if s.startswith('[') and s.endswith(']'):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, (list, tuple)):
                return [str(x) for x in out]
        except Exception:
            pass
    # comma-separated
    return [x.strip() for x in s.split(',') if x.strip()]


def get_kernel_samples(dataset: 'MyOwnDataset'):
    """
    Return a dataset filtered to a subset of kernels.
    """
    wanted = _parse_kernel_list(getattr(FLAGS, 'test_kernels', None))

    file_li = dataset.processed_file_names
    if not file_li:
        return MyOwnDataset(data_files=[])

    kernel_to_files = defaultdict(list)
    for fp in file_li:
        try:
            d = _torch_load_pt(fp)
            k = str(getattr(d, 'kernel', getattr(d, 'gname', 'unknown')))
        except Exception:
            k = 'unknown'
        kernel_to_files[k].append(fp)

    if wanted:
        selected = []
        for k in wanted:
            selected.extend(kernel_to_files.get(k, []))
        saver.log_info(f"[get_kernel_samples] selected kernels={wanted}, graphs={len(selected)}")
        return MyOwnDataset(data_files=selected)

    first_kernel = sorted(kernel_to_files.keys())[0]
    selected = kernel_to_files[first_kernel]
    saver.log_info(f"[get_kernel_samples] FLAGS.test_kernels not set; using first kernel='{first_kernel}' graphs={len(selected)}")
    return MyOwnDataset(data_files=selected)


def split_train_test_kernel(dataset: 'MyOwnDataset'):
    """Split dataset into train/test by kernel names.

    Used when FLAGS.test_kernels is not None.
    Returns dict: {'train': MyOwnDataset, 'test': MyOwnDataset}
    """
    test_kernels = _parse_kernel_list(getattr(FLAGS, 'test_kernels', None))
    if not test_kernels:
        # nothing to split on
        return {'train': dataset, 'test': None}

    file_li = dataset.processed_file_names
    train_files, test_files = [], []

    for fp in file_li:
        try:
            d = _torch_load_pt(fp)
            k = str(getattr(d, 'kernel', getattr(d, 'gname', 'unknown')))
        except Exception:
            k = 'unknown'
        if k in test_kernels:
            test_files.append(fp)
        else:
            train_files.append(fp)

    saver.log_info(
        f"[split_train_test_kernel] kernels(test)={test_kernels} -> train_graphs={len(train_files)} test_graphs={len(test_files)}"
    )
    return {'train': MyOwnDataset(data_files=train_files), 'test': MyOwnDataset(data_files=test_files)}


def _coo_to_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))


def transform_X_torch(X):
    X = torch.FloatTensor(np.array(X))
    X = coo_matrix(X)
    return _coo_to_sparse(X).to_dense()


def create_edge_index(g):
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    edges = list(g.edges)

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    first = edges[0]
    if len(first) == 2:
        edge_pairs = edges
    elif len(first) >= 3:
        edge_pairs = [(u, v) for u, v, *_ in edges]
    else:
        raise RuntimeError(f"Unexpected edge tuple format in graph edges: {first}")

    edge_index = torch.LongTensor(edge_pairs).t().contiguous()
    return edge_index


def _encode_edge_dict(g, ftypes=None, ptypes=None):
    X_ftype, X_ptype = [], []
    for _, _, edata in g.edges(data=True):
        X_ftype.append([edata.get("flow", 0)])
        X_ptype.append([edata.get("position", 0)])
        if ftypes is not None:
            ftypes[edata.get("flow", 0)] += 1
        if ptypes is not None:
            ptypes[edata.get("position", 0)] += 1
    return {"X_ftype": X_ftype, "X_ptype": X_ptype}


def _encode_edge_torch(edge_dict, enc_ftype, enc_ptype):
    X_ftype = enc_ftype.transform(edge_dict["X_ftype"])
    X_ptype = enc_ptype.transform(edge_dict["X_ptype"])
    X = hstack((X_ftype, X_ptype)) if getattr(FLAGS, "encode_edge_position", True) else coo_matrix(X_ftype)
    if isinstance(X, csr_matrix):
        X = X.tocoo()
    return _coo_to_sparse(X).to_dense()


def _encode_X_dict(g, ntypes=None, ptypes=None, itypes=None, ftypes=None, btypes=None):
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")

    X_ntype, X_ptype, X_numeric = [], [], []
    X_itype, X_ftype, X_btype = [], [], []
    X_contextnids, X_pragmanids, X_pseudonids, X_icmpnids = [], [], [], []
    X_pragma_per_node, X_pragmascopenids = [], []

    num_nodes = g.number_of_nodes()
    for nid in range(num_nodes):
        node = nid
        ndata = g.nodes[node]

        ntype = int(ndata.get("type", 0))
        text = str(ndata.get("text", ""))
        block = int(ndata.get("block", 0))
        func = int(ndata.get("function", 0))
        full_text = str(ndata.get("full_text", ""))

        if ntypes is not None:
            ntypes[ntype] += 1
        if itypes is not None:
            itypes[text] += 1
        if btypes is not None:
            btypes[block] += 1
        if ftypes is not None:
            ftypes[func] += 1

        pragma_vector = [0, 0, 0, 0]
        if "pseudo" in text:
            X_pseudonids.append(1)
            neighbor_pragmas = {}

            for nb in g.neighbors(node):
                tnb = str(g.nodes[nb].get("text", "")).lower()
                if tnb in ("pipeline", "parallel", "tile"):
                    neighbor_pragmas[tnb] = nb
            if len(neighbor_pragmas) == 0:
                X_pragmascopenids.append(0)
            else:
                X_pragmascopenids.append(1)
                if "tile" in neighbor_pragmas:
                    pragma_vector[0] = 1
                if "pipeline" in neighbor_pragmas:
                    pragma_vector[1] = 10
                if "parallel" in neighbor_pragmas:
                    ptxt = str(g.nodes[neighbor_pragmas["parallel"]].get("full_text", "")).upper()
                    if "FACTOR=" in ptxt:
                        try:
                            fval = int(ptxt.split("FACTOR=")[-1].split()[0])
                        except Exception:
                            fval = 0
                    else:
                        fval = 0
                    pragma_vector[2] = 1
                    pragma_vector[3] = fval
        else:
            X_pseudonids.append(0)
            X_pragmascopenids.append(0)

        X_pragma_per_node.append(pragma_vector)

        numeric = 0
        if "icmp" in full_text:
            tail = full_text.split(",")[-1].strip()
            if tail.isdigit():
                numeric = int(tail)
                X_icmpnids.append(1)
            else:
                X_icmpnids.append(0)
        else:
            X_icmpnids.append(0)

        if "pragma" in full_text.lower() or text.lower() in ("pipeline", "parallel", "tile"):
            ptype = full_text.upper() if full_text else text.upper()
            X_pragmanids.append(1)
            X_contextnids.append(0)
        else:
            ptype = "None"
            X_pragmanids.append(0)
            X_contextnids.append(0 if "pseudo" in text else 1)

        if ptypes is not None:
            ptypes[ptype] += 1

        X_ntype.append([ntype])
        X_ptype.append([ptype])
        X_numeric.append([numeric])
        X_itype.append([text])
        X_ftype.append([func])
        X_btype.append([block])

    X_pragma_per_node = transform_X_torch(X_pragma_per_node)

    return {
        "X_ntype": X_ntype,
        "X_ptype": X_ptype,
        "X_numeric": X_numeric,
        "X_itype": X_itype,
        "X_ftype": X_ftype,
        "X_btype": X_btype,
        "X_contextnids": torch.FloatTensor(np.array(X_contextnids)),
        "X_pragmanids": torch.FloatTensor(np.array(X_pragmanids)),
        "X_pragmascopenids": torch.FloatTensor(np.array(X_pragmascopenids)),
        "X_pseudonids": torch.FloatTensor(np.array(X_pseudonids)),
        "X_icmpnids": torch.FloatTensor(np.array(X_icmpnids)),
        "X_pragma_per_node": X_pragma_per_node,
    }


def _encode_X_torch(x_dict, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype):
    X_ntype = enc_ntype.transform(x_dict["X_ntype"])
    X_ptype = enc_ptype.transform(x_dict["X_ptype"])
    X_itype = enc_itype.transform(x_dict["X_itype"])
    X_ftype = enc_ftype.transform(x_dict["X_ftype"])
    X_btype = enc_btype.transform(x_dict["X_btype"])
    X_numeric = x_dict["X_numeric"]
    X = hstack((X_ntype, X_ptype, X_numeric, X_itype, X_ftype, X_btype))
    return _coo_to_sparse(X.tocoo()).to_dense()


def _list_zip_files(glob_expr) -> list[str]:
    import glob as _glob

    if glob_expr is None:
        raise ValueError("dataset_zips_glob is None")

    if isinstance(glob_expr, (list, tuple)):
        exprs = [str(x).strip() for x in glob_expr if str(x).strip()]
    else:
        s = str(glob_expr).strip()
        if not s:
            raise ValueError("dataset_zips_glob is empty")
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    exprs = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    exprs = [s]
            except Exception:
                exprs = [s]
        elif "," in s:
            exprs = [x.strip() for x in s.split(",") if x.strip()]
        else:
            exprs = [s]

    files = []
    for expr in exprs:
        matched = sorted(_glob.glob(expr, recursive=True))
        if matched:
            files.extend(matched)
            continue

        if Path(expr).is_file():
            files.append(expr)
            continue

        raise FileNotFoundError(f"No zip files matched: {expr}")

    files = sorted(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No zip files matched: {glob_expr}")
    return files


def _get_zip_design_counts(zip_files: list[str]) -> list[int]:
    counts = []
    for zfp in zip_files:
        with ZipFile(zfp, "r") as z:
            csv_name = _pick_data_all_csv_name(z)
            df = pd.read_csv(io.StringIO(_read_zip_text(z, csv_name)), usecols=["design_id"])
            counts.append(len(df))
    return counts


def _compute_balanced_per_zip_quotas(available_counts: list[int], dataset_size: int | None) -> list[int | None]:
    if dataset_size is None:
        return [None] * len(available_counts)

    if dataset_size < 0:
        raise ValueError(f"dataset_size must be >= 0, got {dataset_size}")

    if len(available_counts) == 0:
        return []

    total_available = sum(available_counts)
    if total_available <= 0:
        return [0] * len(available_counts)

    target_total = min(dataset_size, total_available)

    raw_quotas = [
        (target_total * cnt) / total_available
        for cnt in available_counts
    ]

    quotas = [int(np.floor(q)) for q in raw_quotas]
    assigned = sum(quotas)
    leftover = target_total - assigned

    # distribute remaining samples by largest fractional part
    frac_order = sorted(
        range(len(available_counts)),
        key=lambda i: (raw_quotas[i] - quotas[i], available_counts[i]),
        reverse=True,
    )

    for i in frac_order:
        if leftover <= 0:
            break
        if quotas[i] < available_counts[i]:
            quotas[i] += 1
            leftover -= 1

    # if any leftover remains because some zips hit capacity, redistribute
    while leftover > 0:
        progressed = False
        for i in range(len(available_counts)):
            if quotas[i] < available_counts[i]:
                quotas[i] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break

    return quotas


def _read_zip_text(z: ZipFile, name: str) -> str:
    with z.open(name, "r") as f:
        return f.read().decode("utf-8", errors="ignore")


def _pick_data_all_csv_name(z: ZipFile) -> str:
    names = set(z.namelist())
    if "data_all.csv" in names:
        return "data_all.csv"
    candidates = [n for n in names if n.endswith("data_all.csv")]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError("Could not find data_all.csv in zip.")


def _extract_single_gexf_from_artifacts_zip(outer_zip: ZipFile, design_id: str) -> str:
    art_path = f"{design_id}/artifacts.zip"
    if art_path not in outer_zip.namelist():
        raise FileNotFoundError(f"Missing {art_path} in dataset zip")

    art_bytes = outer_zip.read(art_path)
    with ZipFile(io.BytesIO(art_bytes), "r") as az:
        prefix = getattr(FLAGS, "graph_in_artifacts_subpath", "artifacts/graph").rstrip("/") + "/"
        gexf_candidates = [n for n in az.namelist() if n.startswith(prefix) and n.endswith(".gexf")]
        if len(gexf_candidates) != 1:
            raise RuntimeError(f"Expected exactly 1 .gexf under {prefix}, got {gexf_candidates}")
        gexf_name = gexf_candidates[0]
        tmpdir = tempfile.mkdtemp(prefix="harp_graph_")
        out_fp = join(tmpdir, basename(gexf_name))
        with az.open(gexf_name, "r") as gf, open(out_fp, "wb") as out:
            out.write(gf.read())
        return out_fp


def _get_num(row: dict, key: str, default: float = 0.0) -> float:
    if key not in row:
        return default
    v = row.get(key, default)
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        try:
            return float(s)
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default


def _pick_first(row: dict, keys: list[str], default: float = 0.0) -> float:
    for k in keys:
        v = _get_num(row, k, default=None)
        if v is not None:
            return float(v)
    return float(default)


def _map_row_to_targets(row: dict) -> dict:
    """
    Produce both synthesis QoR targets and implementation QoR targets.
    """
    def _apply_target_normalization(out: dict) -> dict:
        """
        Normalize regression targets according to FLAGS.norm_method.
        Supported:
        - norm_method="off": no normalization
        - norm_method="gnndse": match the GNN-DSE paper:
            * resources: used / available_resources
            * latency: T_latency = log2(NormalizationFactor / latency)
        """
        norm_method = getattr(FLAGS, "norm_method", "off") or "off"
        out2 = dict(out)

        # No normalization
        if norm_method in ("off", "logmse"):
            return out2

        # GNN-DSE paper normalization
        if norm_method == "gnndse":
            eps = float(getattr(FLAGS, "epsilon", 1e-3))
            norm_factor = float(getattr(FLAGS, "normalizer", 1e7))

            # latency objective (stored in `perf`): log2(norm_factor / latency)
            latency = float(out2.get("perf", 0.0))
            latency = max(latency, eps)
            out2["perf"] = float(np.log2(norm_factor / latency))
            out2["actual_perf"] = out2["perf"]

            lut_tot = float(FPGA_TOTAL_RESOURCES.get("LUT", 1.0))
            ff_tot = float(FPGA_TOTAL_RESOURCES.get("FF", 1.0))
            dsp_tot = float(FPGA_TOTAL_RESOURCES.get("DSP", 1.0))
            bram_tot = float(FPGA_TOTAL_RESOURCES.get("BRAM", 1.0))

            def _safe_div(x, d):
                d = d if d > 0 else 1.0
                return float(x) / float(d)

            out2["syn-LUT"] = _safe_div(out2.get("syn-LUT", 0.0), lut_tot)
            out2["syn-FF"]  = _safe_div(out2.get("syn-FF", 0.0), ff_tot)
            out2["syn-DSP"] = _safe_div(out2.get("syn-DSP", 0.0), dsp_tot)
            out2["syn-BRAM"] = _safe_div(out2.get("syn-BRAM", 0.0), bram_tot)

            out2["impl-LUT"] = _safe_div(out2.get("impl-LUT", 0.0), lut_tot)
            out2["impl-FF"]  = _safe_div(out2.get("impl-FF", 0.0), ff_tot)
            out2["impl-DSP"] = _safe_div(out2.get("impl-DSP", 0.0), dsp_tot)
            out2["impl-BRAM"] = _safe_div(out2.get("impl-BRAM", 0.0), bram_tot)

            return out2

        raise ValueError(f"Unknown norm_method: {norm_method}")

    out = {}
    perf = _pick_first(
        row,
        ["synthesis__latency_average_cycles", "synthesis__latency_worst_cycles", "synthesis__latency_best_cycles"],
        default=0.0,
    )
    out["perf"] = perf
    out["actual_perf"] = perf

    lut_syn = _get_num(row, "synthesis__resources_lut_used", 0.0)
    ff_syn  = _get_num(row, "synthesis__resources_ff_used", 0.0)
    dsp_syn = _get_num(row, "synthesis__resources_dsp_used", 0.0)
    bram_syn = _get_num(row, "synthesis__resources_bram_used", 0.0)

    lut_impl = _get_num(row, "implementation__utilization__Total LUTs", 0.0)
    ff_impl  = _get_num(row, "implementation__utilization__FFs", 0.0)
    dsp_impl = _get_num(row, "implementation__utilization__DSP Blocks", 0.0)

    ramb36_impl = _get_num(row, "implementation__utilization__RAMB36", 0.0)
    ramb18_impl = _get_num(row, "implementation__utilization__RAMB18", 0.0)
    bram_impl = 2.0 * ramb36_impl + 1.0 * ramb18_impl

    out["syn-LUT"] = float(lut_syn)
    out["syn-FF"] = float(ff_syn)
    out["syn-DSP"] = float(dsp_syn)
    out["syn-BRAM"] = float(bram_syn)

    out["impl-LUT"] = float(lut_impl)
    out["impl-FF"] = float(ff_impl)
    out["impl-DSP"] = float(dsp_impl)
    out["impl-BRAM"] = float(bram_impl)

    out["total-LUT"] = FPGA_TOTAL_RESOURCES["LUT"]
    out["total-FF"] = FPGA_TOTAL_RESOURCES["FF"]
    out["total-DSP"] = FPGA_TOTAL_RESOURCES["DSP"]
    out["total-BRAM"] = FPGA_TOTAL_RESOURCES["BRAM"]

    out["quality"] = 0.0
    out["kernel_speedup"] = 0.0

    out = _apply_target_normalization(out)
    return out


def _sample_df_rows_balanced(df: pd.DataFrame, quota: int | None, rng: random.Random) -> pd.DataFrame:
    if quota is None:
        return df

    if quota <= 0:
        return df.iloc[[]].copy()

    n = len(df)
    if quota >= n:
        return df.sample(frac=1.0, random_state=rng.randint(0, 10**9)).reset_index(drop=True)

    sampled_idx = rng.sample(list(range(n)), quota)
    return df.iloc[sampled_idx].reset_index(drop=True)



def _prepare_sampled_zip_rows(zip_files: list[str], quotas: list[int | None], rng: random.Random):
    sampled = []
    total_rows = 0
    t0 = time.time()

    for zip_idx, zfp in enumerate(zip_files):
        zip_t0 = time.time()
        saver.info(f"[ZIP] {zfp}")
        _progress(f"[SAMPLE {zip_idx + 1}/{len(zip_files)}] reading csv from {basename(zfp)}")
        with ZipFile(zfp, "r") as z:
            csv_name = _pick_data_all_csv_name(z)
            df = pd.read_csv(io.StringIO(_read_zip_text(z, csv_name)))

        if "design_id" not in df.columns:
            raise RuntimeError(f"{csv_name} must contain design_id. got columns: {list(df.columns)}")

        raw_count = len(df)
        quota = quotas[zip_idx]
        df = _sample_df_rows_balanced(df, quota, rng)
        rows = df.to_dict("records")
        total_rows += len(rows)

        _progress(
            f"[SAMPLE {zip_idx + 1}/{len(zip_files)}] raw={raw_count} sampled={len(rows)} "
            f"quota={quota} zip_elapsed={_format_seconds(time.time() - zip_t0)} "
        )
        sampled.append((zfp, rows))

    _progress(
        f"[SAMPLE DONE] sampled_total={total_rows} across {len(zip_files)} zip(s) "
        f"elapsed={_format_seconds(time.time() - t0)}"
    )
    return sampled


def _create_empty_encoders():
    return {
        "enc_ntype": OneHotEncoder(handle_unknown="ignore"),
        "enc_ptype": OneHotEncoder(handle_unknown="ignore"),
        "enc_itype": OneHotEncoder(handle_unknown="ignore"),
        "enc_ftype": OneHotEncoder(handle_unknown="ignore"),
        "enc_btype": OneHotEncoder(handle_unknown="ignore"),
        "enc_ftype_edge": OneHotEncoder(handle_unknown="ignore"),
        "enc_ptype_edge": OneHotEncoder(handle_unknown="ignore"),
    }


def _fit_encoders_from_sampled_rows(sampled_zip_rows):
    encoders = _create_empty_encoders()

    X_ntype_all, X_ptype_all, X_itype_all, X_ftype_all, X_btype_all = [], [], [], [], []
    edge_ftype_all, edge_ptype_all = [], []
    skipped = 0

    ntypes = Counter()
    ptypes = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    total_rows = sum(len(rows) for _, rows in sampled_zip_rows)
    progress_every = _choose_progress_every(total_rows)
    fit_t0 = time.time()
    seen = 0

    _progress(
        f"[ENCODER FIT] start total_designs={total_rows}"
    )

    for zip_idx, (zfp, rows) in enumerate(sampled_zip_rows):
        zip_t0 = time.time()
        # _progress(
        #     f"[ENCODER FIT][ZIP {zip_idx + 1}/{len(sampled_zip_rows)}] start "
        #     f"name={basename(zfp)} rows={len(rows)}"
        # )
        with ZipFile(zfp, "r") as z:
            for row in rows:
                design_id = str(row["design_id"])
                try:
                    gexf_fp = _extract_single_gexf_from_artifacts_zip(z, design_id)
                    g = nx.read_gexf(gexf_fp)
                except Exception as e:
                    saver.warning(f"skip design_id={design_id} during encoder fit due to missing graph: {e}")
                    skipped += 1
                    seen += 1
                    # if seen % progress_every == 0 or seen == total_rows:
                    #     _progress(
                    #         f"[ENCODER FIT] done={seen}/{total_rows} skipped={skipped} "
                    #         f"elapsed={_format_seconds(time.time() - fit_t0)} "
                    #         f"maxrss={_get_maxrss_mb():.1f} MB"
                    #     )
                    continue

                d_node = _encode_X_dict(g, ntypes, ptypes, itypes, ftypes, btypes)
                d_edge = _encode_edge_dict(g, ftypes_edge, ptypes_edge)

                X_ntype_all += d_node["X_ntype"]
                X_ptype_all += d_node["X_ptype"]
                X_itype_all += d_node["X_itype"]
                X_ftype_all += d_node["X_ftype"]
                X_btype_all += d_node["X_btype"]
                edge_ftype_all += d_edge["X_ftype"]
                edge_ptype_all += d_edge["X_ptype"]

                seen += 1
                # if seen % progress_every == 0 or seen == total_rows:
                #     _progress(
                #         f"[ENCODER FIT] done={seen}/{total_rows} skipped={skipped} "
                #         f"last_design={design_id} nodes={g.number_of_nodes()} edges={g.number_of_edges()} "
                #         f"elapsed={_format_seconds(time.time() - fit_t0)} "
                #         f"maxrss={_get_maxrss_mb():.1f} MB"
                #     )

                del g, d_node, d_edge
                if seen % progress_every == 0:
                    gc.collect()

        _progress(
            f"[ENCODER FIT][ZIP {zip_idx + 1}/{len(sampled_zip_rows)}] done "
            f"name={basename(zfp)} zip_elapsed={_format_seconds(time.time() - zip_t0)} "
        )

    _progress("[ENCODER FIT] fitting sklearn encoders now")
    encoders["enc_ptype"].fit(X_ptype_all)
    encoders["enc_ntype"].fit(X_ntype_all)
    encoders["enc_itype"].fit(X_itype_all)
    encoders["enc_ftype"].fit(X_ftype_all)
    encoders["enc_btype"].fit(X_btype_all)
    encoders["enc_ftype_edge"].fit(edge_ftype_all)
    encoders["enc_ptype_edge"].fit(edge_ptype_all)
    saver.log_info(f"Encoder fitting done. skipped_graphs_during_fit={skipped}")
    _progress(
        f"[ENCODER FIT DONE] skipped={skipped} elapsed={_format_seconds(time.time() - fit_t0)} "
    )
    return encoders


def _build_pyg_data_from_graph(g, row, d_node, d_edge, encoders):
    label = _map_row_to_targets(row)
    design_id = str(row["design_id"])

    X = _encode_X_torch(
        d_node,
        encoders["enc_ntype"],
        encoders["enc_ptype"],
        encoders["enc_itype"],
        encoders["enc_ftype"],
        encoders["enc_btype"],
    )
    edge_attr = _encode_edge_torch(
        d_edge,
        encoders["enc_ftype_edge"],
        encoders["enc_ptype_edge"],
    )
    edge_index = create_edge_index(g)

    gname = row.get("design__name", design_id)
    kernel = row.get("design__name", gname)
    pragmas = torch.zeros((1, 1), dtype=torch.float32)

    data = Data(
        gname=str(gname),
        kernel=str(kernel),
        key=str(design_id),
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        X_contextnids=d_node["X_contextnids"],
        X_pragmanids=d_node["X_pragmanids"],
        X_pragmascopenids=d_node["X_pragmascopenids"],
        X_pseudonids=d_node["X_pseudonids"],
        X_icmpnids=d_node["X_icmpnids"],
        X_pragma_per_node=d_node["X_pragma_per_node"],
        pragmas=pragmas,

        perf=torch.FloatTensor(np.array([label["perf"]])),
        actual_perf=torch.FloatTensor(np.array([label["actual_perf"]])),
        kernel_speedup=torch.FloatTensor(np.array([label["kernel_speedup"]])),
        quality=torch.FloatTensor(np.array([label["quality"]])),

        syn_BRAM=torch.FloatTensor(np.array([label["syn-BRAM"]])),
        syn_DSP=torch.FloatTensor(np.array([label["syn-DSP"]])),
        syn_LUT=torch.FloatTensor(np.array([label["syn-LUT"]])),
        syn_FF=torch.FloatTensor(np.array([label["syn-FF"]])),

        impl_BRAM=torch.FloatTensor(np.array([label["impl-BRAM"]])),
        impl_DSP=torch.FloatTensor(np.array([label["impl-DSP"]])),
        impl_LUT=torch.FloatTensor(np.array([label["impl-LUT"]])),
        impl_FF=torch.FloatTensor(np.array([label["impl-FF"]])),

        total_BRAM=torch.FloatTensor(np.array([label["total-BRAM"]])),
        total_DSP=torch.FloatTensor(np.array([label["total-DSP"]])),
        total_LUT=torch.FloatTensor(np.array([label["total-LUT"]])),
        total_FF=torch.FloatTensor(np.array([label["total-FF"]])),
    )
    return data, design_id


def get_data_list_from_hlsfactory_zips():
    if getattr(FLAGS, "dataset_zips_glob", None) is None:
        raise ValueError("FLAGS.dataset_zips_glob must be set (glob of HLSFactory data_packaging zips).")

    build_t0 = time.time()
    zip_files = _list_zip_files(FLAGS.dataset_zips_glob)
    saver.log_info(f"Found {len(zip_files)} dataset zip(s)")
    _progress(
        f"[START] dataset={FLAGS.dataset} zips={len(zip_files)} dataset_size={getattr(FLAGS, 'dataset_size', None)} "
        f"force_regen={getattr(FLAGS, 'force_regen', False)} encoder_path={getattr(FLAGS, 'encoder_path', None)} "
        f"save_dir={SAVE_DIR}"
    )

    available_counts = _get_zip_design_counts(zip_files)
    quotas = _compute_balanced_per_zip_quotas(available_counts, getattr(FLAGS, "dataset_size", None))
    rng = random.Random(FLAGS.random_seed)

    saver.log_info(f"[ZIP COUNTS] available per zip: {dict(zip([basename(z) for z in zip_files], available_counts))}")
    saver.log_info(f"[ZIP QUOTAS] target per zip: {dict(zip([basename(z) for z in zip_files], quotas))}")

    sampled_zip_rows = _prepare_sampled_zip_rows(zip_files, quotas, rng)
    total_sampled = sum(len(rows) for _, rows in sampled_zip_rows)
    build_progress_every = _choose_progress_every(total_sampled)
    _progress(
        f"[BUILD SETUP] total_sampled={total_sampled} progress_every={build_progress_every} "
    )

    if getattr(FLAGS, "encoder_path", None) is not None:
        saver.info(f"loading encoder from {FLAGS.encoder_path}")
        _progress(f"[ENCODER] loading existing encoders from {FLAGS.encoder_path}")
        encoders = load(FLAGS.encoder_path, saver.logdir)
        _progress(f"[ENCODER] loaded existing encoders")
    else:
        encoders = _fit_encoders_from_sampled_rows(sampled_zip_rows)

    init_feat_dict = {}
    nns, ads = [], []
    built_data = []
    saved_files = []
    graph_idx = 0
    skipped = 0

    if getattr(FLAGS, "force_regen", False):
        saver.log_info(f"Streaming-save enabled; deleting existing files in {SAVE_DIR}")
        import shutil as _shutil
        if exists(SAVE_DIR):
            _shutil.rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        _progress(f"[SAVE SETUP] cleared save dir {SAVE_DIR}")

    for zip_idx, (zfp, rows) in enumerate(sampled_zip_rows):
        zip_t0 = time.time()
        # _progress(
        #     f"[BUILD ZIP {zip_idx + 1}/{len(sampled_zip_rows)}] start name={basename(zfp)} rows={len(rows)}"
        # )
        with ZipFile(zfp, "r") as z:
            for row_idx, row in enumerate(rows, start=1):
                design_id = str(row["design_id"])
                try:
                    gexf_fp = _extract_single_gexf_from_artifacts_zip(z, design_id)
                    g = nx.read_gexf(gexf_fp)
                except Exception as e:
                    saver.warning(f"skip design_id={design_id} due to missing graph: {e}")
                    skipped += 1
                    current_done = graph_idx + skipped
                    # if current_done % build_progress_every == 0 or current_done == total_sampled:
                    #     _progress(
                    #         f"[BUILD] done={current_done}/{total_sampled} built={graph_idx} skipped={skipped} "
                    #         f"elapsed={_format_seconds(time.time() - build_t0)} "
                    #     )
                    continue

                d_node = _encode_X_dict(g)
                d_edge = _encode_edge_dict(g)
                data, design_id = _build_pyg_data_from_graph(g, row, d_node, d_edge, encoders)

                init_feat_dict[design_id] = [1, 1]
                nns.append(data.x.shape[0])
                ads.append(data.edge_index.shape[1] / max(data.x.shape[0], 1))

                if getattr(FLAGS, "force_regen", False):
                    out_fp = osp.join(SAVE_DIR, f"data_{graph_idx}.pt")
                    torch.save(data, out_fp)
                    saved_files.append(out_fp)
                else:
                    built_data.append(data)

                graph_idx += 1
                current_done = graph_idx + skipped
                # if current_done % build_progress_every == 0 or current_done == total_sampled:
                #     _progress(
                #         f"[BUILD] done={current_done}/{total_sampled} built={graph_idx} skipped={skipped} "
                #         f"elapsed={_format_seconds(time.time() - build_t0)} "
                #     )

                del g, d_node, d_edge, data
                if current_done % build_progress_every == 0:
                    gc.collect()

        _progress(
            f"[BUILD ZIP {zip_idx + 1}/{len(sampled_zip_rows)}] done name={basename(zfp)} "
            f"zip_elapsed={_format_seconds(time.time() - zip_t0)} built={graph_idx} skipped={skipped} "
        )

    if nns:
        print_stats(nns, "number of nodes")
    if ads:
        print_stats(ads, "avg degrees")

    _progress(
        f"[SUMMARY] built={graph_idx} skipped={skipped} elapsed={_format_seconds(time.time() - build_t0)} "
    )

    if getattr(FLAGS, "force_regen", False):
        _progress("[FINALIZE] saving encoders and pragma_dim")
        save(encoders, ENCODER_PATH)
        save(init_feat_dict, join(SAVE_DIR, "pragma_dim"))
        _progress(
            f"[DONE] saved_graphs={len(saved_files)} elapsed={_format_seconds(time.time() - build_t0)} "
        )
        return MyOwnDataset(data_files=saved_files), init_feat_dict

    saver.warning(
        "Dataset was built in memory but not saved because force_regen=False. "
        "For build_dataset, use --force_regen True so streaming save can keep memory low."
    )
    _progress(
        f"[DONE NO SAVE] in_memory_graphs={len(built_data)} elapsed={_format_seconds(time.time() - build_t0)} "
    )
    return MyOwnDataset(), init_feat_dict


def get_data_list():
    return get_data_list_from_hlsfactory_zips()