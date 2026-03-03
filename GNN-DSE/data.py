# data.py
from __future__ import annotations

import ast

import io
import tempfile
from collections import Counter, defaultdict
from glob import glob
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
        return glob(join(SAVE_DIR, "*.pt"))

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


def _torch_load_pt(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        return torch.load(path)


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
    """Return a dataset filtered to a subset of kernels.
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

    # fallback: choose the first kernel deterministically (sorted for reproducibility)
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
    edge_index = torch.LongTensor(list(g.edges)).t().contiguous()
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
    # [MOD] Ensure node ids are contiguous integers aligned with create_edge_index()
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
            # [MOD] node ids are ints now
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


def _list_zip_files(glob_expr: str) -> list[str]:
    import glob as _glob
    files = sorted(_glob.glob(glob_expr, recursive=True))
    if not files:
        raise FileNotFoundError(f"No zip files matched: {glob_expr}")
    return files

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

        # ---- No normalization ----
        if norm_method in ("off", "logmse"):
            return out2

        # ---- GNN-DSE paper normalization ----
        if norm_method == "gnndse":
            eps = float(getattr(FLAGS, "epsilon", 1e-3))
            norm_factor = float(getattr(FLAGS, "normalizer", 1e7))

            # latency objective (stored in `perf`): log2(norm_factor / latency)
            latency = float(out2.get("perf", 0.0))
            latency = max(latency, eps)
            out2["perf"] = float(np.log2(norm_factor / latency))
            out2["actual_perf"] = out2["perf"]

            # resources -> utilization fractions
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

    # lut_total = lut_impl if lut_impl > 0 else lut_syn
    # ff_total = ff_impl if ff_impl > 0 else ff_syn
    # dsp_total = dsp_impl if dsp_impl > 0 else dsp_syn
    # bram_total = bram_impl if bram_impl > 0 else bram_syn

    out["total-LUT"] = FPGA_TOTAL_RESOURCES["LUT"]  # [MOD]
    out["total-FF"] = FPGA_TOTAL_RESOURCES["FF"]  # [MOD]
    out["total-DSP"] = FPGA_TOTAL_RESOURCES["DSP"]  # [MOD]
    out["total-BRAM"] = FPGA_TOTAL_RESOURCES["BRAM"]  # [MOD]

    out["quality"] = 0.0
    out["kernel_speedup"] = 0.0

    out = _apply_target_normalization(out)
    return out


def get_data_list_from_hlsfactory_zips():
    if getattr(FLAGS, "dataset_zips_glob", None) is None:
        raise ValueError("FLAGS.dataset_zips_glob must be set (glob of HLSFactory data_packaging zips).")

    zip_files = _list_zip_files(FLAGS.dataset_zips_glob)
    saver.log_info(f"Found {len(zip_files)} dataset zip(s)")

    if getattr(FLAGS, "encoder_path", None) is not None:
        saver.info(f"loading encoder from {FLAGS.encoder_path}")
        encoders = load(FLAGS.encoder_path, saver.logdir)
        enc_ntype = encoders["enc_ntype"]
        enc_ptype = encoders["enc_ptype"]
        enc_itype = encoders["enc_itype"]
        enc_ftype = encoders["enc_ftype"]
        enc_btype = encoders["enc_btype"]
        enc_ftype_edge = encoders["enc_ftype_edge"]
        enc_ptype_edge = encoders["enc_ptype_edge"]
    else:
        enc_ntype = OneHotEncoder(handle_unknown="ignore")
        enc_ptype = OneHotEncoder(handle_unknown="ignore")
        enc_itype = OneHotEncoder(handle_unknown="ignore")
        enc_ftype = OneHotEncoder(handle_unknown="ignore")
        enc_btype = OneHotEncoder(handle_unknown="ignore")
        enc_ftype_edge = OneHotEncoder(handle_unknown="ignore")
        enc_ptype_edge = OneHotEncoder(handle_unknown="ignore")

    X_ntype_all, X_ptype_all, X_itype_all, X_ftype_all, X_btype_all = [], [], [], [], []
    edge_ftype_all, edge_ptype_all = [], []
    data_list = []
    init_feat_dict = {}

    ntypes = Counter()
    ptypes = Counter()
    itypes = Counter()
    ftypes = Counter()
    btypes = Counter()
    ptypes_edge = Counter()
    ftypes_edge = Counter()

    for zfp in zip_files:
        saver.info(f"[ZIP] {zfp}")
        with ZipFile(zfp, "r") as z:
            csv_name = _pick_data_all_csv_name(z)
            df = pd.read_csv(io.StringIO(_read_zip_text(z, csv_name)))

            if "design_id" not in df.columns:
                raise RuntimeError(f"{csv_name} must contain design_id. got columns: {list(df.columns)}")

            for _, row_s in df.iterrows():
                row = row_s.to_dict()
                design_id = str(row["design_id"])

                try:
                    gexf_fp = _extract_single_gexf_from_artifacts_zip(z, design_id)
                except Exception as e:
                    saver.warning(f"skip design_id={design_id} due to missing graph: {e}")
                    continue

                g = nx.read_gexf(gexf_fp)

                d_node = _encode_X_dict(g, ntypes, ptypes, itypes, ftypes, btypes)
                d_edge = _encode_edge_dict(g, ftypes_edge, ptypes_edge)

                X_ntype_all += d_node["X_ntype"]
                X_ptype_all += d_node["X_ptype"]
                X_itype_all += d_node["X_itype"]
                X_ftype_all += d_node["X_ftype"]
                X_btype_all += d_node["X_btype"]
                edge_ftype_all += d_edge["X_ftype"]
                edge_ptype_all += d_edge["X_ptype"]

                label = _map_row_to_targets(row)

                pragmas = torch.zeros((1, 1), dtype=torch.float32)
                init_feat_dict[design_id] = [1, 1]

                g._harp_tmp = {
                    "d_node": d_node,
                    "d_edge": d_edge,
                    "label": label,
                    "design_id": design_id,
                    "row": row,
                    "pragmas": pragmas,
                }
                data_list.append(g)

    if getattr(FLAGS, "encoder_path", None) is None:
        enc_ptype.fit(X_ptype_all)
        enc_ntype.fit(X_ntype_all)
        enc_itype.fit(X_itype_all)
        enc_ftype.fit(X_ftype_all)
        enc_btype.fit(X_btype_all)
        enc_ftype_edge.fit(edge_ftype_all)
        enc_ptype_edge.fit(edge_ptype_all)
        saver.log_info("Encoder fitting done.")

    pyg_list = []
    for g in data_list:
        tmp = g._harp_tmp
        d_node, d_edge = tmp["d_node"], tmp["d_edge"]
        label = tmp["label"]
        design_id = tmp["design_id"]
        row = tmp["row"]

        X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
        edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)
        edge_index = create_edge_index(g)

        gname = row.get("design__name", design_id)
        kernel = row.get("design__name", gname)

        pyg_list.append(
            Data(
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
                pragmas=tmp["pragmas"],

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

                # totals kept for compatibility even if not trained on
                total_BRAM=torch.FloatTensor(np.array([label["total-BRAM"]])),
                total_DSP=torch.FloatTensor(np.array([label["total-DSP"]])),
                total_LUT=torch.FloatTensor(np.array([label["total-LUT"]])),
                total_FF=torch.FloatTensor(np.array([label["total-FF"]])),
            )
        )

    nns = [d.x.shape[0] for d in pyg_list]
    print_stats(nns, "number of nodes")
    ads = [d.edge_index.shape[1] / d.x.shape[0] for d in pyg_list]
    print_stats(ads, "avg degrees")

    if getattr(FLAGS, "force_regen", False):
        saver.log_info(f"Saving {len(pyg_list)} graphs to disk {SAVE_DIR}; Deleting existing files")
        import shutil as _shutil
        if exists(SAVE_DIR):
            _shutil.rmtree(SAVE_DIR)
        create_dir_if_not_exists(SAVE_DIR)
        for i in range(len(pyg_list)):
            torch.save(pyg_list[i], osp.join(SAVE_DIR, f"data_{i}.pt"))

        obj = {
            "enc_ntype": enc_ntype, "enc_ptype": enc_ptype,
            "enc_itype": enc_itype, "enc_ftype": enc_ftype,
            "enc_btype": enc_btype,
            "enc_ftype_edge": enc_ftype_edge, "enc_ptype_edge": enc_ptype_edge,
        }
        save(obj, ENCODER_PATH)
        save(init_feat_dict, join(SAVE_DIR, "pragma_dim"))

    return MyOwnDataset(), init_feat_dict


def get_data_list():
    return get_data_list_from_hlsfactory_zips()