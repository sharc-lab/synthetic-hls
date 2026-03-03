# dse.py  (inference + ranking only)

from __future__ import annotations

from collections import OrderedDict, defaultdict
from os.path import join, basename
import csv

import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from config import FLAGS
from saver import saver
from model import Net

# We reuse the same dataset split helpers as train.py
from data import get_kernel_samples, split_dataset, split_dataset_resample, split_train_test_kernel
import data
from utils import _get_y_with_target

SAVE_DIR = data.SAVE_DIR


if FLAGS.task != "regression":
    raise ValueError(
        f"[dse.py] This simplified dse.py supports regression-only. Got FLAGS.task={FLAGS.task}"
    )


def _target_list():
    t = FLAGS.target
    if not isinstance(t, list):
        t = [t]
    # keep the same encode_log convention used elsewhere
    return ['actual_perf' if FLAGS.encode_log and x == 'perf' else x for x in t]


def process_split_data(dataset):
    dataset_dict = defaultdict(list)
    dataset_dict['train'] = dataset
    dataset_dict['test'] = None
    if not FLAGS.all_kernels:
        dataset = get_kernel_samples(dataset)
        dataset_dict['train'] = dataset
    elif FLAGS.test_kernels is not None:
        dataset_dict = split_train_test_kernel(dataset)
    return dataset_dict


def get_train_val_count(num_graphs, val_ratio, test_ratio):
    if FLAGS.test_kernels is not None:
        r1 = int(num_graphs * (1.0 - val_ratio))
        r2 = int(num_graphs * (val_ratio))
    else:
        r1 = int(num_graphs * (1.0 - val_ratio - test_ratio))
        r2 = int(num_graphs * (val_ratio))
    return r1, r2


def gen_loaders(split_list):
    train_loader = DataLoader(split_list[0], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(split_list[1], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(split_list[2], batch_size=FLAGS.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    loader = test_loader if len(test_loader.dataset) > 0 else train_loader
    num_features = loader.dataset[0].num_features
    edge_dim = loader.dataset[0].edge_attr.shape[1]
    return train_loader, val_loader, test_loader, num_features, edge_dim


def load_model(num_features: int, edge_dim: int, pragma_dim=None) -> Net:
    model = Net(num_features, edge_dim=edge_dim, init_pragma_dict=pragma_dim).to(FLAGS.device)

    if FLAGS.model_path is None:
        raise RuntimeError("[dse.py] FLAGS.model_path must be set for inference (trained checkpoint).")

    model_path = FLAGS.model_path[0] if isinstance(FLAGS.model_path, list) else FLAGS.model_path
    saver.info(f"[dse.py] Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    saver.log_model_architecture(model)

    return model


def predict_all(model: Net, loader: DataLoader, tag: str):
    """
    Run inference over an existing set of graphs and collect:
      gname, pragmas, true, pred, score
    """
    model.eval()

    targets = _target_list()
    rows = []
    points_dict = OrderedDict({t: {"pred": []} for t in targets})

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"infer:{tag}"):
            batch = batch.to(FLAGS.device)

            out_dict, total_loss, loss_dict, gae_loss = model(batch)

            # metadata extraction
            # NOTE: Your dataset already stores these fields for each graph.
            gnames = _get_y_with_target(batch, "gname")
            pragmas = _get_y_with_target(batch, "pragmas")

            # iterate graphs in batch
            bs = len(gnames)
            for i in range(bs):
                gname_i = gnames[i]
                pragma_i = pragmas[i]
                # pragma may be tensor/list; normalize to a compact string
                if hasattr(pragma_i, "tolist"):
                    pragma_list = pragma_i.tolist()
                else:
                    pragma_list = list(pragma_i)
                pragma_str = "-".join([str(int(x)) for x in pragma_list])

                r = {"gname": gname_i, "pragma": pragma_str}

                # gather per-target predictions and labels
                score = 0.0
                for t in targets:
                    pred = float(out_dict[t][i].item())
                    true = float(_get_y_with_target(batch, t)[i].item())

                    r[f"pred_{t}"] = pred
                    r[f"true_{t}"] = true

                    points_dict[t]["pred"].append((true, pred))

                    score += pred

                r["score"] = score
                rows.append(r)

    return rows, points_dict


def write_csv(rows, out_csv):
    if not rows:
        saver.info(f"[dse.py] No rows to write: {out_csv}")
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    saver.info(f"[dse.py] wrote {out_csv} ({len(rows)} rows)")


def run_inference_and_rank(dataset, pragma_dim=None, val_ratio=FLAGS.val_ratio, test_ratio=FLAGS.val_ratio, resample=-1):
    """
    No space exploration now, just predict + rank existing design-point graphs
    """
    dataset_dict = process_split_data(dataset)
    num_graphs = len(dataset_dict["train"])
    r1, r2 = get_train_val_count(num_graphs, val_ratio, test_ratio)

    if resample == -1:
        splits = split_dataset(dataset_dict["train"], r1, r2, dataset_test=dataset_dict["test"])
    else:
        splits = split_dataset_resample(
            dataset_dict["train"],
            1.0 - val_ratio - test_ratio,
            val_ratio,
            test_ratio,
            test_id=resample,
        )

    train_loader, val_loader, test_loader, num_features, edge_dim = gen_loaders(splits)
    model = load_model(num_features, edge_dim, pragma_dim=pragma_dim)

    # choose which split to run on
    tag = "test" if len(test_loader.dataset) > 0 else "train"
    loader = test_loader if len(test_loader.dataset) > 0 else train_loader

    rows, points_dict = predict_all(model, loader, tag=tag)

    # sort by score (ascending: smaller is better)
    rows_sorted = sorted(rows, key=lambda x: x["score"])

    # output files
    out_dir = saver.get_log_dir()
    out_all = join(out_dir, f"pred_{tag}_all.csv")
    out_top = join(out_dir, f"pred_{tag}_top{getattr(FLAGS, 'topk', 50)}.csv")

    write_csv(rows_sorted, out_all)

    topk = getattr(FLAGS, "topk", 50)
    write_csv(rows_sorted[:topk], out_top)

    saver.info(f"[dse.py] top-{topk} written: {out_top}")

    # optional: print metrics if you use 'inf' subtask convention
    if "inf" in FLAGS.subtask:
        from train import _report_rmse_etc  # reuse the same report function
        _report_rmse_etc(points_dict, f"[{tag}] inference metrics", print_result=True)

def main(dataset, pragma_dim=None):
    run_inference_and_rank(dataset, pragma_dim=pragma_dim)

