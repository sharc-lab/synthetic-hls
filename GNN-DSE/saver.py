# saver.py
from __future__ import annotations

import csv
import os
import time
from os.path import join
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional

import torch

from config import FLAGS
from utils import (
    create_dir_if_not_exists,
    get_ts,
    get_host,
    get_src_path,
    extract_config_code,
    save,
)

# Prefer torch tensorboard; fallback to no-op
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None


class _NullWriter:
    def add_text(self, *args, **kwargs): return None
    def add_scalar(self, *args, **kwargs): return None
    def close(self): return None


class Saver:
    """
    Minimal, import-safe saver.
    IMPORTANT: does NOT create sub-runs or open files at import time.
    Call saver.new_sub_saver('run1') when you actually start train/inference/rank.
    """
    def __init__(self):
        self.writer = _NullWriter()
        self.subdir = None

        directory_name = f"results_{FLAGS.dataset}_{FLAGS.tag}_{get_ts()}"
        self.logdir = join(get_src_path(), "logs", directory_name)
        create_dir_if_not_exists(self.logdir)

        # will be set by new_sub_saver()
        self.run_dir = self.logdir
        self.model_logdir = join(self.logdir, "model")
        create_dir_if_not_exists(self.model_logdir)

        print(f"Logging root to {self.logdir}")

    # basic logging
    def _write_line(self, msg: str):
        fp = join(self.run_dir, "log.txt")
        with open(fp, "a") as f:
            f.write(msg.rstrip() + "\n")

    def info(self, msg: str):
        print(msg)
        self._write_line(f"[INFO] {msg}")

    def warning(self, msg: str):
        print(msg)
        self._write_line(f"[WARN] {msg}")

    def error(self, msg: str):
        print(msg)
        self._write_line(f"[ERROR] {msg}")

    def log_info(self, msg: Any):
        s = msg if isinstance(msg, str) else pformat(msg)
        self.info(s)

    # run management
    def new_sub_saver(self, subdir: str = "run1"):
        self.subdir = subdir
        self.run_dir = join(self.logdir, subdir)
        create_dir_if_not_exists(self.run_dir)
        self.model_logdir = join(self.run_dir, "model")
        create_dir_if_not_exists(self.model_logdir)

        if SummaryWriter is None:
            self.writer = _NullWriter()
        else:
            self.writer = SummaryWriter(join(self.run_dir, "runs"))

        # dump info/config snapshot once
        self._save_info_snapshot()
        return self

    def _save_info_snapshot(self):
        s = ""
        s += "==========INFO==========\n"
        s += f"user: {FLAGS.user}\n"
        s += f"hostname: {get_host()}\n"
        s += f"tstamp: {get_ts()}\n\n"
        s += "FLAGS:\n"
        s += pformat(vars(FLAGS), width=120) + "\n\n"
        s += "config:\n" + extract_config_code() + "\n"
        s += "========================\n"

        fp = join(self.run_dir, "info.txt")
        with open(fp, "w") as f:
            f.write(s)
        try:
            self.writer.add_text("info", s)
        except Exception:
            pass

    # misc utilities used by train.py
    def get_log_dir(self) -> str:
        return self.run_dir

    def log_model_architecture(self, model: torch.nn.Module):
        fp = join(self.run_dir, "model.txt")
        with open(fp, "w") as f:
            f.write(str(model) + "\n")
        try:
            self.writer.add_text("model", str(model))
        except Exception:
            pass

    def save_model_state_dict(self, model: torch.nn.Module, name: str = "train_model_state_dict.pth"):
        save(model.state_dict(), join(self.model_logdir, name))

    def log_dict_of_dicts_to_csv(self, name: str, d: Dict[str, Dict[str, Any]], header: list[str]):
        fp = join(self.run_dir, f"{name}.csv")
        # ensure header unique, stable
        hdr = []
        for h in header:
            if h not in hdr:
                hdr.append(h)
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            for k, row in d.items():
                if k == "header":
                    continue
                w.writerow({h: row.get(h, "") for h in hdr})

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass


saver = Saver()
