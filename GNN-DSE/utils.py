# utils.py
from __future__ import annotations

import getpass
import pickle
import socket
import time
from os.path import dirname as _dirname
from pathlib import Path
from typing import Any, Iterable

import torch

# Basic environment helpers
def get_user() -> str:
    return getpass.getuser()

def get_host() -> str:
    return socket.gethostname()

def get_ts() -> str:
    """Timestamp string for log folders."""
    return time.strftime("%Y%m%d-%H%M%S")

def dirname(p: str) -> str:
    return _dirname(p)

def get_root_path() -> str:
    """Project root path (folder containing this file)."""
    return str(Path(__file__).resolve().parent)

def get_src_path() -> str:
    """Alias used by Saver for code snapshot."""
    return get_root_path()

# IO helpers
def create_dir_if_not_exists(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save(obj: Any, path: str) -> None:
    create_dir_if_not_exists(str(Path(path).parent))
    torch.save(obj, path)

def load(path: str, map_location: str | None = None) -> Any:
    if map_location is None:
        return torch.load(path, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)

def save_pickle(obj: Any, path: str) -> None:
    create_dir_if_not_exists(str(Path(path).parent))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Dataset save path helper
def get_save_path() -> str:
    from config import FLAGS
    root = Path(get_root_path()) / "save"
    folder = f"{FLAGS.dataset}__{FLAGS.tag}__{FLAGS.graph_type}"
    return str(root / folder)

# Small reporting helpers
def print_stats(name: str, values: Iterable[Any]) -> None:
    values = list(values)
    print(f"[{name}] count={len(values)}")

def _get_y_with_target(data, target: str):
    return getattr(data, target.replace("-", "_"))

def get_model_info_as_str(model: torch.nn.Module) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"params={n_params}, trainable={n_trainable}"

class OurTimer:
    def __init__(self):
        self.t0 = time.time()
    def reset(self):
        self.t0 = time.time()
    def elapsed(self) -> float:
        return time.time() - self.t0
    def time_and_clear(self) -> float:
        """Return elapsed seconds since last reset, then reset."""
        t = self.elapsed()
        self.reset()
        return t

class MLP(torch.nn.Module):
    """
    Compatible with the original GNN-DSE/HARP-style API used in model.py.

    Expected call pattern in model.py:
        MLP(in_channels, out_channels,
            activation_type=...,
            hidden_channels=[...],
            num_hidden_lyr=...)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation_type: str = "elu",
        hidden_channels=None,
        num_hidden_lyr: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_list = []
        elif isinstance(hidden_channels, (list, tuple)):
            hidden_list = [int(x) for x in hidden_channels]
        else:
            # allow passing a single int
            hidden_list = [int(hidden_channels)]

        # Sanitize: remove duplicated endpoints if present
        if len(hidden_list) > 0 and hidden_list[0] == int(in_channels):
            hidden_list = hidden_list[1:]
        if len(hidden_list) > 0 and hidden_list[-1] == int(out_channels):
            hidden_list = hidden_list[:-1]

        # If num_hidden_lyr is provided, it refers to the number of hidden layers.
        # If it conflicts with hidden_list length, we keep hidden_list as source of truth.
        _ = num_hidden_lyr  # kept for API compatibility

        act = (activation_type or "elu").lower()
        if act == "relu":
            act_layer = torch.nn.ReLU
        elif act == "leakyrelu":
            act_layer = lambda: torch.nn.LeakyReLU(negative_slope=0.2)
        elif act == "gelu":
            act_layer = torch.nn.GELU
        elif act == "tanh":
            act_layer = torch.nn.Tanh
        else:
            # default: ELU (matches HARP defaults in many configs)
            act_layer = torch.nn.ELU

        layers = []
        prev = int(in_channels)
        for h in hidden_list:
            layers.append(torch.nn.Linear(prev, int(h)))
            layers.append(act_layer() if callable(act_layer) else act_layer)
            if dropout and dropout > 0:
                layers.append(torch.nn.Dropout(p=float(dropout)))
            prev = int(h)

        layers.append(torch.nn.Linear(prev, int(out_channels)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP_multi_objective(torch.nn.Module):
    """
    Multi-head MLP used by model.py when FLAGS.target is a list.

    model.py expects:
        out = self.MLPs(out_embed)  -> dict-like
        out[target_name] -> Tensor
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation_type: str = "elu",
        hidden_channels=None,
        objectives=None,
        num_common_lyr: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if objectives is None:
            raise ValueError("MLP_multi_objective requires `objectives` (list of target names).")
        self.objectives = [str(o) for o in objectives]

        if hidden_channels is None:
            hidden_list = []
        elif isinstance(hidden_channels, (list, tuple)):
            hidden_list = [int(x) for x in hidden_channels]
        else:
            hidden_list = [int(hidden_channels)]

        # Sanitize endpoints
        if len(hidden_list) > 0 and hidden_list[0] == int(in_channels):
            hidden_list = hidden_list[1:]
        if len(hidden_list) > 0 and hidden_list[-1] == int(out_channels):
            hidden_list = hidden_list[:-1]

        # split hidden layers into shared + per-head
        num_common_lyr = max(0, int(num_common_lyr))
        shared_hidden = hidden_list[:num_common_lyr]
        head_hidden = hidden_list[num_common_lyr:]

        shared_out_dim = int(in_channels) if len(shared_hidden) == 0 else int(shared_hidden[-1])

        # shared trunk (may be identity)
        if len(shared_hidden) == 0:
            self.shared = torch.nn.Identity()
        else:
            self.shared = MLP(
                int(in_channels),
                shared_out_dim,
                activation_type=activation_type,
                hidden_channels=shared_hidden,
                num_hidden_lyr=len(shared_hidden),
                dropout=dropout,
            )

        # per-objective heads
        self.heads = torch.nn.ModuleDict()
        for obj in self.objectives:
            self.heads[obj] = MLP(
                shared_out_dim,
                int(out_channels),
                activation_type=activation_type,
                hidden_channels=head_hidden,
                num_hidden_lyr=len(head_hidden),
                dropout=dropout,
            )

    def forward(self, x):
        h = self.shared(x)
        return {obj: self.heads[obj](h) for obj in self.objectives}

# Plotting helpers
def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_loss_trend(epochs, train_losses, val_losses, test_losses, out_dir, file_name="losses.png"):
    """
    Saves a single plot with train/val/test curves.
    """
    plt = _safe_import_matplotlib()
    create_dir_if_not_exists(out_dir)
    save_path = str(Path(out_dir) / file_name)

    plt.figure()
    plt.plot(list(epochs), list(train_losses), label="train")
    if val_losses is not None and len(val_losses) > 0:
        plt.plot(list(epochs), list(val_losses), label="val")
    if test_losses is not None and len(test_losses) > 0:
        plt.plot(list(epochs), list(test_losses), label="test")
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_lr_trend(lr_list, epoch_num, out_dir, file_name="lr.png"):
    plt = _safe_import_matplotlib()
    create_dir_if_not_exists(out_dir)
    save_path = str(Path(out_dir) / file_name)

    plt.figure()
    plt.plot(list(range(len(lr_list))), list(lr_list))
    plt.title("lr")
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_points_with_subplot(*args, **kwargs):
    return None

def plot_points_with_subplot_sigma(*args, **kwargs):
    return None

def plot_scatter_line(*args, **kwargs):
    return None

def plot_dist(*args, **kwargs):
    return None

def extract_config_code() -> str:
    """Return a compact snapshot of FLAGS for logging."""
    from config import FLAGS  # lazy import
    keys = sorted(vars(FLAGS).keys())
    lines = [f"{k}={getattr(FLAGS, k)!r}" for k in keys]
    return "\n".join(lines)
