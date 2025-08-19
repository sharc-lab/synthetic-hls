import shutil
from pathlib import Path
from typing import Generic, TypeVar

T_unwrap = TypeVar("T_unwrap")


def unwrap(value: T_unwrap | None, msg: str = "Value is None") -> T_unwrap:
    if value is None:
        raise ValueError(msg)
    return value


def auto_find_bin(bin_name: str) -> Path | None:
    match = shutil.which(bin_name)
    if match:
        return Path(match)
    return None
