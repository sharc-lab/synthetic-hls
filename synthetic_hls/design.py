import shutil
import tomllib
from pathlib import Path
from typing import Self

CPP_EXTENSIONS = [
    ".c",
    ".cc",
    ".cpp",
]
H_EXTENSIONS = [
    ".h",
    ".hh",
    ".hpp",
]
SOURCE_FILE_EXTENSIONS = CPP_EXTENSIONS + H_EXTENSIONS


class Design:
    def __init__(self, design_dir: Path, name: str | None = None, tags: list[str] = []):
        self.design_dir = design_dir
        if name is None:
            name = design_dir.name
        self.name = name
        self.tags = tags

        if not self.design_dir.exists():
            raise FileNotFoundError(
                f"Design directory {self.design_dir} does not exist"
            )

        if not self.design_dir.is_dir():
            raise NotADirectoryError(
                f"Design directory {self.design_dir} is not a directory"
            )

    @property
    def files(self) -> list[Path]:
        return [f for f in self.design_dir.glob("*") if f.is_file()]

    @property
    def source_files(self) -> list[Path]:
        return [f for f in self.files if f.suffix in SOURCE_FILE_EXTENSIONS]

    @property
    def h_files(self) -> list[Path]:
        return [f for f in self.files if f.suffix in H_EXTENSIONS]

    @property
    def cpp_files(self) -> list[Path]:
        return [f for f in self.files if f.suffix in CPP_EXTENSIONS]

    @property
    def not_source_files(self) -> list[Path]:
        return [f for f in self.files if f.suffix not in SOURCE_FILE_EXTENSIONS]

    @property
    def tb_file(self) -> Path:
        tb_matches = [f for f in self.files if f.name.endswith("_tb.cpp")]
        if len(tb_matches) != 1:
            raise ValueError(f"Expected 1 _tb file, found {len(tb_matches)}")
        return tb_matches[0]

    @property
    def kernel_fp(self) -> Path:
        cpp_files = self.cpp_files
        cpp_files = [f for f in cpp_files if f != self.tb_file]
        if len(cpp_files) != 1:
            raise ValueError(f"Expected 1 kernel file, found {len(cpp_files)}")
        return cpp_files[0]

    @property
    def kernel_description_fp(self) -> Path:
        fp = self.design_dir / "kernel_description.md"
        if not fp.exists():
            raise FileNotFoundError(f"Kernel description file {fp} does not exist")
        return fp

    @property
    def top_file(self) -> Path:
        return self.design_dir / "top.txt"

    @property
    def top_fn(self) -> str:
        top_fn = self.top_file.read_text().strip()
        if not top_fn:
            raise ValueError(f"Top file {self.top_file} is empty")
        return top_fn

    @property
    def opt_fp(self) -> Path:    
        fp = self.design_dir / "opt_template.tcl"
        if not fp.exists():
            return None
        return fp

    @property
    def pareto_score_fp(self) -> Path:
        fp = self.design_dir / "pareto_score.txt"
        if not fp.exists():
            return None
        return fp

    @property
    def call_graph_fp(self) -> Path:
        fp = self.design_dir / "call_grdaph.json"
        if not fp.exists():
            return None
        return fp   

    @property
    def toml_data(self) -> dict:
        return tomllib.loads((self.design_dir / "hls_eval_config.toml").read_text())

    @property
    def tags_all(self) -> list[str]:
        return self.toml_data.get("tags", []) + self.tags

    @property
    def tags_in_config(self) -> list[str]:
        return self.toml_data.get("tags", [])

    def copy_to(self, dest: Path) -> "Design":
        shutil.copytree(
            self.design_dir,
            dst=dest,
        )
        return Design(dest, tags=self.tags)

def find_design_dirs(start_dir) -> list[Path]:
    all_dirs = [d for d in start_dir.rglob("*") if d.is_dir()]
    design_dirs = [d for d in all_dirs if (d / "hls_eval_config.toml").exists()]
    return design_dirs