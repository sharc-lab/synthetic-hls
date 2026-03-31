import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from hlsfactory.data_packaging import DataAggregatorXilinx, CompleteHLSData
from hlsfactory.framework import Design, DesignDataset
from hlsfactory.utils import get_work_dir, remove_and_make_new_dir_if_exists

ArtifactCollection = dict[str, list[Path] | None]

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "zip_data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()

raw_data_dir = CURRENT_DIR / "saved_data"

dataset_base = DesignDataset.from_dir(
    "dataset_base__post_frontend",
    raw_data_dir / "dataset_base__post_frontend",
)

dataset_final = DesignDataset.from_dir(
    "dataset_final__post_frontend",
    raw_data_dir / "dataset_final__post_frontend",
)

designs = (
    dataset_final.designs
)

xilinx_aggregator = DataAggregatorXilinx()
data_all: list[CompleteHLSData] = []

for design in designs:
    hls_design_data = xilinx_aggregator.gather_hls_design_data(design)
    hls_synthesis_data = xilinx_aggregator.gather_hls_synthesis_data(design)
    execution_data = xilinx_aggregator.gather_execution_data(design)
    data: ArtifactCollection = {}
    graph_files = list(design.dir.glob("*.gexf"))
    data["graph"] = [graph_files[0]]

    archive_buffer = io.BytesIO()
    archive = ZipFile(archive_buffer, "w", ZIP_DEFLATED)
    top_level = "artifacts"
    archive.write(data["graph"][0], f"{top_level}/graph/{data['graph'][0].name}")
    archive.close()

    design_data = CompleteHLSData(
        design=hls_design_data,
        synthesis=hls_synthesis_data,
        implementation=None,
        execution=execution_data,
        artifacts=archive_buffer,
    )

    data_all.append(design_data)

output_archive_fp = DATA_DIR / "dataset_final.zip"
xilinx_aggregator.aggregated_data_to_archive(
    data_all,
    output_archive_fp,
)
print(output_archive_fp)
