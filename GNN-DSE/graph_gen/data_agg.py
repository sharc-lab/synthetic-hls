from pathlib import Path

from hlsfactory.data_packaging import DataAggregatorXilinx
from hlsfactory.framework import DesignDataset
from hlsfactory.utils import get_work_dir, remove_and_make_new_dir_if_exists

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "zip_data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()

test_data_dir = CURRENT_DIR / "saved_data"

dataset_final = DesignDataset.from_dir(
    "dataset_final__post_frontend",
    test_data_dir / "dataset_final__post_frontend",
)

designs = (
    dataset_final.designs
)

xilinx_aggregator = DataAggregatorXilinx()

data = xilinx_aggregator.gather_multiple_designs(designs, n_jobs=32)
output_archive_fp = DATA_DIR / "dataset_final.zip"
xilinx_aggregator.aggregated_data_to_archive(
    data,
    output_archive_fp,
)
print(output_archive_fp)
