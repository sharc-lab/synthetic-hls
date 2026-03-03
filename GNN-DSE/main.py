# main.py
from __future__ import annotations

from os.path import join

from config import FLAGS
from utils import load

from data import get_data_list, MyOwnDataset


def _load_pragma_dim(dataset: MyOwnDataset):
    # pragma_dim is saved by data.py as SAVE_DIR/pragma_dim (via utils.save)
    pragma_dim_fp = join(dataset.root, "pragma_dim")
    try:
        return load(pragma_dim_fp, print_msg=False, saver=None)
    except Exception:
        return {"all": [1, 1]}  # safe fallback


if __name__ == "__main__":
    # Subtasks:
    #   - build_dataset: generate and save .pt dataset files
    #   - train: train GNN model on saved .pt dataset
    #   - inference: run inference on saved .pt dataset using a trained model
    #   - rank: use trained model to rank DSE samples
    if FLAGS.subtask == "build_dataset":
        if not FLAGS.force_regen:
            print("[main.py] build_dataset usually expects --force_regen True to write .pt files.")
        dataset, pragma_dim = get_data_list()

        print("[DONE] build_dataset finished.")
        raise SystemExit(0)

    # Non-build tasks use existing saved .pt dataset
    from saver import saver
    saver.new_sub_saver("run1")

    dataset = MyOwnDataset()
    pragma_dim = _load_pragma_dim(dataset)

    if FLAGS.subtask == "train":
        from train import train_main
        test_ratio, resample_list = FLAGS.val_ratio, [-1]
        if FLAGS.resample:
            test_ratio, resample_list = 0.25, list(range(4))

        for ind, r in enumerate(resample_list):
            saver.info(f"Starting training with resample {r}")
            train_main(dataset, pragma_dim, test_ratio=test_ratio, resample=r)
            if ind + 1 < len(resample_list):
                saver.new_sub_saver(subdir=f"run{ind+2}")
                saver.log_info("\n\n")

    elif FLAGS.subtask == "inference":
        if FLAGS.model_path is None:
            raise RuntimeError("--model_path must be set for inference")
        from train import inference
        inference(dataset, init_pragma_dict=pragma_dim, model_path=FLAGS.model_path, test_ratio=FLAGS.val_ratio)

    elif FLAGS.subtask == "rank":
        from dse import run_inference_and_rank
        run_inference_and_rank(dataset, pragma_dim=pragma_dim)

    else:
        raise NotImplementedError(f"Unknown subtask: {FLAGS.subtask}")

    saver.close()
