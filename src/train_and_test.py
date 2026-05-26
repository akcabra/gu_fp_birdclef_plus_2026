from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import make_tf_dataset
from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_mixed_split, save_split
from src.metrics import challenge_score_from_arrays, sigmoid
from src.model import build_model
from src.train import predict_dataset, train_finetune, train_head_only
from src.utils import set_seed


def main():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    set_seed(cfg["seed"])

    train_csv, soundscape_csv, _, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)

    train_rows, val_rows = make_mixed_split(
        train_csv,
        soundscape_csv,
        cfg["data_root"],
        cfg["soundscape_val_fraction"],
        cfg["seed"],
    )
    train_rows = attach_targets(train_rows, label_space)
    val_rows = attach_targets(val_rows, label_space)
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")

    train_ds = make_tf_dataset(train_rows, batch_size=cfg["batch_size"], training=True)
    val_ds = make_tf_dataset(val_rows, batch_size=cfg["batch_size"], training=False)

    model = build_model(cfg, trainable_backbone=False)
    train_head_only(model, train_ds, val_ds, label_space.labels, cfg)
    train_finetune(model, train_ds, val_ds, label_space.labels, cfg)

    y_true, logits = predict_dataset(model, val_ds)
    score = challenge_score_from_arrays(y_true, sigmoid(logits), label_space.labels)
    print(f"validation challenge score: {score:.5f}")


if __name__ == "__main__":
    main()
