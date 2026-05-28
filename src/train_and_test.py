from pathlib import Path
from pprint import pformat
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import make_tf_dataset
from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_mixed_split, save_split
from src.metrics import challenge_score_from_arrays, sigmoid
from src.model import build_model
from src.train import format_duration, predict_dataset, train_finetune, train_head_only
from src.utils import set_seed


def print_run_header(cfg: dict) -> None:
    print("Run configuration")
    print(pformat(cfg, sort_dicts=True))
    print()


def print_data_summary(train_rows, val_rows) -> None:
    print("Data")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")
    print()


def best_epoch_from_history(history, phase: str) -> tuple[str, float] | None:
    if history is None:
        return None
    scores = history.history.get("val_challenge_score", [])
    if not scores:
        return None
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return f"{phase} epoch {best_index + 1}", scores[best_index]


def best_epoch_from_histories(head_history, finetune_history) -> tuple[str, float]:
    candidates = [
        best_epoch_from_history(head_history, "head"),
        best_epoch_from_history(finetune_history, "finetune"),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    return max(candidates, key=lambda candidate: candidate[1])


def print_experiment_summary(
    cfg: dict,
    head_history,
    finetune_history,
    final_score: float,
    training_seconds: float,
) -> None:
    best_epoch, best_score = best_epoch_from_histories(head_history, finetune_history)
    print()
    print("Experiment summary")
    print(f"experiment name: {cfg['experiment_name']}")
    print(f"seed: {cfg['seed']}")
    print(f"loss: {cfg['loss']}")
    print(f"sampling: {cfg['sampling']}")
    print(f"augmentation: {cfg['augmentation']}")
    print(f"head_lr: {cfg['head_lr']}")
    print(f"dropout: {cfg['dropout']}")
    print(f"best epoch: {best_epoch}")
    print(f"best val_challenge_score: {best_score:.5f}")
    print(f"final val_challenge_score: {final_score:.5f}")
    print(f"training time: {format_duration(training_seconds)}")
    print(f"notes: {cfg['notes']}")


def main():
    run_start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    print_run_header(cfg)
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
    print_data_summary(train_rows, val_rows)
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")

    train_ds = make_tf_dataset(train_rows, batch_size=cfg["batch_size"], training=True)
    val_ds = make_tf_dataset(val_rows, batch_size=cfg["batch_size"], training=False)

    model = build_model(cfg, trainable_backbone=False)
    print("Model")
    model.summary(print_fn=print)
    print()

    train_start = perf_counter()
    head_history = train_head_only(model, train_ds, val_ds, label_space.labels, cfg)
    finetune_history = train_finetune(model, train_ds, val_ds, label_space.labels, cfg)
    training_seconds = perf_counter() - train_start
    print(f"Total training time: {format_duration(training_seconds)}")

    eval_start = perf_counter()
    y_true, logits = predict_dataset(model, val_ds)
    score = challenge_score_from_arrays(y_true, sigmoid(logits), label_space.labels)
    print(f"Evaluation time: {format_duration(perf_counter() - eval_start)}")
    print(f"validation challenge score: {score:.5f}")
    print_experiment_summary(cfg, head_history, finetune_history, score, training_seconds)
    print(f"Total run time: {format_duration(perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
