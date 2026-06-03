from __future__ import annotations

from pathlib import Path
from time import perf_counter
import argparse
import csv
import re
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


SCORE_RE = re.compile(r"validation challenge score:\s*([0-9.]+)")
BEST_SCORE_RE = re.compile(r"best val_challenge_score:\s*([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config-defined Perch or PaSST experiment across CV folds.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to a YAML config file")
    return parser.parse_args()


def fold_list(cfg: dict) -> list[int]:
    folds = cfg.get("cv_folds_to_run")
    if folds:
        if isinstance(folds, str):
            return [int(part.strip()) for part in folds.split(",") if part.strip()]
        return [int(fold) for fold in folds]
    return list(range(int(cfg["soundscape_cv_folds"])))


def relative_project_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cv_model(cfg: dict) -> str:
    model = str(cfg.get("cv_model", "")).lower()
    if model not in {"perch", "passt"}:
        raise ValueError("Config must set cv_model to 'perch' or 'passt'")
    return model


def cv_python(cfg: dict) -> str:
    return str(cfg.get("cv_python") or sys.executable)


def apply_fold_paths(cfg: dict, model: str, experiment_name: str, cache_name: str, fold: int) -> dict:
    cfg = dict(cfg)
    fold_name = f"{experiment_name}_fold{fold}"
    cfg["experiment_name"] = fold_name
    cfg["soundscape_eval_split"] = "cv"
    cfg["soundscape_fold"] = fold

    if model == "perch":
        cfg["train_embedding_cache_path"] = f"outputs/embedding_cache/{cache_name}/fold{fold}_train.npz"
        cfg["val_embedding_cache_path"] = f"outputs/embedding_cache/{cache_name}/fold{fold}_val.npz"
        cfg["best_model_weights_path"] = f"outputs/checkpoints/{experiment_name}/fold{fold}_best.weights.h5"
        cfg["save_model_weights_path"] = f"outputs/checkpoints/{experiment_name}/fold{fold}_latest.weights.h5"
        cfg["perch_val_predictions_path"] = f"outputs/predictions/{experiment_name}/fold{fold}_perch_val_predictions.npz"
    else:
        cfg["passt_train_cache_path"] = f"outputs/passt_cache/{cache_name}/fold{fold}_train.npz"
        cfg["passt_val_cache_path"] = f"outputs/passt_cache/{cache_name}/fold{fold}_val.npz"
        cfg["passt_best_model_weights_path"] = f"outputs/checkpoints/{experiment_name}/fold{fold}_passt_best.pt"
        cfg["passt_save_model_weights_path"] = f"outputs/checkpoints/{experiment_name}/fold{fold}_passt_latest.pt"
        cfg["passt_val_predictions_path"] = f"outputs/predictions/{experiment_name}/fold{fold}_passt_val_predictions.npz"

    return cfg


def write_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def command_for(model: str, python_executable: str, config_path: Path) -> list[str]:
    module = "src.perch.train_and_test" if model == "perch" else "src.passt.train_and_test"
    return [python_executable, "-m", module, "--config", str(config_path)]


def extract_scores(output: str) -> tuple[str, str]:
    final_scores = SCORE_RE.findall(output)
    best_scores = BEST_SCORE_RE.findall(output)
    final_score = final_scores[-1] if final_scores else ""
    best_score = best_scores[-1] if best_scores else ""
    return final_score, best_score


def artifact_paths(cfg: dict, model: str) -> dict[str, str]:
    if model == "perch":
        return {
            "prediction_path": cfg.get("perch_val_predictions_path", ""),
            "best_checkpoint_path": cfg.get("best_model_weights_path", ""),
            "latest_checkpoint_path": cfg.get("save_model_weights_path", ""),
            "train_cache_path": cfg.get("train_embedding_cache_path", ""),
            "val_cache_path": cfg.get("val_embedding_cache_path", ""),
        }
    return {
        "prediction_path": cfg.get("passt_val_predictions_path", ""),
        "best_checkpoint_path": cfg.get("passt_best_model_weights_path", ""),
        "latest_checkpoint_path": cfg.get("passt_save_model_weights_path", ""),
        "train_cache_path": cfg.get("passt_train_cache_path", ""),
        "val_cache_path": cfg.get("passt_val_cache_path", ""),
    }


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "experiment_name",
        "fold",
        "return_code",
        "duration_seconds",
        "final_val_challenge_score",
        "best_val_challenge_score",
        "config_path",
        "log_path",
        "prediction_path",
        "best_checkpoint_path",
        "latest_checkpoint_path",
        "train_cache_path",
        "val_cache_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    model = cv_model(base_cfg)
    experiment_name = str(base_cfg["experiment_name"])
    cache_name = str(base_cfg.get("cv_cache_name") or experiment_name)
    folds = fold_list(base_cfg)
    dry_run = bool(base_cfg.get("cv_dry_run", False))
    python_executable = cv_python(base_cfg)

    config_dir = PROJECT_ROOT / "outputs" / "cv_configs" / experiment_name
    log_dir = PROJECT_ROOT / "outputs" / "cv_logs" / experiment_name
    summary_path = PROJECT_ROOT / "outputs" / "cv_summaries" / f"{experiment_name}_{model}.csv"
    rows = []

    for fold in folds:
        fold_cfg = apply_fold_paths(base_cfg, model, experiment_name, cache_name, fold)
        config_path = config_dir / f"{model}_fold{fold}.yaml"
        log_path = log_dir / f"{model}_fold{fold}.log"
        write_config(fold_cfg, config_path)
        command = command_for(model, python_executable, config_path)
        print(f"Fold {fold}: {' '.join(command)}")

        start = perf_counter()
        return_code = 0
        output = ""
        if dry_run:
            output = "dry run\n"
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()
                    lines.append(line)
                return_code = process.wait()
            output = "".join(lines)

        duration = perf_counter() - start
        final_score, best_score = extract_scores(output)
        row = {
            "model": model,
            "experiment_name": fold_cfg["experiment_name"],
            "fold": fold,
            "return_code": return_code,
            "duration_seconds": f"{duration:.1f}",
            "final_val_challenge_score": final_score,
            "best_val_challenge_score": best_score,
            "config_path": relative_project_path(config_path),
            "log_path": relative_project_path(log_path),
        }
        row.update({key: relative_project_path(value) for key, value in artifact_paths(fold_cfg, model).items()})
        rows.append(row)
        write_summary(summary_path, rows)
        print(
            f"Fold {fold} finished: return_code={return_code}, "
            f"final_score={final_score or 'n/a'}, best_score={best_score or 'n/a'}"
        )
        if return_code != 0:
            raise SystemExit(return_code)

    print(f"Wrote fold summary to {summary_path}")


if __name__ == "__main__":
    main()
