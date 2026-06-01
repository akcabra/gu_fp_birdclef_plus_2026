from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.audio import crop_or_pad
from src.metrics import sigmoid


def iter_soundscape_windows(path: str | Path):
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    n_windows = int(np.ceil(len(audio) / 160000))
    for idx in range(n_windows):
        start_seconds = idx * 5
        clip = crop_or_pad(audio, start_seconds=start_seconds)
        yield start_seconds + 5, clip


def make_submission(model, test_dir: str | Path, sample_submission_path: str | Path, output_path: str | Path, cfg: dict) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_path)
    labels = list(sample.columns[1:])
    rows = []
    test_dir = Path(test_dir)

    for audio_path in sorted(test_dir.glob("*.ogg")):
        for end_second, clip in iter_soundscape_windows(audio_path):
            logits = model.predict(clip[None, :], verbose=0)
            probs = sigmoid(logits)[0]
            row_id = f"{audio_path.stem}_{int(end_second)}"
            rows.append({"row_id": row_id, **dict(zip(labels, probs))})

    submission = pd.DataFrame(rows)
    if len(submission) == len(sample):
        submission = submission.set_index("row_id").reindex(sample["row_id"]).reset_index()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
