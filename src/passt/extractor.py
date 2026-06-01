from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io

import numpy as np


@dataclass
class PaSSTBatchOutput:
    embeddings: np.ndarray
    logits: np.ndarray | None = None


class PaSSTExtractor:
    """Thin runtime wrapper around hear21passt.

    The imports stay inside the class so TensorFlow-only runs do not require
    torch/PaSST to be installed.
    """

    def __init__(
        self,
        device: str = "auto",
        arch: str = "",
        include_logits: bool = False,
    ) -> None:
        import torch

        from hear21passt import base as passt_base

        self.torch = torch
        self.passt_base = passt_base
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.arch = arch
        self.include_logits = include_logits

        with contextlib.redirect_stdout(io.StringIO()):
            if arch:
                self.embedding_model = passt_base.get_basic_model(arch=arch)
            else:
                self.embedding_model = passt_base.load_model()
        self.embedding_model.eval().to(self.device)

        self.logit_model = None
        if include_logits:
            with contextlib.redirect_stdout(io.StringIO()):
                if arch:
                    self.logit_model = passt_base.get_basic_model(mode="logits", arch=arch)
                else:
                    self.logit_model = passt_base.load_model(mode="logits")
            self.logit_model.eval().to(self.device)

    def extract(self, waveforms: np.ndarray) -> PaSSTBatchOutput:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveform batch with shape (batch, samples), got {waveforms.shape}")

        with self.torch.no_grad():
            tensor = self.torch.from_numpy(waveforms).to(self.device)
            with contextlib.redirect_stdout(io.StringIO()):
                embeddings = self.passt_base.get_scene_embeddings(tensor, self.embedding_model)
            embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)

            logits_np = None
            if self.logit_model is not None:
                with contextlib.redirect_stdout(io.StringIO()):
                    logits = self.logit_model(tensor)
                logits_np = logits.detach().cpu().numpy().astype(np.float32)

        return PaSSTBatchOutput(embeddings=embeddings_np, logits=logits_np)
