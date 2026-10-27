"""IndicWav2Vec emotion classifier: architecture, weight loading, prediction.

The architecture is a copy of `IndicW2VEmotion` from the Colab notebook. The
checkpoint (`state_dict`) already contains the fine-tuned encoder, so we only
need the *shape* of the encoder (config) - not the original fairseq file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .audio_utils import load_audio, split_into_chunks
from .config import CORE_LABELS, LABELS, NUM_LABELS, SAMPLE_RATE, WEIGHTS_PATH


def build_encoder_config() -> Wav2Vec2Config:
    """Same values the notebook derived from the fairseq IndicWav2Vec *base* config."""
    return Wav2Vec2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        feat_extract_norm="group",       # fairseq extractor_mode="default"
        do_stable_layer_norm=False,      # fairseq layer_norm_first=False
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        hidden_act="gelu",
        layerdrop=0.0,
        # Must stay > 0: the checkpoint contains `masked_spec_embed`, and HF only
        # creates that parameter when mask_time_prob > 0. It is inactive in eval().
        mask_time_prob=0.05,
        mask_time_length=10,
    )


class IndicW2VEmotion(nn.Module):
    def __init__(self, num_labels: int = NUM_LABELS, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = Wav2Vec2Model(build_encoder_config())
        H = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(H, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_values, lengths):
        h = self.encoder(input_values).last_hidden_state                       # (B, frames, 768)
        frame_lens = self.encoder._get_feat_extract_output_lengths(lengths.to(h.device))
        frame_lens = frame_lens.clamp(max=h.size(1))
        mask = (torch.arange(h.size(1), device=h.device)[None, :] < frame_lens[:, None]).unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)                  # mean over real frames
        return self.head(pooled)


# ---------------------------------------------------------------------------
@dataclass
class ClipResult:
    duration: float
    starts: list            # chunk start (s)
    ends: list              # chunk end (s)
    chunk_probs: np.ndarray  # (n_chunks, NUM_LABELS)
    silent: np.ndarray       # (n_chunks,) bool
    probs: np.ndarray        # (NUM_LABELS,) duration-weighted average over non-silent chunks

    def top(self, core_only: bool = False):
        return top_label(self.probs, core_only)


def top_label(probs: np.ndarray, core_only: bool = False):
    """Return (label, confidence). With core_only, restrict to the 7 reliable emotions
    and renormalise so confidences stay comparable."""
    p = np.asarray(probs, dtype=np.float64)
    if core_only:
        p = p[: len(CORE_LABELS)]
        p = p / max(p.sum(), 1e-9)
    i = int(p.argmax())
    return LABELS[i], float(p[i])


class EmotionPredictor:
    def __init__(self, weights_path=WEIGHTS_PATH, device: str | None = None, batch_size: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = IndicW2VEmotion()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)   # strict: fail loudly on any mismatch
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def _score_chunks(self, chunks) -> np.ndarray:
        """Softmax probabilities, one row per chunk, in the original order.

        Only chunks of EQUAL length are batched together, so nothing is ever
        zero-padded. This matters: IndicWav2Vec-base uses group-norm and no
        attention mask, so padding leaks into the result and would make a clip's
        prediction depend on which other chunks happened to share its batch.
        """
        probs = np.zeros((len(chunks), NUM_LABELS), dtype=np.float32)
        by_len: dict[int, list[int]] = {}
        for idx, c in enumerate(chunks):
            by_len.setdefault(len(c[2]), []).append(idx)

        for length, idxs in by_len.items():
            for i in range(0, len(idxs), self.batch_size):
                sel = idxs[i:i + self.batch_size]
                x = torch.from_numpy(np.stack([chunks[j][2] for j in sel])).to(self.device)
                lengths = torch.full((len(sel),), length)
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(x, lengths)
                else:
                    logits = self.model(x, lengths)
                probs[sel] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return probs

    def predict_waveform(self, wav: np.ndarray) -> ClipResult:
        chunks = split_into_chunks(wav)
        chunk_probs = self._score_chunks(chunks)
        silent = np.array([c[3] for c in chunks])
        weights = np.array([c[1] - c[0] for c in chunks], dtype=np.float64)
        use = ~silent if (~silent).any() else np.ones_like(silent)   # all silent -> keep all
        w = weights * use
        probs = (chunk_probs * w[:, None]).sum(0) / w.sum()
        return ClipResult(
            duration=len(wav) / SAMPLE_RATE,
            starts=[c[0] for c in chunks], ends=[c[1] for c in chunks],
            chunk_probs=chunk_probs, silent=silent, probs=probs,
        )

    def predict_file(self, path_or_file) -> ClipResult:
        return self.predict_waveform(load_audio(path_or_file))
