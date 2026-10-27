"""Run the fine-tuned model over every clip in data/raw_audio and write the CSVs
that the dashboard reads.

Folder layout decides the topic and language tags:
    data/raw_audio/education/hindi/clip1.wav   -> topic = "education", language = "hindi"
    data/raw_audio/election/tamil/clip2.mp3    -> topic = "election",  language = "tamil"
    data/raw_audio/education/clip3.wav         -> topic = "education", language = "unknown"

Usage (from the project root):
    python -m scripts.batch_infer
    python -m scripts.batch_infer --audio-dir data/raw_audio --weights weights/indicw2v_emotion_best_v3.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (AUDIO_EXTENSIONS, CHUNKS_CSV, LABELS, PREDICTIONS_CSV,
                        RAW_AUDIO_DIR, WEIGHTS_PATH)
from src.model import EmotionPredictor


def find_clips(audio_dir: Path):
    for p in sorted(audio_dir.rglob("*")):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            parts = p.relative_to(audio_dir).parts
            topic = parts[0] if len(parts) > 1 else "unlabelled"
            language = parts[1].lower() if len(parts) > 2 else "unknown"
            yield p, topic, language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=RAW_AUDIO_DIR)
    ap.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--out", type=Path, default=PREDICTIONS_CSV)
    ap.add_argument("--chunks-out", type=Path, default=CHUNKS_CSV)
    ap.add_argument("--resume", action="store_true", help="skip clips already in the output CSV")
    args = ap.parse_args()

    clips = list(find_clips(args.audio_dir))
    if not clips:
        sys.exit(f"No audio found under {args.audio_dir}. Put clips in data/raw_audio/<topic>/")
    if not args.weights.exists():
        sys.exit(f"Weights not found: {args.weights}")

    clip_rows, chunk_rows, done = [], [], set()
    if args.resume and args.out.exists():
        old, old_chunks = pd.read_csv(args.out), pd.read_csv(args.chunks_out)
        clip_rows, chunk_rows = old.to_dict("records"), old_chunks.to_dict("records")
        done = set(old["path"])

    print(f"Loading model on first use... ({len(clips)} clips found)")
    predictor = EmotionPredictor(args.weights)
    print(f"Device: {predictor.device}")

    failed, t0 = [], time.time()
    for n, (path, topic, language) in enumerate(clips, 1):
        rel = path.relative_to(args.audio_dir).as_posix()
        if rel in done:
            continue
        try:
            res = predictor.predict_file(path)
        except Exception as e:                      # unreadable / corrupt file
            failed.append((rel, str(e)[:120]))
            print(f"[{n}/{len(clips)}] FAILED {rel}: {e}")
            continue

        label, conf = res.top(core_only=False)
        row = {
            "path": rel, "filename": path.name, "topic": topic, "language": language,
            "duration_sec": round(res.duration, 2), "n_chunks": len(res.starts),
            "predicted_emotion": label, "confidence": round(conf, 4),
        }
        row.update({f"p_{l}": round(float(p), 5) for l, p in zip(LABELS, res.probs)})
        clip_rows.append(row)

        for k, (s, e) in enumerate(zip(res.starts, res.ends)):
            crow = {"path": rel, "chunk": k, "start_sec": round(s, 2), "end_sec": round(e, 2),
                    "silent": bool(res.silent[k])}
            crow.update({f"p_{l}": round(float(p), 5) for l, p in zip(LABELS, res.chunk_probs[k])})
            chunk_rows.append(crow)

        print(f"[{n}/{len(clips)}] {rel:50s} {label:10s} {conf:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clip_rows).to_csv(args.out, index=False)
    pd.DataFrame(chunk_rows).to_csv(args.chunks_out, index=False)
    print(f"\nDone in {time.time() - t0:.0f}s -> {args.out}  ({len(clip_rows)} clips, {len(failed)} failed)")
    for rel, err in failed:
        print("  failed:", rel, "-", err)


if __name__ == "__main__":
    main()
