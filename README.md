# Speech Emotion Dashboard

Fine-tuned IndicWav2Vec -> emotion predictions on indicVoice dataset(100 clips) -> Streamlit dashboard.

##  Run the model on all clips
```bash
python -m scripts.batch_infer            # writes data/predictions.csv + data/chunk_predictions.csv
python -m scripts.batch_infer --resume   # later: only score new clips
```

##  Open the dashboard
```bash
streamlit run app.py
```
"Try a clip" tab = upload one file and classify it live (loads the model on first use).

## Layout
| Path | Role |
|---|---|
| `app.py` | Streamlit UI only |
| `src/model.py` | model class, weight loading, prediction |
| `src/audio_utils.py` | load to 16 kHz mono, cut into 4 s windows |
| `src/config.py` | label order, paths, constants (must match training) |
| `scripts/batch_infer.py` | folder of clips -> CSVs |

## Things to know
- Long clips are scored in 4 s windows (training crop length) and averaged; near-silent windows are skipped.
- Windows are never zero-padded together: this model's group-norm makes padded predictions differ from unpadded ones.
- The 4 rare classes (calm, serious, excited, formal) had 18-35 training clips each.
