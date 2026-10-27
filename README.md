# Speech Emotion Dashboard

Fine-tuned IndicWav2Vec -> emotion predictions on education / election audio -> Streamlit dashboard.

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Copy `indicw2v_emotion_best_v3.pt` from Drive into `weights/`.

## 2. Add audio
Use `topic/language/clip` folders - the folder names become the topic and language tags:
```
data/raw_audio/education/hindi/clip1.wav
data/raw_audio/election/tamil/clip2.mp3
```
(A clip placed directly in a topic folder gets language "unknown".) The dashboard stars (*) any language
that was not in the fine-tuning data, since results there are zero-shot.
(.wav .mp3 .flac .ogg .m4a .opus)

## 3. Run the model on all clips
```bash
python -m scripts.batch_infer            # writes data/predictions.csv + data/chunk_predictions.csv
python -m scripts.batch_infer --resume   # later: only score new clips
```

## 4. Open the dashboard
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
- Label order in `src/config.py` is copied from the notebook's `label2id`. Don't reorder it.
- Long clips are scored in 4 s windows (training crop length) and averaged; near-silent windows are skipped.
- Windows are never zero-padded together: this model's group-norm makes padded predictions differ from unpadded ones.
- The 4 rare classes (calm, serious, excited, formal) had 18-35 training clips each. The dashboard hides them by default ("Core 7 emotions only").
- `predictions.csv` can be loaded straight into Power BI / Tableau if your mentor still wants that view.
