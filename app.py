"""Speech-emotion dashboard (Streamlit). UI only - all model logic lives in src/.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (AUDIO_EXTENSIONS, CHUNKS_CSV, CORE_LABELS, EMOTION_COLORS,
                        LABELS, PREDICTIONS_CSV, RAW_AUDIO_DIR, TRAINED_LANGUAGES, WEIGHTS_PATH)

st.set_page_config(page_title="Speech Emotion Dashboard", page_icon="🎙️", layout="wide")

LOW_CONF = 0.5


# ----------------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------------
@st.cache_data
def load_csv(path: str, mtime: float) -> pd.DataFrame:      # mtime busts the cache when the file changes
    return pd.read_csv(path)


def read_if_exists(path):
    return load_csv(str(path), path.stat().st_mtime) if path.exists() else None


def add_language_fields(df):
    """Older CSVs have no language column; also mark languages absent from fine-tuning with a *."""
    if df is None:
        return None
    df = df.copy()
    if "language" not in df:
        df["language"] = "unknown"
    df["language"] = df["language"].fillna("unknown").astype(str).str.lower()
    df["language_label"] = [l.title() if l in TRAINED_LANGUAGES or l == "unknown" else l.title() + " *"
                            for l in df["language"]]
    return df


def rescore(df: pd.DataFrame, core_only: bool):
    """Recompute the winning emotion from the stored probabilities.

    core_only=True keeps just the 7 reliable emotions and renormalises, because the
    4 rare classes (calm/serious/excited/formal) had only ~20-35 training clips each.
    """
    labels = CORE_LABELS if core_only else LABELS
    P = df[[f"p_{l}" for l in labels]].to_numpy(dtype=float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    out = df.copy()
    out["emotion"] = np.array(labels)[P.argmax(1)]
    out["conf"] = P.max(1)
    return out, pd.DataFrame(P, columns=labels, index=df.index), labels


def color_map():
    return EMOTION_COLORS


def tidy(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend_title_text="", font=dict(size=13))
    return fig


def timeline_fig(starts, ends, probs: pd.DataFrame, silent):
    """Heatmap: emotion (rows) x 4-second window (columns)."""
    labels_x = [f"{s:.0f}-{e:.0f}s" + (" (silent)" if q else "") for s, e, q in zip(starts, ends, silent)]
    fig = px.imshow(probs.T.to_numpy(), x=labels_x, y=list(probs.columns), aspect="auto",
                    color_continuous_scale="Blues", zmin=0, zmax=1,
                    labels=dict(x="time window", y="", color="prob"))
    return tidy(fig, 300)


# ----------------------------------------------------------------------------
# live prediction (model is only imported/loaded when the second tab is used)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the fine-tuned model...")
def get_predictor():
    from src.model import EmotionPredictor
    return EmotionPredictor(WEIGHTS_PATH)


@st.cache_data(show_spinner="Analysing audio...")
def analyse_upload(file_bytes: bytes):
    return get_predictor().predict_file(io.BytesIO(file_bytes))


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
preds_raw = add_language_fields(read_if_exists(PREDICTIONS_CSV))
chunks_raw = read_if_exists(CHUNKS_CSV)

with st.sidebar:
    st.header("Filters")
    core_only = st.toggle("Core 7 emotions only", value=True,
                          help="Hides calm / serious / excited / formal, which the model saw very few "
                               "examples of in training and predicts unreliably.")
    if preds_raw is not None:
        topics = sorted(preds_raw["topic"].unique())
        sel_topics = st.multiselect("Topic", topics, default=topics)
        langs = sorted(preds_raw["language_label"].unique())
        sel_langs = st.multiselect("Language", langs, default=langs,
                                   help="* = language that was NOT in the fine-tuning data (zero-shot).")
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    with st.expander("About these predictions"):
        st.markdown(
            "- Model: IndicWav2Vec fine-tuned for emotion classification.\n"
            "- It was trained on **acted, studio-recorded** emotional speech, mostly Bengali and "
            "Malayalam. Real-world clips (debates, interviews, call-ins) are a different kind of "
            "audio, so treat results as **indicative, not ground truth**.\n"
            "- Long clips are scored in 4-second windows and averaged; silent windows are ignored.\n"
            "- Low confidence usually means the clip is unlike the training data."
        )

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
st.title("🎙️ Speech Emotion Dashboard")
st.caption("Emotion predictions from a fine-tuned IndicWav2Vec model on open-source education and election audio.")

tab_dash, tab_live = st.tabs(["Dashboard", "Try a clip"])

# ============================ DASHBOARD TAB ==================================
with tab_dash:
    if preds_raw is None or preds_raw.empty:
        st.info(
            "No predictions yet. Put clips in `data/raw_audio/<topic>/` (for example "
            "`education/` and `election/`), then run:\n\n"
            "```\npython -m scripts.batch_infer\n```\nand refresh this page."
        )
    else:
        scored, P, labels = rescore(preds_raw, core_only)
        keep = (scored["topic"].isin(sel_topics) & scored["language_label"].isin(sel_langs)
                & (scored["conf"] >= min_conf))
        view, P = scored[keep], P[keep]

        if view.empty:
            st.warning("No clips match the current filters.")
        else:
            # ---- KPIs
            top_emotion = view["emotion"].value_counts().idxmax()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clips analysed", f"{len(view):,}")
            c2.metric("Audio analysed", f"{view['duration_sec'].sum() / 60:.1f} min")
            c3.metric("Most common emotion", top_emotion.title())
            c4.metric("Low-confidence clips", f"{(view['conf'] < LOW_CONF).mean():.0%}",
                      help=f"Share of clips with confidence below {LOW_CONF:.0%}.")

            group = st.radio("Compare by", ["topic", "language_label"], horizontal=True,
                             format_func=lambda g: "Topic" if g == "topic" else "Language")
            gname = "topic" if group == "topic" else "language"
            if group != "topic" and view["language_label"].str.endswith("*").any():
                st.caption("* language not in the fine-tuning data - treat these results as lower reliability.")

            # ---- row 1: distribution + by group
            left, right = st.columns(2)
            with left:
                counts = view["emotion"].value_counts().rename_axis("emotion").reset_index(name="clips")
                fig = px.bar(counts, x="clips", y="emotion", color="emotion", orientation="h",
                             color_discrete_map=color_map(), title="Emotion distribution")
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
                st.plotly_chart(tidy(fig), width="stretch")
            with right:
                share = (view.groupby([group, "emotion"]).size().rename("n").reset_index())
                share["share"] = share["n"] / share.groupby(group)["n"].transform("sum")
                fig = px.bar(share, x=group, y="share", color="emotion", color_discrete_map=color_map(),
                             title=f"Emotion mix by {gname}", category_orders={"emotion": labels})
                fig.update_xaxes(title="")
                fig.update_yaxes(tickformat=".0%", title="")
                st.plotly_chart(tidy(fig), width="stretch")

            # ---- row 2: soft profile + confidence
            left, right = st.columns(2)
            with left:
                prof = P.groupby(view[group]).mean()
                fig = px.imshow(prof, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                                title=f"Average emotion probability by {gname}",
                                labels=dict(x="", y="", color="avg prob"))
                st.plotly_chart(tidy(fig), width="stretch")
            with right:
                fig = px.histogram(view, x="conf", color="emotion", nbins=20, color_discrete_map=color_map(),
                                   title="How confident is the model?", category_orders={"emotion": labels})
                fig.update_xaxes(title="confidence", range=[0, 1])
                fig.update_yaxes(title="clips")
                st.plotly_chart(tidy(fig), width="stretch")

            # ---- clip explorer
            st.subheader("Clip explorer")
            table = view[["topic", "language_label", "filename", "emotion", "conf", "duration_sec"]].sort_values("conf", ascending=False)
            st.dataframe(
                table, hide_index=True, width="stretch", height=260,
                column_config={
                    "language_label": st.column_config.TextColumn("language"),
                    "conf": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                    "duration_sec": st.column_config.NumberColumn("seconds", format="%.1f"),
                },
            )
            st.download_button("Download filtered results (CSV)", view.to_csv(index=False).encode(),
                               file_name="emotion_predictions_filtered.csv", mime="text/csv",
                               help="Same data can be loaded into Power BI / Tableau / Excel.")

            pick = st.selectbox("Inspect a clip", view["path"].tolist(),
                                format_func=lambda p: f"{view.loc[view['path'] == p, 'topic'].iat[0]} / {p.split('/')[-1]}")
            row = view[view["path"] == pick].iloc[0]
            a, b = st.columns([1, 2])
            with a:
                st.metric("Predicted emotion", row["emotion"].title(), f"{row['conf']:.0%} confidence", delta_color="off")
                audio_file = RAW_AUDIO_DIR / pick
                if audio_file.exists():
                    st.audio(str(audio_file))
                else:
                    st.caption("Audio file not found locally - only the predictions are available.")
            with b:
                if chunks_raw is not None:
                    ch = chunks_raw[chunks_raw["path"] == pick]
                    if not ch.empty:
                        ch_scored, ch_P, _ = rescore(ch, core_only)
                        st.plotly_chart(timeline_fig(ch["start_sec"], ch["end_sec"], ch_P, ch["silent"]),
                                        width="stretch")
                        st.caption("Each column is one 4-second window; darker = higher probability. "
                                   "The clip-level result above averages the non-silent windows.")

# ============================ LIVE TAB =======================================
with tab_live:
    st.write("Upload a short clip to classify it with the fine-tuned model.")
    if not WEIGHTS_PATH.exists():
        st.error(f"Model weights not found at `{WEIGHTS_PATH}`. Place the `.pt` file there to enable live prediction.")
    else:
        up = st.file_uploader("Audio file", type=sorted(e.lstrip(".") for e in AUDIO_EXTENSIONS))
        if up is not None:
            data = up.getvalue()
            st.audio(data)
            try:
                res = analyse_upload(data)
            except Exception as e:
                st.error(f"Could not read that file: {e}")
            else:
                labs = CORE_LABELS if core_only else LABELS
                p = res.probs[: len(labs)]
                p = p / p.sum()
                best = int(p.argmax())
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted emotion", labs[best].title())
                m2.metric("Confidence", f"{p[best]:.0%}")
                m3.metric("Length", f"{res.duration:.1f} s")
                if p[best] < LOW_CONF:
                    st.warning("Low confidence - this clip may be quite different from the training data.")

                dist = pd.DataFrame({"emotion": labs, "probability": p})
                fig = px.bar(dist, x="probability", y="emotion", color="emotion", orientation="h",
                             color_discrete_map=color_map(), title="Probability per emotion")
                fig.update_layout(yaxis=dict(categoryorder="total ascending"), showlegend=False)
                fig.update_xaxes(range=[0, 1])
                st.plotly_chart(tidy(fig), width="stretch")

                if len(res.starts) > 1:
                    cp = res.chunk_probs[:, : len(labs)]
                    cp = pd.DataFrame(cp / cp.sum(1, keepdims=True), columns=labs)
                    st.plotly_chart(timeline_fig(res.starts, res.ends, cp, res.silent), width="stretch")
