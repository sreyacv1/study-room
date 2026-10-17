"""NLP Text Classifier: a Fake News Detector and a Sentiment Analyzer in one Streamlit app.

Run with:  streamlit run app.py

Both models are TF-IDF + Logistic Regression pipelines. On first launch, any model
missing from models/ is trained from a public dataset (downloaded once) and saved,
so later launches load it in about a second.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(__file__).parent / "models"

# Each model: where its data comes from, how to turn that data into (text, label),
# and how to present the two classes.
MODELS = {
    "Fake News Detector": {
        "file": "fake_news.joblib",
        "source": "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
        "dataset": "fake_or_real_news (6,335 news articles, labelled FAKE / REAL)",
        "labels": {0: "Fake", 1: "Real"},
        "icons": {0: "🚩", 1: "✅"},
        "prompt": "Paste a news headline and/or article text",
        "example": (
            "The Senate passed the bipartisan infrastructure bill on Tuesday by a vote "
            "of 69 to 30, sending the measure to the House, where its fate is tied to "
            "a larger budget package Democrats hope to pass this fall."
        ),
    },
    "Sentiment Analyzer": {
        "file": "sentiment.joblib",
        "source": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
        "dataset": "IMDB movie reviews (25,000 reviews, labelled negative / positive)",
        "labels": {0: "Negative", 1: "Positive"},
        "icons": {0: "👎", 1: "👍"},
        "prompt": "Type or paste a review, comment or any opinionated text",
        "example": (
            "I went in with low expectations, but the acting was superb and the story "
            "kept me hooked until the very last scene. Easily one of the best films I "
            "have seen this year."
        ),
    },
}


def load_dataset(name: str) -> tuple[pd.Series, pd.Series]:
    """Download the training data for a model and return (texts, 0/1 labels)."""
    cfg = MODELS[name]
    if name == "Fake News Detector":
        df = pd.read_csv(cfg["source"])
        texts = df["title"].fillna("") + " " + df["text"].fillna("")
        labels = (df["label"] == "REAL").astype(int)
    else:
        df = pd.read_parquet(cfg["source"])
        texts, labels = df["text"], df["label"]
    return texts, labels


def train(name: str) -> dict:
    """Fit a TF-IDF + Logistic Regression pipeline and report held-out accuracy."""
    texts, labels = load_dataset(name)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english" if name == "Fake News Detector" else None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=100_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=4.0, max_iter=1000)),
    ])
    pipeline.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    return {"pipeline": pipeline, "accuracy": accuracy, "n_train": len(x_train)}


@st.cache_resource(show_spinner=False)
def get_model(name: str) -> dict:
    """Load a saved model, training and saving it first if it doesn't exist yet."""
    path = MODEL_DIR / MODELS[name]["file"]
    if path.exists():
        return joblib.load(path)
    with st.spinner(f"First run: downloading data and training the {name} (about a minute)…"):
        bundle = train(name)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return bundle


def top_words(pipeline: Pipeline, text: str, k: int = 8) -> pd.DataFrame:
    """The words in this text that pushed the prediction hardest in each direction."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    vec = tfidf.transform([text])
    vocab = tfidf.get_feature_names_out()
    contrib = vec.multiply(clf.coef_[0]).tocsr()
    rows = sorted(zip(contrib.indices, contrib.data), key=lambda t: abs(t[1]), reverse=True)
    return pd.DataFrame(
        [(vocab[i], w) for i, w in rows[:k]], columns=["term", "weight"]
    )


# ---------------------------------------------------------------- UI

st.set_page_config(page_title="NLP Text Classifier", page_icon="🧠", layout="centered")

st.sidebar.title("🧠 NLP Text Classifier")
choice = st.sidebar.radio("Choose a model", list(MODELS), key="model")
cfg = MODELS[choice]

try:
    model = get_model(choice)
except Exception as exc:  # usually no internet on first run
    st.error(
        f"Couldn't load or train the {choice}: {exc}\n\n"
        "The first run needs an internet connection to download the training data."
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** TF-IDF (1–2 grams) + Logistic Regression  \n"
    f"**Trained on:** {cfg['dataset']}  \n"
    f"**Held-out accuracy:** {model['accuracy']:.1%}"
)
st.sidebar.caption(
    "Predictions reflect patterns in the training data, not fact-checking. "
    "Treat the fake-news score as a style signal, not a verdict."
)

st.title(f"{cfg['icons'][1]} {choice}")
st.write(f"Classifies text as **{cfg['labels'][1]}** or **{cfg['labels'][0]}**.")

text_key = f"text_{choice}"
if st.button("Try an example"):
    st.session_state[text_key] = cfg["example"]
text = st.text_area(cfg["prompt"], height=200, key=text_key, placeholder="Start typing…")

if text.strip():
    pipeline = model["pipeline"]
    probs = pipeline.predict_proba([text])[0]
    pred = int(probs.argmax())
    confidence = float(probs[pred])
    label = cfg["labels"][pred]

    col1, col2 = st.columns(2)
    col1.metric("Prediction", f"{cfg['icons'][pred]} {label}")
    col2.metric("Confidence", f"{confidence:.1%}")
    st.progress(confidence)

    if confidence < 0.6:
        st.info("Low confidence: the model sees this text as borderline.")
    if len(text.split()) < 8:
        st.caption("Very short text gives the model little to go on; longer input is more reliable.")

    with st.expander("Class probabilities and the words that drove this prediction"):
        st.bar_chart(pd.DataFrame(
            {"probability": probs}, index=[cfg["labels"][0], cfg["labels"][1]]
        ))
        words = top_words(pipeline, text)
        if words.empty:
            st.write("None of these words appear in the model's vocabulary.")
        else:
            words["pushes toward"] = words["weight"].map(
                lambda w: cfg["labels"][1] if w > 0 else cfg["labels"][0]
            )
            st.dataframe(words[["term", "pushes toward", "weight"]],
                         hide_index=True, use_container_width=True)
else:
    st.caption("The prediction appears here as soon as you enter text.")
