import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# AUTHOR CONFIG
# ------------------------------------------------------------------
AUTHOR_NAME = "Shreshtha Pandey"

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="IT Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# ------------------------------------------------------------------
# STYLING (Light & Dark Blue Gradient Palette)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: linear-gradient(180deg, #f0f6ff 0%, #e0effe 45%, #f8fbff 100%);
        color: #0f172a;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        border-radius: 20px;
        padding: 36px 42px;
        color: #ffffff;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    .hero h1 {
        color: #ffffff;
        margin: 0 0 10px 0;
        font-size: 2.3rem;
        font-weight: 800;
    }
    .hero p {
        color: #cbd5e1;
        margin: 0;
        font-size: 1.1rem;
        max-width: 780px;
        line-height: 1.6;
    }

    /* Info Cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #bfdbfe;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.05);
    }
    .info-card h4 {
        color: #1e3a8a;
        margin: 0 0 8px 0;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .info-card p {
        color: #475569;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        border-radius: 14px;
        padding: 20px 22px;
        color: #ffffff;
        text-align: left;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.15);
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .stat-card .label {
        font-size: 0.9rem;
        color: #bfdbfe;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: #ffffff;
        margin-top: 15px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.2);
    }
    .result-card .small {
        color: #bfdbfe;
        font-size: 1rem;
        font-weight: 500;
    }
    .result-card .big {
        font-size: 2.4rem;
        font-weight: 800;
        margin-top: 6px;
        text-transform: capitalize;
        color: #60a5fa;
    }

    h2, h3 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        box-shadow: 0 6px 15px rgba(15, 23, 42, 0.3);
    }
    div.stButton > button:focus-visible {
        outline: 3px solid #2563eb;
        outline-offset: 2px;
    }

    /* Footer */
    .footer {
        margin-top: 50px;
        padding: 20px;
        text-align: center;
        color: #ffffff;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        font-size: 1rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("ticket_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf


try:
    model, tfidf = load_artifacts()
    model_ok = True
except Exception:
    model_ok = False


# ------------------------------------------------------------------
# HEADER / HERO SECTION
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>IT Ticket Classifier</h1>
        <p>Intelligently categorizes incoming IT support requests to route them 
        to the right support team automatically, reducing resolution times.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="info-card"><h4>The Problem</h4>'
        "<p>Service desks handle thousands of daily tickets. Manual triage is slow "
        "and prone to misrouting, causing major resolution delays.</p></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="info-card"><h4>The Solution</h4>'
        "<p>An NLP-powered machine learning classifier that accurately analyzes "
        "ticket descriptions and assigns categories instantly.</p></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="info-card"><h4>How It Works</h4>'
        "<p>Text features are extracted using TF-IDF and classified using a high-performance "
        "Linear SVM trained on 47,800+ tickets.</p></div>",
        unsafe_allow_html=True,
    )

st.write("")

# ------------------------------------------------------------------
# NAVIGATION TABS
# ------------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "predict"

b1, b2 = st.columns(2)
with b1:
    if st.button("🎫 Predict a Ticket"):
        st.session_state.view = "predict"
with b2:
    if st.button("📊 Model Statistics"):
        st.session_state.view = "stats"

st.write("")


# ------------------------------------------------------------------
# STATS VIEW
# ------------------------------------------------------------------
def stat_card(value, label):
    return (
        f'<div class="stat-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>'
    )


def show_stats():
    st.subheader("Model Performance")

    cols = st.columns(4)
    cards = [
        ("87.0%", "Accuracy (Linear SVM)"),
        ("0.869", "Macro F1 Score"),
        ("0.870", "Weighted F1 Score"),
        ("8", "Ticket Categories"),
    ]
    for col, (v, l) in zip(cols, cards):
        col.markdown(stat_card(v, l), unsafe_allow_html=True)

    st.write("")
    cols = st.columns(4)
    cards = [
        ("47,823", "Cleaned Tickets"),
        ("38,258 / 9,565", "Train / Test Split"),
        ("20,000", "TF-IDF Features"),
        ("43.6", "Avg. Words / Ticket"),
    ]
    for col, (v, l) in zip(cols, cards):
        col.markdown(stat_card(v, l), unsafe_allow_html=True)

    st.write("")
    st.subheader("Model Comparison")
    comp = pd.DataFrame(
        {
            "Model": ["Linear SVM", "Logistic Regression", "Naive Bayes"],
            "Accuracy": [0.870, 0.864, 0.780],
            "Precision (macro)": [0.867, 0.846, 0.879],
            "Recall (macro)": [0.871, 0.884, 0.680],
            "Weighted F1": [0.870, 0.865, 0.774],
            "Macro F1": [0.869, 0.862, 0.733],
        }
    )
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(comp, hide_index=True, use_container_width=True)
    with right:
        st.bar_chart(comp.set_index("Model")[["Macro F1"]], color="#2563eb")
    st.caption(
        "Linear SVM delivers the best overall balance of precision and recall. "
        "Naive Bayes exhibits high precision but suffers from low recall on smaller classes."
    )

    st.subheader("Per-Category Results (Linear SVM)")
    per_class = pd.DataFrame(
        {
            "Category": [
                "access", "administrative rights", "hardware", "hr support",
                "internal project", "miscellaneous", "purchase", "storage",
            ],
            "Precision": [0.891, 0.753, 0.871, 0.881, 0.867, 0.824, 0.947, 0.904],
            "Recall": [0.902, 0.812, 0.855, 0.876, 0.873, 0.843, 0.913, 0.897],
            "F1": [0.897, 0.781, 0.862, 0.878, 0.870, 0.833, 0.930, 0.901],
            "Test tickets": [1425, 352, 2722, 2182, 424, 1412, 493, 555],
        }
    )
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(per_class, hide_index=True, use_container_width=True)
    with right:
        st.bar_chart(per_class.set_index("Category")[["F1"]], color="#1e3a8a")

    st.subheader("Class Distribution")
    dist = pd.DataFrame(
        {
            "Category": [
                "hardware", "hr support", "access", "miscellaneous",
                "storage", "purchase", "internal project", "administrative rights",
            ],
            "Share of Data (%)": [28.5, 22.8, 14.9, 14.8, 5.8, 5.2, 4.4, 3.7],
        }
    )
    st.bar_chart(dist.set_index("Category"), color="#2563eb")
    st.caption(
        "Dataset classes show noticeable imbalance (~7.7x scale difference between largest and smallest classes), "
        "making macro F1 and balanced weighting vital."
    )

    st.subheader("Common Misclassifications")
    errors = pd.DataFrame(
        {
            "Actual": [
                "hardware", "hr support", "hardware", "hr support",
                "miscellaneous", "hardware", "miscellaneous", "hardware",
            ],
            "Predicted": [
                "hr support", "hardware", "miscellaneous", "miscellaneous",
                "hardware", "access", "hr support", "administrative rights",
            ],
            "Tickets": [113, 107, 100, 82, 79, 75, 66, 58],
        }
    )
    st.dataframe(errors, hide_index=True, use_container_width=True)
    st.caption(
        "Most confusion happens between hardware, HR support, and miscellaneous tickets due to overlapping technical vocabulary."
    )

    if os.path.exists("confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("confusion_matrix.png", use_container_width=True)

    st.subheader("Performance & Footprint")
    cols = st.columns(3)
    cols[0].markdown(stat_card("0.021 sec", "Inference time (9,565 test tickets)"), unsafe_allow_html=True)
    cols[1].markdown(stat_card("1.28 MB", "Saved Model Size"), unsafe_allow_html=True)
    cols[2].markdown(stat_card("0.82 MB", "Saved TF-IDF Size"), unsafe_allow_html=True)


# ------------------------------------------------------------------
# PREDICT VIEW
# ------------------------------------------------------------------
SAMPLES = {
    "Choose a sample ticket": "",
    "Purchase order (from test data)": (
        "new purchase po thursday march pm purchase po dear purchased monitor "
        "please log installation please take consideration mandatory receipts "
        "section order"
    ),
    "Purchase order, retina laptop (from test data)": (
        "new purchase po friday purchase po dear purchased pro retina intel "
        "please log installation please take consideration mandatory receipts "
        "section order"
    ),
}


def show_predict():
    st.subheader("Predict Support Ticket Category")
    st.write(
        "Enter or paste a ticket description below to automatically classify it into the correct IT support group."
    )

    sample = st.selectbox("Try a sample ticket", list(SAMPLES.keys()))
    text = st.text_area(
        "Ticket description",
        value=SAMPLES[sample],
        height=160,
        placeholder="Paste your IT support ticket details here...",
    )

    if st.button("🚀 Classify Ticket", key="predict_btn"):
        if not model_ok:
            st.error(
                "Model files missing! Please ensure 'ticket_model.pkl' and 'tfidf.pkl' "
                "are placed in the same directory as app.py."
            )
            return
        if len(text.split()) < 3:
            st.warning("Please enter at least 3 words for accurate classification.")
            return

        vec = tfidf.transform([text.lower()])
        pred = model.predict(vec)[0]

        st.markdown(
            f'<div class="result-card"><div class="small">Predicted Category</div>'
            f'<div class="big">🖥️ {pred}</div></div>',
            unsafe_allow_html=True,
        )

        if hasattr(model, "decision_function"):
            scores = model.decision_function(vec)[0]
            exp = np.exp(scores - scores.max())
            rel = exp / exp.sum()
            top = np.argsort(rel)[::-1][:3]
            st.write("")
            st.markdown("**Top 3 Category Predictions**")
            for i in top:
                st.write(f"**{model.classes_[i]}**")
                st.progress(float(rel[i]))
            st.caption(
                "Progress bars represent relative decision confidence scores. "
                "Extremely brief or vague queries may yield lower confidence."
            )


# ------------------------------------------------------------------
# RENDER VIEW
# ------------------------------------------------------------------
if st.session_state.view == "stats":
    show_stats()
else:
    show_predict()

# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown(
    f'<div class="footer">Built with ❤️ by {AUTHOR_NAME} • IT Ticket AI Classification System</div>',
    unsafe_allow_html=True,
)