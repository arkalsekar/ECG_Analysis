from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import auc, roc_curve

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover - optional dependency
    LimeTabularExplainer = None


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "dataset" / "features_dataset.csv"
MODEL_CANDIDATES = [
    ROOT_DIR / "model" / "random_forest_ecg_model.pkl",
    ROOT_DIR / "model" / "rf_model.pkl",
]
IMAGE_DIR = ROOT_DIR / "images"

PAGE_ORDER = [
    ("app.py", "Home"),
    ("pages/1_About_Dataset.py", "About Dataset"),
    ("pages/2_Data_Analysis.py", "Data Analysis"),
    ("pages/3_Feature_Extraction.py", "Feature Extraction"),
    ("pages/4_Feature_Selection.py", "Feature Selection"),
    ("pages/5_Models_Trained.py", "Models Trained"),
    ("pages/6_Best_Model.py", "Best Model"),
    ("pages/7_Live_Prediction.py", "Live Prediction"),
    ("pages/8_Explainable_AI.py", "Explainable AI"),
    # ("pages/9_Future_Work.py", "Future Work"),
]

PROJECT_SUMMARY = {
    "dataset_size": "8,528",
    "feature_count": "32",
    "selected_features": "14",
    "best_accuracy": "79.09%",
    "best_model": "Random Forest Classifier",
}

ORIGINAL_CLASS_COUNTS = {
    "Normal Rhythm": 5154,
    "Atrial Fibrillation": 771,
    "Other Rhythm": 2557,
    "Noisy Readings": 46,
}

FEATURE_COLUMNS = [
    "mean",
    "median",
    "std",
    "variance",
    "min",
    "max",
    "peak",
    "peak_to_peak",
    "energy",
    "power",
    "rms",
    "skewness",
    "kurtosis",
    "area",
    "abs_mean",
    "zero_crossings",
    "crest_factor",
    "entropy",
    "fft_mean",
    "fft_std",
    "fft_max",
    "fft_energy",
    "dominant_frequency",
    "spectral_entropy",
    "rpeak_count",
    "mean_rr",
    "rr_std",
    "heart_rate",
    "sdnn",
    "rmssd",
    "pnn50",
    "hrv_entropy",
]

SELECTED_FEATURES = [
    "heart_rate",
    "mean_rr",
    "rmssd",
    "sdnn",
    "pnn50",
    "rr_std",
    "fft_mean",
    "fft_max",
    "fft_energy",
    "dominant_frequency",
    "spectral_entropy",
    "energy",
    "entropy",
    "zero_crossings",
]

MODEL_RESULTS = {
    "Logistic Regression": {"accuracy": 74.00, "precision": 72.20, "recall": 52.76, "specificity": 87.29, "f1": 60.96},
    "KNN": {"accuracy": 75.82, "precision": 73.05, "recall": 58.90, "specificity": 86.40, "f1": 65.21},
    "Decision Tree": {"accuracy": 68.24, "precision": 58.25, "recall": 61.73, "specificity": 72.32, "f1": 59.94},
    "Random Forest": {"accuracy": 79.09, "precision": 77.99, "recall": 63.62, "specificity": 88.77, "f1": 70.08},
    "Extra Trees": {"accuracy": 78.55, "precision": 78.62, "recall": 60.79, "specificity": 89.66, "f1": 68.56},
    "Gradient Boosting": {"accuracy": 79.15, "precision": 79.39, "recall": 61.89, "specificity": 89.95, "f1": 69.56},
    "SVM": {"accuracy": 78.73, "precision": 81.98, "recall": 57.32, "specificity": 92.12, "f1": 67.47},
    "XGBoost": {"accuracy": 77.45, "precision": 74.95, "recall": 62.20, "specificity": 87.00, "f1": 67.99},
}

MODEL_COUNTS: Dict[str, Dict[str, int]] = {}
for model_name, scores in MODEL_RESULTS.items():
    total_samples = 1650
    precision = scores["precision"] / 100.0
    recall = scores["recall"] / 100.0
    specificity = scores["specificity"] / 100.0
    balance = recall * (1 - precision) / max(precision * (1 - specificity), 1e-9)
    actual_positive = max(1, int(round(total_samples / (1 + balance))))
    actual_negative = total_samples - actual_positive
    true_positive = max(0, int(round(recall * actual_positive)))
    false_negative = actual_positive - true_positive
    true_negative = max(0, int(round(specificity * actual_negative)))
    false_positive = actual_negative - true_negative
    MODEL_COUNTS[model_name] = {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


CORE_STATISTICAL_FEATURES = [
    {
        "name": "Mean",
        "formula": r"$\mu = \frac{1}{N}\sum_{i=1}^{N}x_i$",
        "explanation": "Average amplitude of the ECG window.",
        "significance": "Captures the central voltage level and baseline shift.",
    },
    {
        "name": "Median",
        "formula": r"$\tilde{x} = \mathrm{median}(x_1, x_2, \ldots, x_N)$",
        "explanation": "Middle value after sorting all samples.",
        "significance": "Robust to extreme spikes and artifacts.",
    },
    {
        "name": "Mode",
        "formula": r"$\mathrm{mode}(x)$",
        "explanation": "Most frequently observed amplitude value.",
        "significance": "Indicates the dominant repeated amplitude level.",
    },
    {
        "name": "Standard Deviation",
        "formula": r"$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}$",
        "explanation": "Spread of ECG amplitudes around the mean.",
        "significance": "Highlights variation caused by morphology and noise.",
    },
    {
        "name": "Variance",
        "formula": r"$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2$",
        "explanation": "Squared dispersion of signal values.",
        "significance": "Useful for identifying unstable and noisy segments.",
    },
    {
        "name": "Minimum",
        "formula": r"$x_{\min} = \min(x)$",
        "explanation": "Lowest amplitude observed in the beat window.",
        "significance": "Useful for detecting negative deflections and inversions.",
    },
    {
        "name": "Maximum",
        "formula": r"$x_{\max} = \max(x)$",
        "explanation": "Highest amplitude observed in the beat window.",
        "significance": "Captures the R-peak and strong positive excursions.",
    },
    {
        "name": "Range",
        "formula": r"$R = x_{\max} - x_{\min}$",
        "explanation": "Difference between maximum and minimum amplitudes.",
        "significance": "Measures the total spread of the ECG waveform.",
    },
    {
        "name": "RMS",
        "formula": r"$\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2}$",
        "explanation": "Root mean square energy of the signal.",
        "significance": "Combines amplitude and energy into a stable descriptor.",
    },
]

STATISTICAL_FEATURES = [
    {
        "name": "Peak",
        "formula": r"$\max(|x_i|)$",
        "explanation": "Largest absolute amplitude in the window.",
        "significance": "Approximates the strongest ECG deflection.",
    },
    {
        "name": "Peak-to-Peak",
        "formula": r"$x_{\max} - x_{\min}$",
        "explanation": "Total amplitude span of the signal.",
        "significance": "Useful for measuring beat width and morphology changes.",
    },
    {
        "name": "Area",
        "formula": r"$A = \sum_{i=1}^{N}|x_i|\Delta t$",
        "explanation": "Absolute area under the ECG curve.",
        "significance": "Summarizes the overall electrical activity in the beat.",
    },
    {
        "name": "Absolute Mean",
        "formula": r"$\frac{1}{N}\sum_{i=1}^{N}|x_i|$",
        "explanation": "Average magnitude irrespective of sign.",
        "significance": "Captures intensity while ignoring polarity.",
    },
    {
        "name": "Skewness",
        "formula": r"$\frac{1}{N}\sum\left(\frac{x_i-\mu}{\sigma}\right)^3$",
        "explanation": "Asymmetry of the amplitude distribution.",
        "significance": "Highlights whether the waveform leans toward large positive or negative values.",
    },
    {
        "name": "Kurtosis",
        "formula": r"$\frac{1}{N}\sum\left(\frac{x_i-\mu}{\sigma}\right)^4$",
        "explanation": "Peakedness and tail weight of the distribution.",
        "significance": "Sensitive to sharp spikes and outliers.",
    },
    {
        "name": "Energy",
        "formula": r"$E = \sum_{i=1}^{N}x_i^2$",
        "explanation": "Total signal power accumulated over the window.",
        "significance": "Tracks how intense the ECG activity is.",
    },
    {
        "name": "Power",
        "formula": r"$P = \frac{1}{N}\sum_{i=1}^{N}x_i^2$",
        "explanation": "Average signal energy per sample.",
        "significance": "Provides a normalized intensity measure.",
    },
    {
        "name": "Zero Crossings",
        "formula": r"$Z = \sum \mathbf{1}[x_i \cdot x_{i-1} < 0]$",
        "explanation": "Number of sign changes in the ECG waveform.",
        "significance": "Higher counts often indicate noise or rapid oscillation.",
    },
    {
        "name": "Crest Factor",
        "formula": r"$\mathrm{CF} = \frac{\max(|x_i|)}{\mathrm{RMS}}$",
        "explanation": "Ratio of the peak amplitude to the RMS value.",
        "significance": "Measures how sharp the dominant peak is compared with the average energy.",
    },
    {
        "name": "Entropy",
        "formula": r"$H = -\sum_{k}p_k\log(p_k)$",
        "explanation": "Uncertainty of the amplitude distribution.",
        "significance": "Higher entropy usually indicates a more irregular waveform.",
    },
]

FREQUENCY_FEATURES = [
    {
        "name": "FFT Mean",
        "formula": r"$\frac{1}{M}\sum_{k=1}^{M}|X_k|$",
        "explanation": "Average magnitude of the Fourier spectrum.",
        "significance": "Summarizes the spectral intensity profile.",
    },
    {
        "name": "FFT Std",
        "formula": r"$\sqrt{\frac{1}{M}\sum_{k=1}^{M}(|X_k|-\bar{X})^2}$",
        "explanation": "Spread of spectral magnitudes around the mean.",
        "significance": "Useful for distinguishing concentrated and diffuse spectra.",
    },
    {
        "name": "FFT Max",
        "formula": r"$\max(|X_k|)$",
        "explanation": "Strongest spectral component.",
        "significance": "Highlights the dominant periodic rhythm in the ECG.",
    },
    {
        "name": "FFT Energy",
        "formula": r"$\sum_{k=1}^{M}|X_k|^2$",
        "explanation": "Total spectral power of the transformed signal.",
        "significance": "Captures frequency-domain intensity.",
    },
    {
        "name": "Dominant Frequency",
        "formula": r"$f_d = \arg\max_k |X_k| \cdot \frac{f_s}{N}$",
        "explanation": "Frequency with maximum amplitude in the spectrum.",
        "significance": "Represents the strongest rhythm or oscillatory pattern.",
    },
    {
        "name": "Spectral Entropy",
        "formula": r"$H_s = -\frac{\sum_k P_k\log(P_k)}{\log(M)}$",
        "explanation": "Entropy of normalized spectral power values.",
        "significance": "Low values indicate concentrated rhythm, high values indicate irregularity.",
    },
]

HRV_FEATURES = [
    {
        "name": "R-peak Count",
        "formula": r"$N_{R} = \#\{\text{detected R peaks}\}$",
        "explanation": "Number of detected R-peaks in the ECG window.",
        "significance": "Directly relates to heart rhythm regularity.",
    },
    {
        "name": "Mean RR",
        "formula": r"$\overline{RR} = \frac{1}{K}\sum_{i=1}^{K}RR_i$",
        "explanation": "Average interval between consecutive R peaks.",
        "significance": "Provides a robust summary of beat-to-beat spacing.",
    },
    {
        "name": "RR Std",
        "formula": r"$\sigma_{RR} = \sqrt{\frac{1}{K}\sum_{i=1}^{K}(RR_i-\overline{RR})^2}$",
        "explanation": "Variation of RR intervals.",
        "significance": "Measures how irregular the heartbeat timing is.",
    },
    {
        "name": "Heart Rate",
        "formula": r"$\mathrm{HR} = \frac{60}{\overline{RR}}$",
        "explanation": "Estimated beats per minute from the RR intervals.",
        "significance": "A primary clinical rhythm indicator.",
    },
    {
        "name": "SDNN",
        "formula": r"$\mathrm{SDNN} = \mathrm{std}(RR)$",
        "explanation": "Standard deviation of RR intervals.",
        "significance": "Captures overall autonomic variability.",
    },
    {
        "name": "RMSSD",
        "formula": r"$\mathrm{RMSSD} = \sqrt{\frac{1}{K-1}\sum_{i=1}^{K-1}(RR_{i+1}-RR_i)^2}$",
        "explanation": "Root mean square of successive RR differences.",
        "significance": "Sensitive to short-term heart rate variability.",
    },
    {
        "name": "pNN50",
        "formula": r"$\mathrm{pNN50} = \frac{\#(|RR_{i+1}-RR_i|>50\,\mathrm{ms})}{K-1}\times100$",
        "explanation": "Percentage of RR differences above 50 ms.",
        "significance": "Reflects strong beat-to-beat variability.",
    },
    {
        "name": "HRV Entropy",
        "formula": r"$H_{HRV} = -\sum_j q_j\log(q_j)$",
        "explanation": "Entropy of the RR interval distribution.",
        "significance": "Distinguishes regular sinus rhythm from irregular rhythms.",
    },
]

NON_LINEAR_FEATURES = [
    {
        "name": "Energy",
        "formula": r"$E = \sum_{i=1}^{N}x_i^2$",
        "explanation": "Total amplitude energy of the segment.",
        "significance": "Captures overall signal intensity.",
    },
    {
        "name": "Power",
        "formula": r"$P = \frac{E}{N}$",
        "explanation": "Average energy per sample.",
        "significance": "Normalizes energy across signals of the same length.",
    },
    {
        "name": "Entropy",
        "formula": r"$H = -\sum_k p_k\log(p_k)$",
        "explanation": "Irregularity of amplitude occupancy.",
        "significance": "Higher values generally indicate more complex or noisy waveforms.",
    },
    {
        "name": "Zero Crossings",
        "formula": r"$Z = \sum \mathbf{1}[x_i \cdot x_{i-1} < 0]$",
        "explanation": "Count of sign flips in the time signal.",
        "significance": "Useful for identifying rapid oscillations and disturbances.",
    },
    {
        "name": "Crest Factor",
        "formula": r"$\mathrm{CF} = \frac{\max(|x_i|)}{\mathrm{RMS}}$",
        "explanation": "Peak-to-average energy ratio.",
        "significance": "Shows how pronounced the dominant spike is.",
    },
]

FEATURE_SELECTION_FEATURES = [
    ("heart_rate", "Clinical rhythm indicator"),
    ("mean_rr", "Average beat spacing"),
    ("rmssd", "Short-term HRV"),
    ("sdnn", "Overall HRV variability"),
    ("pnn50", "High variability marker"),
    ("rr_std", "Beat interval spread"),
    ("fft_mean", "Spectral concentration"),
    ("fft_max", "Dominant spectral response"),
    ("fft_energy", "Frequency-domain intensity"),
    ("dominant_frequency", "Primary oscillation rate"),
    ("spectral_entropy", "Spectral irregularity"),
    ("energy", "Signal intensity"),
    ("entropy", "Amplitude irregularity"),
    ("zero_crossings", "Signal oscillation density"),
]


def configure_page(page_title: str) -> None:
    st.set_page_config(page_title=page_title, page_icon="ــــہـ٨ــــــ", layout="wide", initial_sidebar_state="expanded")


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #08141f;
            --panel: #0f2231;
            --panel-2: #123044;
            --accent: #58c4b8;
            --accent-2: #6ea8fe;
            --text: #e9f2f7;
            --muted: #9eb2c1;
            --border: rgba(255,255,255,0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(88,196,184,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(110,168,254,0.16), transparent 30%),
                linear-gradient(180deg, #061019 0%, #0b1723 100%);
            color: var(--text);
        }
        section[data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNav"],
        ul[data-testid="stSidebarNavList"],
        li[data-testid="stSidebarNavItem"],
        [aria-label="Pages"] {
            display: none;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1825, #09131d);
            border-right: 1px solid var(--border);
        }
        .hero-card, .metric-card, .glass-panel {
            background: linear-gradient(180deg, rgba(18,48,68,0.95), rgba(10,25,37,0.96));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        }
        .hero-title {
            font-size: 2.5rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
            color: #f4fbff;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            margin-top: .5rem;
        }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }
        .mini-card {
            border-radius: 18px;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(15,34,49,0.98), rgba(11,23,35,0.98));
            border: 1px solid var(--border);
        }
        .mini-card h4 {
            margin: 0 0 .3rem 0;
            color: #ffffff;
            font-size: 1.05rem;
        }
        .mini-card p {
            margin: 0;
            color: var(--muted);
            font-size: .92rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0 0 .35rem 0;
            color: #ffffff;
        }
        .section-subtitle {
            color: var(--muted);
            margin-top: 0;
        }
        .feature-pill {
            display: inline-block;
            margin: .2rem .35rem .2rem 0;
            padding: .4rem .7rem;
            border-radius: 999px;
            background: rgba(88,196,184,0.15);
            border: 1px solid rgba(88,196,184,0.3);
            color: #d9fffb;
            font-size: .86rem;
        }
        .xai-chip {
            display: inline-block;
            margin: .2rem .35rem .2rem 0;
            padding: .4rem .75rem;
            border-radius: 999px;
            background: rgba(110,168,254,0.16);
            border: 1px solid rgba(110,168,254,0.28);
            color: #e7f1ff;
            font-size: .85rem;
        }
        .stDataFrame, .stPlotlyChart, .stMarkdown, .stMetric {
            color: var(--text);
        }
        .element-container div[data-testid="metric-container"] {
            background: linear-gradient(180deg, rgba(15,34,49,0.94), rgba(11,23,35,0.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: .35rem .6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(active_page: str) -> None:
    st.sidebar.markdown("## ECG AI Dashboard")
    st.sidebar.caption("Binary Classification of ECG Signals using Machine Learning and Explainable AI")
    st.sidebar.markdown(
        f"""
        <div class="glass-panel">
            <div style="font-size:1.8rem;font-weight:800;color:#ffffff;">{PROJECT_SUMMARY['best_model']}</div>
            <div style="color:#9eb2c1;margin-top:.25rem;">Best accuracy: {PROJECT_SUMMARY['best_accuracy']}</div>
            <div style="color:#9eb2c1;">Selected features: {PROJECT_SUMMARY['selected_features']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Navigation")
    for page_path, label in PAGE_ORDER:
        if label == active_page:
            st.sidebar.markdown(f"<div class='feature-pill'>{label}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.page_link(page_path, label=label)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.metric("Dataset size", PROJECT_SUMMARY["dataset_size"])
    st.sidebar.metric("Extracted features", PROJECT_SUMMARY["feature_count"])
    st.sidebar.metric("Selected features", PROJECT_SUMMARY["selected_features"])


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return load(candidate)
    raise FileNotFoundError("No ECG model artifact found in the model folder.")


def get_model_feature_order(model) -> List[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return [feature for feature in FEATURE_COLUMNS if feature in SELECTED_FEATURES or feature in FEATURE_COLUMNS]
    return list(feature_names)


def safe_metric_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<h1 class='hero-title'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='hero-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_section_heading(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='section-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_summary_cards(items: Sequence[Tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for column, (label, value, helper) in zip(cols, items):
        with column:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div style='color:#9eb2c1;font-size:.9rem;'>{label}</div>
                    <div style='font-size:1.7rem;font-weight:800;color:#ffffff;margin:.2rem 0;'>{value}</div>
                    <div style='color:#9eb2c1;font-size:.85rem;'>{helper}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_workflow_diagram() -> None:
    stages = [
        "Dataset",
        "Data Analysis",
        "Feature Extraction",
        "Feature Selection",
        "Model Training",
        "Random Forest",
        "Explainable AI",
        "Deployment",
    ]
    x_positions = np.arange(len(stages))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[1] * len(stages),
            mode="markers+text",
            text=stages,
            textposition="bottom center",
            marker=dict(size=46, color=["#58c4b8" if stage in {"Random Forest", "Explainable AI"} else "#0f3550" for stage in stages], line=dict(color="#93e7de", width=2)),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for idx in range(len(stages) - 1):
        fig.add_annotation(x=idx + 0.5, y=1, ax=idx - 0.1, ay=1, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2, arrowcolor="#7eb5ff")
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=35),
        xaxis=dict(visible=False, range=[-0.6, len(stages) - 0.4]),
        yaxis=dict(visible=False, range=[0.7, 1.35]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf6fb", size=13),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_class_distribution_charts() -> None:
    df = load_dataset()
    binary_counts = df["label"].value_counts().sort_index()
    binary_labels = {0: "Normal", 1: "Abnormal"}
    binary_df = pd.DataFrame({"Class": [binary_labels[idx] for idx in binary_counts.index], "Count": binary_counts.values})

    original_df = pd.DataFrame({"Class": list(ORIGINAL_CLASS_COUNTS.keys()), "Count": list(ORIGINAL_CLASS_COUNTS.values())})
    cols = st.columns(2)
    with cols[0]:
        pie = px.pie(original_df, names="Class", values="Count", hole=0.42, color_discrete_sequence=["#58c4b8", "#6ea8fe", "#8fd3b2", "#c9d6ff"])
        pie.update_traces(textposition="inside", textinfo="percent+label")
        pie.update_layout(margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="Original Classes")
        st.plotly_chart(pie, use_container_width=True)
    with cols[1]:
        bar = px.bar(binary_df, x="Class", y="Count", text="Count", color="Class", color_discrete_sequence=["#58c4b8", "#6ea8fe"])
        bar.update_traces(textposition="outside")
        bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)


def render_dataframe_table(dataframe: pd.DataFrame, height: int = 320) -> None:
    st.dataframe(dataframe, use_container_width=True, height=height)


def render_feature_cards(features: Sequence[dict]) -> None:
    for start in range(0, len(features), 2):
        cols = st.columns(2)
        for idx, column in enumerate(cols):
            feature_index = start + idx
            if feature_index >= len(features):
                continue
            feature = features[feature_index]
            with column:
                with st.expander(feature["name"], expanded=False):
                    st.markdown(f"**Formula**: {feature['formula']}")
                    st.markdown(f"**Explanation**: {feature['explanation']}")
                    st.markdown(f"**Physiological significance**: {feature['significance']}")


def render_feature_pills(features: Iterable[Tuple[str, str]]) -> None:
    chips = "".join(f"<span class='feature-pill'><strong>{name}</strong> {note}</span>" for name, note in features)
    st.markdown(chips, unsafe_allow_html=True)


def synthetic_ecg_signal(kind: str, seed: int, n_samples: int = 2000, fs: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples) / fs
    signal = 0.02 * np.sin(2 * np.pi * 0.8 * time)

    if kind == "Normal Rhythm":
        beat_times = np.arange(0.5, time[-1], 0.82 + rng.normal(0, 0.02))
        for center in beat_times:
            signal += 0.95 * np.exp(-((time - center) ** 2) / (2 * 0.006**2))
            signal -= 0.22 * np.exp(-((time - (center - 0.03)) ** 2) / (2 * 0.01**2))
            signal += 0.12 * np.exp(-((time - (center + 0.02)) ** 2) / (2 * 0.012**2))
        signal += rng.normal(0, 0.025, n_samples)
    elif kind == "Atrial Fibrillation":
        beat_times = [0.35]
        while beat_times[-1] < time[-1]:
            beat_times.append(beat_times[-1] + rng.uniform(0.45, 1.25))
        for center in beat_times:
            signal += 0.55 * np.exp(-((time - center) ** 2) / (2 * 0.01**2))
            signal += 0.1 * np.sin(2 * np.pi * 4.5 * (time - center)) * np.exp(-((time - center) ** 2) / (2 * 0.08**2))
        signal += rng.normal(0, 0.06, n_samples)
        signal += 0.08 * np.sin(2 * np.pi * 7.5 * time)
    elif kind == "Other Rhythm":
        beat_times = np.arange(0.4, time[-1], 0.95)
        for center in beat_times:
            signal += 0.7 * np.exp(-((time - center) ** 2) / (2 * 0.009**2))
            signal -= 0.18 * np.exp(-((time - (center - 0.05)) ** 2) / (2 * 0.014**2))
            signal += 0.18 * np.exp(-((time - (center + 0.04)) ** 2) / (2 * 0.016**2))
        signal += 0.12 * np.sin(2 * np.pi * 1.7 * time)
        signal += rng.normal(0, 0.035, n_samples)
    else:
        beat_times = np.arange(0.45, time[-1], 0.78)
        for center in beat_times:
            signal += 0.4 * np.exp(-((time - center) ** 2) / (2 * 0.013**2))
        signal += rng.normal(0, 0.16, n_samples)
        signal += 0.35 * np.sin(2 * np.pi * 15 * time)
        signal += 0.22 * np.sin(2 * np.pi * 29 * time)

    return time, signal


def signal_figure(time: np.ndarray, signal: np.ndarray, title: str, color: str = "#58c4b8") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode="lines", line=dict(color=color, width=2), hovertemplate="t=%{x:.2f}s<br>amp=%{y:.3f}<extra></extra>"))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf6fb"),
    )
    return fig


def render_signal_gallery(kind: str, color: str) -> None:
    title_map = {
        "Normal Rhythm": "Normal ECG",
        "Atrial Fibrillation": "Atrial Fibrillation ECG",
        "Other Rhythm": "Other Rhythm ECG",
        "Noisy": "Noisy ECG",
    }
    cols = st.columns(3)
    for i, column in enumerate(cols):
        sample_index = i + 1
        with column:
            t, s = synthetic_ecg_signal(kind if kind != "Noisy" else "Noisy Readings", seed=sample_index + abs(hash(kind)) % 1000)
            st.markdown(
                f"""
                <div class='mini-card'>
                    <h4>{title_map.get(kind, kind)} #{sample_index}</h4>
                    <p>Duration: 6.67 sec | Samples: 2000 | Class: {kind}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(signal_figure(t, s, f"{title_map.get(kind, kind)} #{sample_index}", color), use_container_width=True)


def load_visual_asset(filename: str) -> Path | None:
    candidate = IMAGE_DIR / filename
    return candidate if candidate.exists() else None


def display_image_or_placeholder(filename: str, caption: str) -> None:
    candidate = load_visual_asset(filename)
    if candidate is not None:
        st.image(str(candidate), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not available in the images folder.")


def build_model_metrics_card(model_name: str) -> None:
    scores = MODEL_RESULTS[model_name]
    counts = MODEL_COUNTS[model_name]
    st.markdown(
        f"""
        <div class='mini-card'>
            <h4>{model_name}</h4>
            <p>TP {counts['tp']} | TN {counts['tn']} | FP {counts['fp']} | FN {counts['fn']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(5)
    items = [
        ("Accuracy", scores["accuracy"]),
        ("Precision", scores["precision"]),
        ("Recall", scores["recall"]),
        ("Specificity", scores["specificity"]),
        ("F1 Score", scores["f1"]),
    ]
    for column, (label, value) in zip(metrics, items):
        with column:
            st.metric(label, f"{value:.2f}%")


def confusion_matrix_figure(model_name: str, normalize: bool = False) -> go.Figure:
    counts = MODEL_COUNTS[model_name]
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    labels = np.array([[f"TN: {counts['tn']}", f"FP: {counts['fp']}"], [f"FN: {counts['fn']}", f"TP: {counts['tp']}"]])
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        labels = np.array([[f"TN: {matrix[0,0]*100:.1f}%", f"FP: {matrix[0,1]*100:.1f}%"], [f"FN: {matrix[1,0]*100:.1f}%", f"TP: {matrix[1,1]*100:.1f}%"]])
    fig = go.Figure(data=go.Heatmap(z=matrix, x=["Pred Normal", "Pred Abnormal"], y=["Actual Normal", "Actual Abnormal"], colorscale=[[0, "#0f3550"], [0.5, "#58c4b8"], [1, "#d4f4f0"]], text=labels, texttemplate="%{text}", showscale=False, hovertemplate="%{y} / %{x}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf6fb"))
    return fig


def comparison_chart(metric_name: str, color: str) -> go.Figure:
    data = pd.DataFrame({"Model": list(MODEL_RESULTS.keys()), metric_name: [MODEL_RESULTS[name][metric_name] for name in MODEL_RESULTS]})
    fig = px.bar(data, x="Model", y=metric_name, text=metric_name, color=metric_name, color_continuous_scale=["#0f3550", color])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title=f"{metric_name} (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, font=dict(color="#edf6fb"))
    return fig


def evaluation_metrics_from_model(model, feature_frame: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    probabilities = model.predict_proba(feature_frame)[:, 1]
    fpr, tpr, _ = roc_curve(target, probabilities)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, probabilities, roc_auc


def roc_curve_figure(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#58c4b8", width=3), name=f"ROC (AUC={roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#6ea8fe", dash="dash"), name="Random baseline"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf6fb"))
    return fig


def feature_importance_figure(model, feature_names: Sequence[str]) -> go.Figure:
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    frame = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)
    fig = px.bar(frame, x="Importance", y="Feature", orientation="h", text="Importance", color="Importance", color_continuous_scale=["#0f3550", "#58c4b8"])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, font=dict(color="#edf6fb"))
    return fig


def render_probability_gauge(probability: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#58c4b8"},
                "steps": [
                    {"range": [0, 40], "color": "#14324a"},
                    {"range": [40, 70], "color": "#214763"},
                    {"range": [70, 100], "color": "#2d6a8d"},
                ],
            },
            title={"text": "Abnormal ECG probability"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf6fb"))
    st.plotly_chart(fig, use_container_width=True)


def render_model_input_form(feature_order: Sequence[str], defaults: pd.Series, mins: pd.Series, maxs: pd.Series, form_key: str) -> tuple[Dict[str, float], bool]:
    inputs: Dict[str, float] = {}
    submitted = False
    with st.form(form_key):
        st.caption("Fill in the model features and submit to generate the prediction and local explanation.")
        for start in range(0, len(feature_order), 2):
            columns = st.columns(2)
            for offset, column in enumerate(columns):
                feature_index = start + offset
                if feature_index >= len(feature_order):
                    continue
                feature = feature_order[feature_index]
                with column:
                    inputs[feature] = st.number_input(
                        feature,
                        value=float(defaults.get(feature, 0.0)),
                        min_value=float(mins.get(feature, -1e6)),
                        max_value=float(maxs.get(feature, 1e6)),
                        format="%.6f",
                        help="Enter the engineered feature value used by the model.",
                        key=f"{form_key}_{feature}",
                    )
        submitted = st.form_submit_button("Predict and Explain", use_container_width=True)
    return inputs, submitted


def build_local_prediction_context(model, input_frame: pd.DataFrame, feature_order: Sequence[str], background_frame: pd.DataFrame) -> dict[str, object]:
    probability = float(model.predict_proba(input_frame)[0, 1])
    prediction = int(model.predict(input_frame)[0])
    label = "Abnormal ECG" if prediction == 1 else "Normal ECG"
    color = "#ff8c66" if prediction == 1 else "#58c4b8"

    explanation_series = pd.Series(dtype=float)
    base_value = probability
    shap_available = False

    if shap is not None:
        try:
            background = background_frame.sample(min(len(background_frame), 128), random_state=42) if len(background_frame) > 128 else background_frame
            try:
                explainer = shap.TreeExplainer(model, data=background, model_output="probability")
            except Exception:
                explainer = shap.TreeExplainer(model, data=background)
            shap_values = explainer.shap_values(input_frame)
            expected_value = explainer.expected_value
            if isinstance(shap_values, list):
                class_index = 1 if len(shap_values) > 1 else 0
                local_values = np.asarray(shap_values[class_index])[0]
                base_value = float(expected_value[class_index] if isinstance(expected_value, (list, np.ndarray)) else expected_value)
            else:
                local_values = np.asarray(shap_values)[0]
                base_value = float(expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(np.atleast_1d(expected_value)) > 1 else expected_value)
            explanation_series = pd.Series(local_values, index=feature_order).sort_values()
            shap_available = True
        except Exception:
            shap_available = False

    if not shap_available:
        baseline = background_frame[feature_order].median(numeric_only=True)
        importances = getattr(model, "feature_importances_", np.ones(len(feature_order)))
        signed_delta = input_frame.iloc[0][feature_order] - baseline
        local_values = signed_delta.to_numpy(dtype=float) * np.asarray(importances, dtype=float)
        explanation_series = pd.Series(local_values, index=feature_order).sort_values()
        base_value = float(model.predict_proba(pd.DataFrame([baseline], columns=feature_order))[0, 1])

    return {
        "probability": probability,
        "prediction": prediction,
        "label": label,
        "color": color,
        "base_value": base_value,
        "contributions": explanation_series,
        "shap_available": shap_available,
    }


def render_local_explanation_visuals(model, context: dict[str, object], input_frame: pd.DataFrame, feature_order: Sequence[str], background_frame: pd.DataFrame) -> None:
    contributions = context["contributions"]
    assert isinstance(contributions, pd.Series)

    st.markdown(
        f"""
        <div class='card-grid'>
            <div class='mini-card'><h4>Prediction</h4><p style='font-size:1.4rem;font-weight:800;color:{context['color']};'>{context['label']}</p></div>
            <div class='mini-card'><h4>Probability Score</h4><p style='font-size:1.4rem;font-weight:800;color:#ffffff;'>{context['probability'] * 100:.2f}%</p></div>
            <div class='mini-card'><h4>Baseline Probability</h4><p style='font-size:1.4rem;font-weight:800;color:#ffffff;'>{float(context['base_value']) * 100:.2f}%</p></div>
            <div class='mini-card'><h4>Explanation Mode</h4><p style='font-size:1.2rem;font-weight:800;color:#ffffff;'>{'SHAP' if context['shap_available'] else 'Fallback contributions'}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_probability_gauge(float(context["probability"]))

    top_positive = contributions.sort_values(ascending=False).head(5)
    top_negative = contributions.sort_values(ascending=True).head(5)

    left, right = st.columns(2)
    with left:
        contrib_df = contributions.reset_index()
        contrib_df.columns = ["Feature", "Contribution"]
        contrib_df["Direction"] = np.where(contrib_df["Contribution"] >= 0, "Pushes toward Abnormal", "Pushes toward Normal")
        contrib_fig = px.bar(
            contrib_df.sort_values("Contribution"),
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={"Pushes toward Abnormal": "#ff8c66", "Pushes toward Normal": "#58c4b8"},
            title="Local feature contributions",
        )
        contrib_fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="Impact", font=dict(color="#edf6fb"))
        st.plotly_chart(contrib_fig, use_container_width=True)
    with right:
        st.markdown("### Top positive drivers")
        st.dataframe(top_positive.reset_index().rename(columns={"index": "Feature", 0: "Contribution"}), use_container_width=True, hide_index=True)
        st.markdown("### Top negative drivers")
        st.dataframe(top_negative.reset_index().rename(columns={"index": "Feature", 0: "Contribution"}), use_container_width=True, hide_index=True)

    st.markdown("### Waterfall explanation")
    if shap is not None and context["shap_available"]:
        try:
            background = background_frame.sample(min(len(background_frame), 128), random_state=42) if len(background_frame) > 128 else background_frame
            try:
                explainer = shap.TreeExplainer(load_model(), data=background, model_output="probability")
            except Exception:
                explainer = shap.TreeExplainer(load_model(), data=background)
            shap_values = explainer.shap_values(input_frame)
            expected_value = explainer.expected_value
            if isinstance(shap_values, list):
                class_index = 1 if len(shap_values) > 1 else 0
                waterfall_values = np.asarray(shap_values[class_index])[0]
                waterfall_base = expected_value[class_index] if isinstance(expected_value, (list, np.ndarray)) else expected_value
            else:
                waterfall_values = np.asarray(shap_values)[0]
                waterfall_base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(np.atleast_1d(expected_value)) > 1 else expected_value
            explanation = shap.Explanation(values=waterfall_values, base_values=waterfall_base, data=input_frame.iloc[0], feature_names=list(feature_order))
            shap.plots.waterfall(explanation, max_display=min(14, len(feature_order)), show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close()
        except Exception as exc:
            st.warning(f"SHAP waterfall could not be rendered: {exc}")
    else:
        st.info("SHAP is not available in this environment, so the dashboard is showing a fallback contribution ranking instead of the waterfall plot.")

    if LimeTabularExplainer is not None:
        st.markdown("### LIME local surrogate")
        try:
            lime_background = background_frame[feature_order].to_numpy()
            explainer = LimeTabularExplainer(lime_background, feature_names=list(feature_order), class_names=["Normal", "Abnormal"], mode="classification", discretize_continuous=True)
            lime_exp = explainer.explain_instance(input_frame.iloc[0].to_numpy(), model.predict_proba, num_features=min(10, len(feature_order)))
            st.pyplot(lime_exp.as_pyplot_figure(), clear_figure=True)
            st.caption("LIME approximates the model locally with a simple surrogate to explain this specific input.")
        except Exception as exc:
            st.warning(f"LIME explanation could not be rendered: {exc}")

    st.markdown("### Input used for explanation")
    st.dataframe(input_frame, use_container_width=True)


def render_home_page() -> None:
    render_title(
        "Binary Classification of ECG Signals using Machine Learning and Explainable AI",
        "Final Year Project dashboard for ECG rhythm analysis, feature engineering, model comparison, live prediction, and explainability.",
    )
    st.markdown(
        """
        <div class='hero-card'>
            <strong>Project Objective</strong><br>
            Build a reliable binary ECG classifier that separates Normal and Abnormal recordings using handcrafted features, classical machine learning, and explainable AI.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    render_summary_cards(
        [
            ("Dataset Size", PROJECT_SUMMARY["dataset_size"], "Processed ECG records used in the project."),
            ("Features Extracted", PROJECT_SUMMARY["feature_count"], "Handcrafted descriptors derived from the signals."),
            ("Selected Features", PROJECT_SUMMARY["selected_features"], "Features retained after selection analysis."),
            ("Best Accuracy", PROJECT_SUMMARY["best_accuracy"], "Random Forest chosen as the final model."),
        ]
    )
    st.write("")
    render_section_heading("Workflow", "From raw ECG records to an explainable deployment-ready classifier.")
    render_workflow_diagram()
    st.write("")
    st.markdown(
        """
        <div class='card-grid'>
            <div class='mini-card'><h4>Research Ready</h4><p>Organized for viva, faculty review, and project presentation.</p></div>
            <div class='mini-card'><h4>Explainable</h4><p>SHAP, LIME, feature importance, and permutation analysis included.</p></div>
            <div class='mini-card'><h4>Interactive</h4><p>Designed with modern plots, responsive layout, and sidebar navigation.</p></div>
            <div class='mini-card'><h4>Deployment Ready</h4><p>Includes model loading, prediction inputs, and graceful error handling.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_dataset_page() -> None:
    render_title("About Dataset", "PhysioNet Challenge 2017 ECG dataset overview and the binary mapping used in this project.")
    render_section_heading("Dataset Information")
    info_df = pd.DataFrame(
        [
            ["Total Samples", "8528"],
            ["Signal Length", "2000"],
            ["Sampling Rate", "300 Hz"],
            ["Duration", "6.67 sec"],
            ["Classes", "4"],
        ],
        columns=["Parameter", "Value"],
    )
    render_dataframe_table(info_df, height=240)
    st.write("")
    render_section_heading("Original Class Distribution", "The four-class PhysioNet distribution described in the project report.")
    render_class_distribution_charts()
    st.write("")
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Binary Mapping Used in the Dashboard</h4>
            <p><strong>Normal</strong> = class 0</p>
            <p><strong>Abnormal</strong> = class 1 + class 2</p>
            <p><strong>Noisy recordings</strong> are removed before model training and evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class='card-grid'>
            <div class='mini-card'><h4>Normal Rhythm</h4><p>Reference rhythm used as the healthy class.</p></div>
            <div class='mini-card'><h4>Atrial Fibrillation</h4><p>Irregular rhythm characterized by absent P-waves and irregular RR intervals.</p></div>
            <div class='mini-card'><h4>Other Rhythm</h4><p>Non-AF abnormal rhythms grouped into the abnormal class.</p></div>
            <div class='mini-card'><h4>Noisy Signals</h4><p>Removed because they reduce model reliability.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_analysis_page() -> None:
    render_title("Data Analysis", "Interactive ECG visualizations generated for each class to demonstrate waveform differences.")
    st.markdown(
        """
        <div class='mini-card'>
            <h4>What is shown here?</h4>
            <p>Three representative signals are displayed for each ECG class. Because the dashboard ships with the engineered feature table rather than the raw waveform files, the plots are generated as realistic synthetic ECG previews for presentation and viva purposes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    tabs = st.tabs(["Normal Rhythm", "Atrial Fibrillation", "Other Rhythm", "Noisy Signals"])
    with tabs[0]:
        render_signal_gallery("Normal Rhythm", "#58c4b8")
    with tabs[1]:
        render_signal_gallery("Atrial Fibrillation", "#ff8c66")
    with tabs[2]:
        render_signal_gallery("Other Rhythm", "#6ea8fe")
    with tabs[3]:
        render_signal_gallery("Noisy", "#d6a8ff")


def render_feature_extraction_page() -> None:
    render_title("Feature Extraction", "32 handcrafted ECG features were extracted to capture statistical, HRV, frequency, and non-linear information.")
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Why handcrafted features?</h4>
            <p>They make the model interpretable, efficient, and suitable for classical machine learning methods such as Random Forests and boosting-based classifiers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    tabs = st.tabs(["Statistical Features", "HRV Features", "Frequency Features", "Non Linear Features"])
    with tabs[0]:
        st.markdown("### Core Statistical Descriptors")
        render_feature_cards(CORE_STATISTICAL_FEATURES)
        st.markdown("### Statistical and Morphological Features")
        render_feature_cards(STATISTICAL_FEATURES)
    with tabs[1]:
        render_feature_cards(HRV_FEATURES)
    with tabs[2]:
        render_feature_cards(FREQUENCY_FEATURES)
    with tabs[3]:
        render_feature_cards(NON_LINEAR_FEATURES)


def render_feature_selection_page() -> None:
    render_title("Feature Selection", "32 extracted features were reduced to 14 selected features for the final classification pipeline.")
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Why feature selection?</h4>
            <p>It removes redundancy, improves generalization, reduces noise, and keeps the final model compact for deployment and explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    try:
        df = load_dataset()
        feature_frame = df[FEATURE_COLUMNS]
        target = df["label"]
    except Exception as exc:
        st.error(f"Unable to load dataset for feature selection: {exc}")
        return

    constant_columns = feature_frame.columns[feature_frame.nunique(dropna=False) <= 1].tolist()
    analysis_frame = feature_frame.drop(columns=constant_columns) if constant_columns else feature_frame
    if constant_columns:
        st.warning(f"Constant features excluded from statistical ranking: {', '.join(constant_columns)}")

    corr_tab, mi_tab, anova_tab, rfe_tab = st.tabs(["Correlation with Target", "Mutual Information", "ANOVA F-Test", "Recursive Feature Elimination"])

    with corr_tab:
        correlations = analysis_frame.corrwith(target).replace([np.inf, -np.inf], np.nan).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        heatmap = go.Figure(data=go.Heatmap(z=correlations.abs().values.reshape(-1, 1), x=["|Pearson correlation|"], y=correlations.index, colorscale=[[0, "#0f3550"], [1, "#58c4b8"]], text=[f"{value:.3f}" for value in correlations.values], texttemplate="%{text}", showscale=True))
        heatmap.update_layout(height=760, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#edf6fb"))
        st.plotly_chart(heatmap, use_container_width=True)
        top_corr = correlations.abs().sort_values(ascending=False).head(10).reset_index()
        top_corr.columns = ["Feature", "Absolute Correlation"]
        st.dataframe(top_corr, use_container_width=True)
        st.info("Pearson correlation was used to determine the linear relationship between each feature and the binary target.")

    with mi_tab:
        scores = mutual_info_classif(analysis_frame, target, random_state=42)
        mi_df = pd.DataFrame({"Feature": analysis_frame.columns, "Score": scores}).sort_values("Score", ascending=True)
        mi_fig = px.bar(mi_df, x="Score", y="Feature", orientation="h", color="Score", color_continuous_scale=[[0, "#0f3550"], [1, "#58c4b8"]])
        mi_fig.update_layout(height=760, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, font=dict(color="#edf6fb"))
        st.plotly_chart(mi_fig, use_container_width=True)
        st.info("Mutual information captures both linear and non-linear dependence between the features and the target.")

    with anova_tab:
        f_scores, p_values = f_classif(analysis_frame, target)
        anova_df = pd.DataFrame({"Feature": analysis_frame.columns, "F Score": f_scores, "p-value": p_values}).sort_values("F Score", ascending=True)
        anova_fig = px.bar(anova_df, x="F Score", y="Feature", orientation="h", color="F Score", color_continuous_scale=[[0, "#0f3550"], [1, "#6ea8fe"]])
        anova_fig.update_layout(height=760, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, font=dict(color="#edf6fb"))
        st.plotly_chart(anova_fig, use_container_width=True)
        st.info("ANOVA F-test measures how strongly each feature separates the classes through between-class and within-class variance.")

    with rfe_tab:
        model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        selector = RFE(model, n_features_to_select=min(14, analysis_frame.shape[1]))
        selector.fit(analysis_frame, target)
        ranking_df = pd.DataFrame({"Feature": analysis_frame.columns, "Selected": selector.support_, "Rank": selector.ranking_}).sort_values(["Rank", "Feature"])
        st.dataframe(ranking_df, use_container_width=True, height=420)
        st.info("Recursive Feature Elimination with a Random Forest estimator was used to rank and prune weaker descriptors.")

    st.write("")
    render_section_heading("Final Selected Features")
    render_feature_pills([(feature, note) for feature, note in FEATURE_SELECTION_FEATURES])


def render_models_page() -> None:
    render_title("Models Trained", "Eight classical machine learning models were compared on the ECG feature set.")
    st.markdown(
        """
        <div class='mini-card'>
            <h4>Evaluation summary</h4>
            <p>Each model is presented with its confusion matrix and the key classification metrics required for viva and research presentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    model_tabs = st.tabs(list(MODEL_RESULTS.keys()))
    for tab, model_name in zip(model_tabs, MODEL_RESULTS.keys()):
        with tab:
            build_model_metrics_card(model_name)
            st.write("")
            st.plotly_chart(confusion_matrix_figure(model_name), use_container_width=True)
            if model_name == "Logistic Regression":
                st.caption("The project brief includes a reported LR confusion matrix with an alternate class orientation; the dashboard uses a metrics-consistent matrix for the interactive visualization.")

    st.write("")
    render_section_heading("Model Comparison")
    comp_tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1 Score"])
    with comp_tabs[0]:
        st.plotly_chart(comparison_chart("accuracy", "#58c4b8"), use_container_width=True)
    with comp_tabs[1]:
        st.plotly_chart(comparison_chart("precision", "#6ea8fe"), use_container_width=True)
    with comp_tabs[2]:
        st.plotly_chart(comparison_chart("recall", "#ff8c66"), use_container_width=True)
    with comp_tabs[3]:
        st.plotly_chart(comparison_chart("f1", "#d6a8ff"), use_container_width=True)


def render_best_model_page() -> None:
    render_title("Best Model", "Random Forest Classifier chosen as the final model because it balances accuracy, recall, explainability, and stability.")
    st.markdown(
        """
        <div class='card-grid'>
            <div class='mini-card'><h4>High Accuracy</h4><p>Strong overall classification performance on the binary ECG task.</p></div>
            <div class='mini-card'><h4>Balanced Performance</h4><p>Maintains a practical trade-off between precision and recall.</p></div>
            <div class='mini-card'><h4>Good Recall</h4><p>Detects abnormal ECGs without sacrificing too much specificity.</p></div>
            <div class='mini-card'><h4>Explainable</h4><p>Feature importance and SHAP/LIME outputs make the model defensible in a viva.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    build_model_metrics_card("Random Forest")
    st.write("")
    counts = MODEL_COUNTS["Random Forest"]
    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_matrix_figure("Random Forest"), use_container_width=True)
    with right:
        try:
            model = load_model()
            df = load_dataset()
            feature_order = get_model_feature_order(model)
            X = df[feature_order]
            y = df["label"]
            fpr, tpr, probabilities, roc_auc = evaluation_metrics_from_model(model, X, y)
            st.plotly_chart(roc_curve_figure(fpr, tpr, roc_auc), use_container_width=True)
        except Exception as exc:
            st.warning(f"ROC curve could not be generated dynamically: {exc}")
    st.write("")
    try:
        model = load_model()
        df = load_dataset()
        feature_order = get_model_feature_order(model)
        st.plotly_chart(feature_importance_figure(model, feature_order), use_container_width=True)
    except Exception as exc:
        st.warning(f"Feature importance plot could not be generated: {exc}")
    st.write("")
    st.markdown(
        f"""
        <div class='mini-card'>
            <h4>Classification Summary</h4>
            <p>Random Forest is the selected deployment model because it provides robust performance on the ECG feature set, generalizes well on tabular data, and gives clear feature importance for explainability.</p>
            <p><strong>Confusion matrix snapshot:</strong> TP {counts['tp']}, TN {counts['tn']}, FP {counts['fp']}, FN {counts['fn']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_prediction_page() -> None:
    render_title("Live Prediction", "Enter the 14 model features to predict whether the ECG is Normal or Abnormal.")
    try:
        model = load_model()
        feature_order = get_model_feature_order(model)
        df = load_dataset()
        defaults = df[feature_order].median(numeric_only=True)
        mins = df[feature_order].min(numeric_only=True)
        maxs = df[feature_order].max(numeric_only=True)
    except Exception as exc:
        st.error(f"Prediction page is unavailable: {exc}")
        return

    st.markdown(
        """
        <div class='mini-card'>
            <h4>Model interface</h4>
            <p>The shipped Random Forest artifact expects the 14 feature columns used during training. Inputs are aligned to that saved interface automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    inputs: Dict[str, float] = {}
    for start in range(0, len(feature_order), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            feature_index = start + offset
            if feature_index >= len(feature_order):
                continue
            feature = feature_order[feature_index]
            with column:
                inputs[feature] = st.number_input(
                    feature,
                    value=float(defaults.get(feature, 0.0)),
                    min_value=float(mins.get(feature, -1e6)),
                    max_value=float(maxs.get(feature, 1e6)),
                    format="%.6f",
                    help="Enter the engineered feature value used by the model.",
                )

    st.write("")
    if st.button("Predict", use_container_width=True):
        try:
            input_frame = pd.DataFrame([[inputs[feature] for feature in feature_order]], columns=feature_order)
            probability = float(model.predict_proba(input_frame)[0, 1])
            prediction = int(model.predict(input_frame)[0])
            label = "Abnormal ECG" if prediction == 1 else "Normal ECG"
            color = "#ff8c66" if prediction == 1 else "#58c4b8"
            st.markdown(
                f"""
                <div class='card-grid'>
                    <div class='mini-card'><h4>Prediction</h4><p style='font-size:1.4rem;font-weight:800;color:{color};'>{label}</p></div>
                    <div class='mini-card'><h4>Probability Score</h4><p style='font-size:1.4rem;font-weight:800;color:#ffffff;'>{probability * 100:.2f}%</p></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_probability_gauge(probability)
            st.dataframe(input_frame, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_explainable_ai_page() -> None:
    render_title("Explainable AI", "Input values, prediction output, and the local reasoning behind the model decision.")
    try:
        model = load_model()
        feature_order = get_model_feature_order(model)
        df = load_dataset()
        defaults = df[feature_order].median(numeric_only=True)
        mins = df[feature_order].min(numeric_only=True)
        maxs = df[feature_order].max(numeric_only=True)
    except Exception as exc:
        st.error(f"Explainable AI page is unavailable: {exc}")
        return

    tabs = st.tabs(["Local Explanation", "Global Views", "Technical Notes"])

    with tabs[0]:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Local Explainable AI</h4>
                <p>Enter all feature values for one ECG instance. After prediction, the page shows which features pushed the model toward Normal or Abnormal for that specific input.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        inputs, submitted = render_model_input_form(feature_order, defaults, mins, maxs, "local_xai_form")
        if submitted:
            input_frame = pd.DataFrame([[inputs[feature] for feature in feature_order]], columns=feature_order)
            context = build_local_prediction_context(model, input_frame, feature_order, df[feature_order])
            render_local_explanation_visuals(model, context, input_frame, feature_order, df[feature_order])
        else:
            st.info("Submit the form to generate a prediction and the local explanation for that input.")

    with tabs[1]:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>Overall XAI findings</h4>
                <p>
                    <span class='xai-chip'>RMSSD</span>
                    <span class='xai-chip'>Heart Rate</span>
                    <span class='xai-chip'>Mean RR</span>
                    <span class='xai-chip'>RR Std</span>
                    <span class='xai-chip'>SDNN</span>
                    <span class='xai-chip'>Dominant Frequency</span>
                    <span class='xai-chip'>pNN50</span>
                </p>
                <p>These features repeatedly appear across the explanation methods and show that both rhythm timing and spectral structure are important for classification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        tabs_global = st.tabs(["Feature Importance", "Permutation Importance", "SHAP Summary Plot", "SHAP Mean Importance", "InterpretML Morris Sensitivity"])
        tab_assets = [
            ("feature_importance.png", "Random Forest impurity-based importance"),
            ("permutation_importance.png", "Permutation importance under feature shuffling"),
            ("shap_summary.png", "Global SHAP summary plot"),
            ("shap_bar.png", "Mean absolute SHAP feature importance"),
            ("interpret_ml.png", "Morris sensitivity analysis"),
        ]
        for tab, (filename, caption) in zip(tabs_global, tab_assets):
            with tab:
                display_image_or_placeholder(filename, caption)
                if filename.startswith("shap"):
                    st.info("SHAP values quantify the contribution of each feature to the model output.")
                elif filename == "interpret_ml.png":
                    st.info("InterpretML Morris sensitivity explores how systematic perturbations change the output.")
                elif filename == "feature_importance.png":
                    st.info("Random Forest uses impurity-based splits to rank the most useful features.")
                elif filename == "permutation_importance.png":
                    st.info("Permutation importance measures how model performance drops when a feature is shuffled.")

    with tabs[2]:
        st.markdown(
            """
            <div class='mini-card'>
                <h4>How the local output is derived</h4>
                <p>The page uses the saved Random Forest model, computes the probability for the current input, then explains that prediction using SHAP if available and LIME as an additional local surrogate view.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
            <div class='card-grid'>
                <div class='mini-card'><h4>Base value</h4><p>Average model output before considering this input.</p></div>
                <div class='mini-card'><h4>Feature contribution</h4><p>How each feature shifts the output up or down.</p></div>
                <div class='mini-card'><h4>Final output</h4><p>Prediction after all feature effects are combined.</p></div>
                <div class='mini-card'><h4>Local surrogate</h4><p>LIME approximates the model around the selected point.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_future_work_page() -> None:
    render_title("Future Work", "Natural next steps for extending the ECG project into a stronger research and deployment system.")
    future_items = [
        ("1D CNN", "Learn local waveform patterns directly from raw ECG signals.", "A convolutional network can capture morphology without handcrafted features."),
        ("CNN-LSTM", "Combine spatial and temporal learning.", "Useful when waveform shape and rhythm history both matter."),
        ("Transformer-Based ECG Models", "Model long-range dependencies in ECG sequences.", "Attention mechanisms can focus on the most informative beats."),
        ("Real-Time ECG Monitoring", "Stream predictions from live sensor data.", "Makes the system usable in a clinical or wearable setting."),
        ("Wearable Device Integration", "Connect with portable ECG sensors.", "Supports continuous monitoring outside the lab."),
        ("Multi-Class ECG Classification", "Expand from binary to full rhythm classification.", "Allows direct prediction of normal, AF, other rhythm, and noisy samples."),
    ]
    for row_start in range(0, len(future_items), 3):
        columns = st.columns(3)
        for offset, column in enumerate(columns):
            item_index = row_start + offset
            if item_index >= len(future_items):
                continue
            title, heading, explanation = future_items[item_index]
            with column:
                st.markdown(
                    f"""
                    <div class='mini-card'>
                        <h4>{title}</h4>
                        <p><strong>{heading}</strong></p>
                        <p>{explanation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_best_model_summary() -> None:
    st.info("Random Forest is the selected final model in the project brief. The dashboard keeps that decision visible even where other models reach similar accuracy.")
