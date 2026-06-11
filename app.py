import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from preprocess import CATEGORICAL_COLS, ATTACK_MAP

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NIDS — Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# ------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------
DATASET_CONFIG = {
    "NSL-KDD": {
        "model_path": "rf_model.pkl",
        "preprocessor_path": "preprocessor.pkl",
        "description": "Classic benchmark dataset with 41 features, 5 attack categories.",
        "best_model": "XGBoost",
        "macro_f1": "0.58",
        "cv_f1": "0.93",
        "train_samples": "125,973",
        "test_samples": "22,544",
        "classes": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "confusion_matrix": "results/confusion_matrix.png",
        "feature_importance": "results/feature_importance.png",
        "shap_summary": "results/shap_summary.png",
        "limitation": (
            "R2L and U2R recall is low due to train/test distribution shift "
            "in NSL-KDD. R2L jumps from ~0.8% of train to ~12.2% of test. "
            "This is a documented dataset property, not a pipeline bug."
        ),
        "upload_hint": "Upload a CSV in NSL-KDD format (no header, 41–43 columns).",
        "format": "nslkdd",
    },
    "UNSW-NB15": {
        "model_path": "rf_model_unsw.pkl",
        "preprocessor_path": "preprocessor_unsw.pkl",
        "description": "Modern dataset with realistic traffic, 10 attack categories mapped to 5.",
        "best_model": "Random Forest",
        "macro_f1": "0.63",
        "cv_f1": "—",
        "train_samples": "175,341",
        "test_samples": "82,332",
        "classes": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "confusion_matrix": "results/unsw_confusion_matrix.png",
        "feature_importance": "results/unsw_feature_importance.png",
        "shap_summary": "results/unsw_shap_summary.png",
        "limitation": (
            "10 UNSW-NB15 attack categories are mapped to 5 standard classes. "
            "Generic, Exploits, Fuzzers, Analysis → R2L. "
            "Backdoor, Shellcode, Worms → U2R."
        ),
        "upload_hint": "Upload a CSV in UNSW-NB15 format (with header, includes proto/service/state columns).",
        "format": "unsw",
    },
}

# NSL-KDD column definitions
COLUMNS_NO_LABEL = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

UNSW_CATEGORICAL_COLS = ['proto', 'service', 'state']

UNSW_DROP_COLS = ['id', 'label', 'attack_cat']

# ------------------------------------------------------------------
# Model loading (cached per dataset)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_path, preprocessor_path):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

@st.cache_resource
def load_explainer(_model, _key):
    return shap.TreeExplainer(_model)

# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------
def preprocess_nslkdd(df_raw, preprocessor):
    if df_raw.shape[1] == 43:
        df_raw.columns = COLUMNS_NO_LABEL + ['label', 'difficulty']
        df_raw = df_raw.drop(['label', 'difficulty'], axis=1)
    elif df_raw.shape[1] == 42:
        df_raw.columns = COLUMNS_NO_LABEL + ['label']
        df_raw = df_raw.drop('label', axis=1)
    else:
        df_raw.columns = COLUMNS_NO_LABEL

    X = preprocessor.transform(df_raw)
    feature_names = (
        CATEGORICAL_COLS +
        [c for c in COLUMNS_NO_LABEL if c not in CATEGORICAL_COLS]
    )
    return pd.DataFrame(X, columns=feature_names)


def preprocess_unsw(df_raw, preprocessor):
    # Drop columns that aren't features
    drop_existing = [c for c in UNSW_DROP_COLS if c in df_raw.columns]
    df_feat = df_raw.drop(columns=drop_existing)

    X = preprocessor.transform(df_feat)
    numeric_cols = [c for c in df_feat.columns if c not in UNSW_CATEGORICAL_COLS]
    feature_names = UNSW_CATEGORICAL_COLS + numeric_cols
    return pd.DataFrame(X, columns=feature_names)

# ------------------------------------------------------------------
# Sidebar — dataset selector
# ------------------------------------------------------------------
st.sidebar.title("🛡️ NIDS Dashboard")
st.sidebar.markdown("---")

selected_dataset = st.sidebar.radio(
    "Dataset",
    options=list(DATASET_CONFIG.keys()),
    help="Switch between NSL-KDD (classic benchmark) and UNSW-NB15 (modern realistic traffic)."
)

cfg = DATASET_CONFIG[selected_dataset]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best model:** {cfg['best_model']}")
st.sidebar.markdown(f"**Macro F1:** {cfg['macro_f1']}")
st.sidebar.markdown(f"**Classes:** {', '.join(cfg['classes'])}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"⚠️ **Note:** {cfg['limitation']}")

# ------------------------------------------------------------------
# Load artifacts for selected dataset
# ------------------------------------------------------------------
try:
    model, preprocessor = load_artifacts(cfg["model_path"], cfg["preprocessor_path"])
    explainer = load_explainer(model, selected_dataset)
    class_names = list(model.classes_)
except Exception as e:
    st.error(f"Could not load model for {selected_dataset}: {e}")
    st.stop()

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.title("Network Intrusion Detection System")
st.markdown(
    f"Classify network traffic into attack categories using machine learning. "
    f"**Active dataset: {selected_dataset}** — {cfg['description']}"
)

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    cfg["upload_hint"],
    type=["csv", "txt"]
)

if uploaded_file is not None:
    # ---- Load ----
    try:
        if cfg["format"] == "nslkdd":
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            df_raw = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_raw):,} records.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ---- Preprocess ----
    try:
        if cfg["format"] == "nslkdd":
            X = preprocess_nslkdd(df_raw.copy(), preprocessor)
        else:
            X = preprocess_unsw(df_raw.copy(), preprocessor)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # ---- Predict ----
    preds = model.predict(X)
    proba = model.predict_proba(X)

    results_df = df_raw.copy()
    if cfg["format"] == "nslkdd":
        results_df.columns = (
            COLUMNS_NO_LABEL + ['label', 'difficulty'][:df_raw.shape[1] - 41]
        )[:df_raw.shape[1]]
    results_df['Prediction'] = preds
    results_df['Confidence'] = proba.max(axis=1).round(3)

    # ---- Summary metrics ----
    st.markdown("---")
    st.subheader("Detection Summary")

    counts = pd.Series(preds).value_counts()
    color_map = {
        'Normal': '🟢', 'DoS': '🔴', 'Probe': '🟡',
        'R2L': '🟠', 'U2R': '🔴'
    }
    cols = st.columns(len(class_names))
    for i, cls in enumerate(class_names):
        count = counts.get(cls, 0)
        cols[i].metric(
            label=f"{color_map.get(cls, '⚪')} {cls}",
            value=count,
            delta=f"{count / len(preds) * 100:.1f}%"
        )

    # ---- Results table ----
    st.markdown("---")
    st.subheader("Prediction Results")

    if cfg["format"] == "nslkdd":
        preview_cols = ['Prediction', 'Confidence'] + COLUMNS_NO_LABEL[:6]
    else:
        feature_preview = [c for c in df_raw.columns
                           if c not in UNSW_DROP_COLS][:6]
        preview_cols = ['Prediction', 'Confidence'] + feature_preview

    
    show_cols = [c for c in preview_cols if c in results_df.columns]
    display_df = results_df[show_cols].head(500)

    try:
        # SAFE display (no styling)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300
        )

        if len(results_df) > 500:
            st.caption(
                f"Showing first 500 of {len(results_df):,} rows. "
                "Download CSV for full results."
            )

    except Exception:
        # fallback
        st.dataframe(display_df, use_container_width=True, height=300)
    csv_out = results_df[['Prediction', 'Confidence']].to_csv(index=False)
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_out,
        file_name=f"nids_predictions_{selected_dataset.lower().replace('-', '_')}.csv",
        mime="text/csv"
    )

    # ---- SHAP explanation ----
    st.markdown("---")
    st.subheader("Explainability — Why did the model flag this record?")
    st.markdown("Select any row to see which features drove that prediction.")

    row_idx = st.slider(
        "Select record index",
        min_value=0,
        max_value=len(X) - 1,
        value=0
    )

    selected_pred = preds[row_idx]
    selected_conf = proba[row_idx].max()
    class_idx = list(class_names).index(selected_pred)

    st.markdown(
        f"**Record {row_idx}** — Predicted: `{selected_pred}` "
        f"— Confidence: `{selected_conf:.1%}`"
    )

    with st.spinner("Computing SHAP values..."):
        shap_values = explainer.shap_values(X.iloc[[row_idx]])

        shap_exp = shap.Explanation(
            values=shap_values[0, :, class_idx],
            base_values=explainer.expected_value[class_idx],
            data=X.iloc[row_idx].values,
            feature_names=list(X.columns)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp, max_display=12, show=False)
        plt.title(
            f"SHAP — Record {row_idx} predicted as {selected_pred}",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ---- Confidence bar chart ----
    st.markdown("---")
    st.subheader("Confidence scores — all classes")
    proba_df = pd.DataFrame(
        {'Confidence': proba[row_idx]},
        index=class_names
    ).round(4)
    st.bar_chart(proba_df)

else:
    # ------------------------------------------------------------------
    # Landing state — pre-generated results, dataset-aware
    # ------------------------------------------------------------------
    st.info(
        f"Upload a CSV file above to run live detection on **{selected_dataset}**. "
        "Showing pre-generated results below."
    )

    st.markdown("---")
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", cfg["best_model"])
    col2.metric("Macro F1 (test)", cfg["macro_f1"])
    col3.metric("Training samples", cfg["train_samples"])
    col4.metric("Test samples", cfg["test_samples"])

    # Cross-dataset comparison table
    st.markdown("---")
    st.subheader("Cross-Dataset Comparison")
    comparison_df = pd.DataFrame({
        "Dataset":        ["NSL-KDD", "UNSW-NB15"],
        "Best Model":     ["XGBoost", "Random Forest"],
        "Macro F1 (test)":["0.58",    "0.63"],
        "CV F1":          ["0.93",    "—"],
        "Train samples":  ["125,973", "175,341"],
        "Test samples":   ["22,544",  "82,332"],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption(
        "UNSW-NB15 achieves higher macro F1 partly because its test set has "
        "less severe distribution shift than NSL-KDD."
    )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Feature Importance", "SHAP Summary"])

    with tab1:
        st.image(cfg["confusion_matrix"], use_container_width=True)
    with tab2:
        st.image(cfg["feature_importance"], use_container_width=True)
    with tab3:
        st.image(cfg["shap_summary"], use_container_width=True)