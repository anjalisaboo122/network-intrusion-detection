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
# Load saved model and preprocessor (cached so they load once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model, preprocessor = load_artifacts()
explainer = load_explainer(model)
class_names = list(model.classes_)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("🛡️ NIDS Dashboard")
st.sidebar.markdown("Upload a CSV of network traffic to detect intrusions.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Dataset:** NSL-KDD")
st.sidebar.markdown("**Classes:** Normal, DoS, Probe, R2L, U2R")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Known limitation:** R2L and U2R recall is low due to "
    "train/test distribution shift in NSL-KDD. "
    "This is a documented dataset property."
)

# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.title("Network Intrusion Detection System")
st.markdown("Classify network traffic into attack categories using machine learning.")

# ------------------------------------------------------------------
# File upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload network traffic CSV (NSL-KDD format, no header)",
    type=["csv", "txt"]
)

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

def preprocess_uploaded(df_raw):
    """Apply saved preprocessor to uploaded data."""
    # If file has 43 cols (with label + difficulty), strip them
    if df_raw.shape[1] == 43:
        df_raw.columns = COLUMNS_NO_LABEL + ['label', 'difficulty']
        df_raw = df_raw.drop(['label', 'difficulty'], axis=1)
    elif df_raw.shape[1] == 42:
        df_raw.columns = COLUMNS_NO_LABEL + ['label']
        df_raw = df_raw.drop('label', axis=1)
    else:
        df_raw.columns = COLUMNS_NO_LABEL

    X = preprocessor.transform(df_raw)
    feature_names = (CATEGORICAL_COLS +
                     [c for c in COLUMNS_NO_LABEL if c not in CATEGORICAL_COLS])
    return pd.DataFrame(X, columns=feature_names)


if uploaded_file is not None:
    # ---- Load and preprocess ----
    try:
        df_raw = pd.read_csv(uploaded_file, header=None)
        st.success(f"Loaded {len(df_raw):,} records.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    try:
        X = preprocess_uploaded(df_raw.copy())
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # ---- Predict ----
    preds = model.predict(X)
    proba = model.predict_proba(X)

    results_df = df_raw.copy()
    results_df.columns = (COLUMNS_NO_LABEL + 
                          ['label', 'difficulty'][:df_raw.shape[1] - 41])[:df_raw.shape[1]]
    results_df['Prediction'] = preds
    results_df['Confidence'] = proba.max(axis=1).round(3)

    # ---- Summary metrics ----
    st.markdown("---")
    st.subheader("Detection Summary")

    counts = pd.Series(preds).value_counts()
    cols = st.columns(len(class_names))
    colors = {
        'Normal': '🟢', 'DoS': '🔴', 'Probe': '🟡',
        'R2L': '🟠', 'U2R': '🔴'
    }
    for i, cls in enumerate(class_names):
        count = counts.get(cls, 0)
        cols[i].metric(
            label=f"{colors.get(cls, '⚪')} {cls}",
            value=count,
            delta=f"{count/len(preds)*100:.1f}%"
        )

    # ---- Results table ----
    st.markdown("---")
    st.subheader("Prediction Results")

    show_cols = ['Prediction', 'Confidence'] + COLUMNS_NO_LABEL[:6]
    st.dataframe(
            results_df[show_cols].style.map(
                lambda v: 'background-color: #ffcccc' if v not in ['Normal'] else '',
                subset=['Prediction']
            ),
            use_container_width=True,
            height=300
        )
    # Download button
    csv_out = results_df[['Prediction', 'Confidence']].to_csv(index=False)
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_out,
        file_name="nids_predictions.csv",
        mime="text/csv"
    )

    # ---- SHAP explanation for selected row ----
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
    class_idx = class_names.index(selected_pred)

    st.markdown(f"**Record {row_idx}** — Predicted: `{selected_pred}` — Confidence: `{selected_conf:.1%}`")
    
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

    # ---- Confidence breakdown per class ----
    st.markdown("---")
    st.subheader("Confidence scores — all classes")
    proba_df = pd.DataFrame(
        {'Confidence': proba[row_idx]},
        index=class_names
    ).round(4)
    st.bar_chart(proba_df)

else:
    # Landing state — show pre-generated plots
    st.info("Upload a CSV file above to run live detection. Showing pre-generated results below.")

    st.markdown("---")
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF Macro F1", "0.4855")
    col2.metric("XGB Macro F1", "0.5728")
    col3.metric("Training samples", "125,973")
    col4.metric("Test samples", "22,544")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Feature Importance", "SHAP Summary"])

    with tab1:
        st.image('results/confusion_matrix.png', use_container_width=True)
    with tab2:
        st.image('results/feature_importance.png', use_container_width=True)
    with tab3:
        st.image('results/shap_summary.png', use_container_width=True)