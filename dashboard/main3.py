import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings("ignore")

from predict_pipeline import run_complete_pipeline, validate_pipeline_results, extract_lime_data
from lime_explainer import create_lime_explainer_from_pipeline, explain_test_instance, explain_custom_instance
from integration_test import test_complete_integration, print_pipeline_summary

# === Constants / Helpers ===
RESAMPLING_METHODS = {
    'SMOTE': 'Synthetic Minority Oversampling Technique',
    'ADASYN': 'Adaptive Synthetic Sampling',
    'ENN': 'Edited Nearest Neighbours',
    'TomekLinks': 'Tomek Links',
    'SMOTEENN': 'SMOTE + Edited Nearest Neighbours',
    'SMOTETomek': 'SMOTE + Tomek Links'
}

ACTIVATION_OPTIONS = {
    'sigmoid': 'Sigmoid (0,1)',
    'tanh': 'Tanh (-1,1)',
    'relu': 'ReLU [0,∞)'
}

# === Session State Initialization ===
def init_session_state():
    defaults = {
        'current_page': 'upload',
        'uploaded_data': None,
        'pipeline_results': None,
        'lime_explainer': None,
        'processing_complete': False,
        'selected_resampling': 'SMOTE',
        'selected_training_mode': 'manual',
        'selected_hidden_neurons': 100,
        'selected_activation': 'sigmoid',
        'selected_threshold': 0.5,
        'optuna_trials': 50,
        'explanation_index': 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# === Page Components ===
def page_upload():
    st.header("📤 Upload Data")
    st.write("Unggah dataset transaksi untuk dideteksi fraud. Format CSV, minimal kolom: amount, inquiryAmount, settlementAmount, merchantId, paymentSource, status, statusCode, createdTime, updatedTime.")
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'], key="upload_csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return

        st.session_state.uploaded_data = df.copy()
        st.success(f"Data berhasil diunggah: {df.shape[0]} baris x {df.shape[1]} kolom")
        st.dataframe(df.head(5), use_container_width=True)
        st.markdown("### Ringkasan cepat")
        st.write(f"- Kolom: {list(df.columns)}")
        st.write(f"- Missing: {df.isnull().sum().sum()} nilai")
    else:
        st.info("Belum ada file diupload.")

def page_process():
    st.header("🔧 Preprocessing & Model Pipeline")
    st.write("Atur parameter pipeline dan jalankan seluruh alur sesuai desain sistem.")

    # Sidebar-like controls (could be inline)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resampling")
        st.session_state.selected_resampling = st.selectbox("Pilih metode resampling", list(RESAMPLING_METHODS.keys()),
                                                           index=list(RESAMPLING_METHODS.keys()).index(st.session_state.selected_resampling))
        st.markdown(f"**Deskripsi:** {RESAMPLING_METHODS[st.session_state.selected_resampling]}")

        st.subheader("Training Mode")
        st.session_state.selected_training_mode = st.radio("Manual vs Auto-tuning", ['manual', 'auto'],
                                                          index=0 if st.session_state.selected_training_mode == 'manual' else 1)
    with col2:
        st.subheader("ELM Hyperparameters")
        if st.session_state.selected_training_mode == 'manual':
            st.session_state.selected_hidden_neurons = st.slider("Hidden Neurons", 10, 300, st.session_state.selected_hidden_neurons)
            st.session_state.selected_activation = st.selectbox("Activation Function", list(ACTIVATION_OPTIONS.keys()),
                                                                index=list(ACTIVATION_OPTIONS.keys()).index(st.session_state.selected_activation))
            st.session_state.selected_threshold = st.slider("Decision Threshold", 0.1, 0.9, st.session_state.selected_threshold, 0.05)
        else:
            st.session_state.optuna_trials = st.number_input("Optuna Trials", min_value=10, max_value=200, value=st.session_state.optuna_trials, step=10)

    st.markdown("---")
    if st.session_state.uploaded_data is None:
        st.warning("Silakan unggah data dulu di menu Upload.")
        return

    if not st.session_state.processing_complete:
        if st.button("🔄 Start Preprocessing & Analysis", use_container_width=True):
            with st.spinner("Menjalankan pipeline lengkap..."):
                pipeline_results, err = execute_full_flow(st.session_state.uploaded_data)
                if err:
                    st.error(f"❌ {err}")
                    return
                st.success("✅ Pipeline selesai dijalankan.")
                # Optionally, show a summary
                print_pipeline_summary(pipeline_results)
                st.session_state.processing_complete = True
                st.experimental_rerun()
    else:
        st.success("✅ Pipeline sudah dijalankan. Lihat di tab Analisis atau Interpretasi LIME.")

def page_analysis():
    st.header("📊 Analisis Hasil")
    if st.session_state.pipeline_results is None:
        st.error("Pipeline belum dijalankan. Kembali ke halaman preprocessing.")
        return

    results = st.session_state.pipeline_results

    st.subheader("1. Performance Metrics")
    metrics_df = results['metrics_table']
    st.table(metrics_df.set_index("Metric"))

    st.subheader("2. Confusion Matrix")
    cm = results['confusion_matrix']
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm.values,
        x=cm.columns,
        y=cm.index,
        hovertemplate="Pred: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        colorscale="Blues"
    ))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("3. Resampling Summary")
    st.write(f"Metode: **{st.session_state.selected_resampling}**")
    resampling_method = results.get('resampling_method')
    prep = results.get('preprocessing_results', {})
    # if resampling stats exist inside pipeline_results (the wrapper might need to expose them)
    resampling_stats = results.get('resampling_method')  # fallback if not present
    # Show original vs resampled distribution if available
    if 'resampling_method' in results:
        st.write(f"Resampling method applied: {results['resampling_method']}")

    if results.get('model_results'):
        st.subheader("4. Model Configuration")
        mdl = results['model_results']
        st.write(f"- Mode: {mdl.get('mode', '')}")
        params = mdl.get('parameters', {})
        for k, v in params.items():
            st.write(f"  - {k}: {v}")

    st.markdown("---")
    st.subheader("5. Preprocessing Snapshot")
    if prep:
        with st.expander("Lihat detail preprocessing steps"):
            st.json(prep['steps'])
            st.markdown("### Summary")
            st.json(prep['summary'])

def page_explanation():
    st.header("🔍 Interpretasi dengan LIME")
    if st.session_state.pipeline_results is None or st.session_state.lime_explainer is None:
        st.error("Pipeline / explainer belum siap. Jalankan dulu di halaman preprocessing.")
        return

    # Select instance to explain
    lime_data = st.session_state.pipeline_results['lime_data']
    max_idx = len(lime_data['X_test']) - 1
    st.session_state.explanation_index = st.number_input("Pilih indeks instance", min_value=0, max_value=max_idx,
                                                        value=st.session_state.explanation_index, step=1)

    num_features = st.slider("Jumlah fitur yang ditampilkan di explanation", 3, 15, 5)

    # Explain selected test instance
    explanation_result = explain_test_instance(st.session_state.pipeline_results, instance_idx=st.session_state.explanation_index,
                                               num_features=num_features)

    if explanation_result is None:
        st.error("Gagal menghasilkan explanation untuk instance tersebut.")
        return

    st.subheader("Instance Prediction")
    st.write(f"- Actual: **{explanation_result['actual_class']}**")
    st.write(f"- Predicted: **{explanation_result['predicted_class']}**")
    st.write(f"- Confidence: **{explanation_result['confidence']:.3f}**")

    st.subheader("Feature Contribution")
    df_exp = explanation_result['explanation_data']['explanation_df']
    st.dataframe(df_exp[['Feature', 'Importance', 'Impact']].reset_index(drop=True))

    # Optional: custom instance
    st.markdown("---")
    st.subheader("Jelaskan custom instance (misal dari input manual)")
    if st.button("Explain custom first test instance"):
        custom_res = explain_custom_instance(st.session_state.pipeline_results,
                                            st.session_state.pipeline_results['lime_data']['X_test'].iloc[0].values,
                                            num_features=num_features)
        if custom_res:
            st.write(f"- Predicted: **{custom_res['predicted_class']}**, Confidence: {custom_res['confidence']:.3f}")
            st.dataframe(custom_res['explanation_data']['explanation_df'][['Feature', 'Importance', 'Impact']])
        else:
            st.error("Gagal menjelaskan custom instance.")

def execute_full_flow(df):
    """Orchestrate pipeline and populate session state"""
    pipeline_params = {
        'df': df,
        'resampling_method': st.session_state.selected_resampling,
        'training_mode': st.session_state.selected_training_mode,
        'random_state': 42
    }
    if st.session_state.selected_training_mode == 'manual':
        pipeline_params.update({
            'hidden_neurons': st.session_state.selected_hidden_neurons,
            'activation': st.session_state.selected_activation,
            'threshold': st.session_state.selected_threshold
        })
    else:
        pipeline_params.update({
            'n_trials': st.session_state.optuna_trials
        })

    results = run_complete_pipeline(**pipeline_params)

    if results is None:
        return None, "Pipeline gagal dijalankan."

    valid, msg = validate_pipeline_results(results)
    if not valid:
        return None, f"Validasi gagal: {msg}"

    explainer = create_lime_explainer_from_pipeline(results)
    if explainer is None:
        return None, "Gagal membuat LIME explainer."

    st.session_state.pipeline_results = results
    st.session_state.lime_explainer = explainer
    st.session_state.processing_complete = True

    return results, None

# === Navigation UI (sidebar) ===
def render_sidebar():
    st.markdown("### 🧭 Navigation")
    pages = {
        'upload': '📤 Upload Data',
        'process': '🔧 Preprocessing',
        'analysis': '📊 Analisis',
        'explanation': '🔍 LIME'
    }
    for key, label in pages.items():
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            if key == 'process' and st.session_state.uploaded_data is None:
                st.error("Upload data dulu dulu.")
                continue
            if key in ['analysis', 'explanation'] and not st.session_state.processing_complete:
                st.error("Jalankan pipeline terlebih dahulu.")
                continue
            st.session_state.current_page = key
            st.experimental_rerun()

def show_progress():
    st.markdown("### 🔁 Progress")
    items = [
        ("Data Uploaded", st.session_state.uploaded_data is not None),
        ("Pipeline Done", st.session_state.processing_complete),
        ("LIME Ready", st.session_state.lime_explainer is not None)
    ]
    for name, done in items:
        st.write(f"{'✅' if done else '⏳'} {name}")

# === Main ===
def main():
    st.set_page_config(page_title="🛡️ Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Fraud Detection System")
        render_sidebar()
        st.markdown("---")
        st.subheader("🛠️ Configuration")
        st.write(f"- Resampling: **{st.session_state.selected_resampling}**")
        st.write(f"- Training mode: **{st.session_state.selected_training_mode}**")
        if st.session_state.selected_training_mode == 'manual':
            st.write(f"- Hidden neurons: {st.session_state.selected_hidden_neurons}")
            st.write(f"- Activation: {st.session_state.selected_activation}")
            st.write(f"- Threshold: {st.session_state.selected_threshold}")
        else:
            st.write(f"- Optuna trials: {st.session_state.optuna_trials}")
        st.markdown("---")
        show_progress()

    # Content area routing
    if st.session_state.current_page == 'upload':
        page_upload()
    elif st.session_state.current_page == 'process':
        page_process()
    elif st.session_state.current_page == 'analysis':
        page_analysis()
    elif st.session_state.current_page == 'explanation':
        page_explanation()
    else:
        st.info("Pilih menu di sidebar untuk mulai.")

    # Footer integration test trigger
    st.markdown("---")
    if st.button("🧪 Jalankan Integration Test (dummy)"):
        if st.session_state.uploaded_data is not None:
            success = test_complete_integration(st.session_state.uploaded_data)
            if success:
                st.success("✅ Semua test integrasi lulus.")
            else:
                st.error("❌ Ada kegagalan di test integrasi. Cek log di konsol.")
        else:
            st.warning("Unggah data dulu untuk dites.")

if __name__ == "__main__":
    main()
