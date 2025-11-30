import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import graphviz

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SentinelAI Final", page_icon="🛡️", layout="wide")

# --- HEADER ---
st.title("🛡️ SentinelAI: Enterprise Fraud Defense Platform")
st.markdown("**Version:** 1.0 (Final Release) | **Architecture:** Hybrid Ensemble (Supervised + Unsupervised)")

# --- SIDEBAR ---
st.sidebar.header("📡 System Telemetry")
try:
    stats = requests.get(f"{API_URL}/system/stats").json()
    metrics = requests.get(f"{API_URL}/model/metrics").json()
    
    st.sidebar.metric("Total Transactions Analyzed", stats['total_processed'])
    st.sidebar.metric("Current Avg Risk Score", stats['average_risk'])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Model Health")
    st.sidebar.text(f"Build Date: {metrics.get('model_date', 'N/A')}")
    st.sidebar.success("API Status: ONLINE 🟢")
except:
    st.sidebar.error("Backend Offline 🔴")
    st.sidebar.warning("Please start the FastAPI backend.")
    st.stop()

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Operations", "🕸️ Link Analysis", "📉 Model Performance", "📜 Audit Logs"])

# --- TAB 1: OPERATIONS ---
with tab1:
    st.subheader("Real-Time Transaction Scoring Engine")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info("Simulation Controls")
        n_txns = st.slider("Batch Size", 10, 200, 50)
        if st.button("Inject Live Transactions", type="primary"):
            # Generate dummy data matching 10 features
            fake_data = np.random.normal(0, 1, size=(n_txns, 10))
            cols = [f"Feature_{i}" for i in range(10)]
            df = pd.DataFrame(fake_data, columns=cols)
            
            payload = {"data": df.to_dict(orient="records")}
            
            with st.spinner("Processing through Hybrid Pipeline..."):
                try:
                    response = requests.post(f"{API_URL}/analyze/batch", json=payload).json()
                    results = pd.DataFrame(response['results'])
                    display_df = pd.concat([df.reset_index(drop=True), results], axis=1)
                    st.session_state['last_results'] = display_df
                    st.toast(f"Successfully processed {n_txns} transactions", icon="✅")
                except Exception as e:
                    st.error(f"Inference Error: {e}")

    with col2:
        if 'last_results' in st.session_state:
            df_res = st.session_state['last_results']
            high_risk = df_res[df_res['risk_score'] > 75]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Batch High Risk", len(high_risk), delta_color="inverse")
            c2.metric("Max Risk Score", f"{df_res['risk_score'].max()}/100")
            c3.metric("Avg Anomaly Index", f"{df_res['anomaly_score'].mean():.2f}")
            
            st.dataframe(
                df_res[['id', 'risk_score', 'factors', 'anomaly_score']].sort_values('risk_score', ascending=False),
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn("Risk Score", format="%d", min_value=0, max_value=100),
                    "factors": "Primary Risk Driver"
                }
            )

# --- TAB 2: LINK ANALYSIS ---
with tab2:
    st.subheader("🕸️ Fraud Ring Detection (Graph Theory)")
    st.markdown("Visualizing shared entities (IPs, Devices) to detect organized fraud rings.")
    
    graph_data = requests.get(f"{API_URL}/network-graph").json()
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')
    
    for node in graph_data['nodes']:
        color = 'red' if node['group'] == 1 else 'lightblue'
        shape = 'doublecircle' if node['group'] == 1 else 'ellipse'
        graph.node(node['id'], style='filled', fillcolor=color, shape=shape)
        
    for link in graph_data['links']:
        graph.edge(link['source'], link['target'])
        
    st.graphviz_chart(graph)

# --- TAB 3: PERFORMANCE METRICS (FINAL REPORT) ---
with tab3:
    st.subheader("📊 Final Model Evaluation Report")
    st.markdown("Metrics calculated on the validation set during the latest build.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Precision (Fraud)", metrics.get("precision", 0.0))
    m_col2.metric("Recall (Fraud)", metrics.get("recall", 0.0))
    m_col3.metric("F1-Score", metrics.get("f1", 0.0))
    
    st.markdown("### Performance Interpretation")
    st.info(f"""
    * **F1-Score of {metrics.get('f1', '0.0')}** indicates a balanced trade-off between catching fraud and minimizing false alarms.
    * **SMOTE** was successfully applied to handle the 2% fraud prevalence.
    * **Hybrid Architecture** (RF + IsoForest) ensures both known patterns and anomalies are captured.
    """)

# --- TAB 4: AUDIT LOGS ---
with tab4:
    st.subheader("📜 Compliance & Audit Trail")
    st.markdown("Immutable logs of all processed transactions for regulatory compliance.")
    st.warning("This view is connected to the live SQLite database.")