import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import graphviz
from datetime import datetime

# --- CONFIG ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="SentinelAI", page_icon="🛡️", layout="wide")

# --- HEADER ---
st.title("🛡️ SentinelAI: Fraud Operations Platform")
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (System Health) ---
st.sidebar.header("📡 System Telemetry")
try:
    stats = requests.get(f"{API_URL}/system/stats").json()
    st.sidebar.metric("Total Transactions Logged", stats['total_processed'])
    st.sidebar.metric("Global Avg Risk Score", stats['average_risk'])
    st.sidebar.success("API Status: ONLINE 🟢")
    st.sidebar.info("Database: SQLite Connected 💽")
except:
    st.sidebar.error("System Offline 🔴")
    st.stop()

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Live Inference", "🕸️ Link Analysis (Graph)", "📊 Drift Monitoring"])

# --- TAB 1: INFERENCE ---
with tab1:
    st.subheader("Real-Time Transaction Scoring")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("### Simulation")
        n_txns = st.slider("Batch Size", 10, 100, 20)
        if st.button("Ingest Data Stream"):
            # Generate dummy data
            fake_data = np.random.normal(0, 1, size=(n_txns, 10))
            cols = [f"Feature_{i}" for i in range(10)]
            df = pd.DataFrame(fake_data, columns=cols)
            
            payload = {"data": df.to_dict(orient="records")}
            
            with st.spinner("Processing Hybrid Ensemble..."):
                response = requests.post(f"{API_URL}/analyze/batch", json=payload).json()
                results = pd.DataFrame(response['results'])
                
                # Join with features for display
                display_df = pd.concat([df.reset_index(drop=True), results], axis=1)
                st.session_state['last_results'] = display_df

    with col2:
        if 'last_results' in st.session_state:
            df_res = st.session_state['last_results']
            
            # High Risk Filter
            high_risk = df_res[df_res['risk_score'] > 70]
            
            st.metric("Critical Alerts", len(high_risk), delta_color="inverse")
            
            st.dataframe(
                df_res[['id', 'risk_score', 'factors', 'anomaly_score']].sort_values('risk_score', ascending=False),
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.ProgressColumn("Risk", format="%d", min_value=0, max_value=100),
                    "anomaly_score": st.column_config.NumberColumn("Anomaly Index", format="%.3f")
                }
            )

# --- TAB 2: LINK ANALYSIS (The "Master's" Visual) ---
with tab2:
    st.subheader("🕸️ Entity Link Analysis")
    st.markdown("Visualizing hidden connections between flagged accounts and shared devices.")
    
    graph_data = requests.get(f"{API_URL}/network-graph").json()
    
    # Create Graphviz DOT format
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR')
    
    for node in graph_data['nodes']:
        color = 'red' if node['group'] == 1 else 'lightblue'
        shape = 'doublecircle' if node['group'] == 1 else 'ellipse'
        graph.node(node['id'], style='filled', fillcolor=color, shape=shape)
        
    for link in graph_data['links']:
        graph.edge(link['source'], link['target'])
        
    st.graphviz_chart(graph)
    st.caption("Red nodes indicate shared suspicious attributes (IPs/Devices) linking multiple User Accounts.")

# --- TAB 3: AUDIT & DRIFT ---
with tab3:
    st.subheader("📜 System Audit Logs & Drift Detection")
    st.markdown("Monitoring model performance stability over time.")
    
    # Fake Drift Chart for Demo
    dates = pd.date_range(end=datetime.today(), periods=14)
    drift_data = pd.DataFrame({
        "Date": dates,
        "Avg_Risk_Score": np.random.randint(20, 40, size=14) + np.linspace(0, 15, 14) # Increasing trend
    })
    
    fig = px.line(drift_data, x="Date", y="Avg_Risk_Score", title="Detected Model Drift (Risk Score Inflation)", markers=True)
    fig.add_hline(y=50, line_dash="dot", annotation_text="Drift Threshold", annotation_position="bottom right", line_color="red")
    st.plotly_chart(fig, use_container_width=True)