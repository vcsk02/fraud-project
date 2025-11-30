from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
import shap
from datetime import datetime
import os

# --- CONFIGURATION & DATABASE ---
app = FastAPI(title="SentinelAI: Enterprise Fraud Engine", version="2.0.0")

# Use absolute path for DB to avoid errors
db_path = os.path.join(os.path.dirname(__file__), 'fraud_logs.db')
db_engine = create_engine(f'sqlite:///{db_path}', echo=False)

# Initialize DB Table
with db_engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp DATETIME,
            transaction_id TEXT,
            risk_score FLOAT,
            model_confidence FLOAT,
            anomaly_score FLOAT,
            top_factor TEXT,
            is_flagged BOOLEAN
        )
    """))
    conn.commit()

# --- GLOBAL STATE ---
models = {}
N_FEATURES = 10  # STRICTLY ENFORCED FEATURE COUNT

# --- STARTUP: TRAINING & MLOPS ---
@app.on_event("startup")
def startup_event():
    print("⏳ System Startup: Initializing Models...")
    
    # 1. Generate Training Data (Strictly 10 Features to match Frontend)
    n = 5000
    # Generate columns Feature_0 to Feature_9
    X = np.random.normal(0, 1, (n, N_FEATURES))
    
    # Inject simple fraud pattern
    # Fraud cases (class 1) have higher values in Feature_1
    X[:100, 1] = X[:100, 1] + 5 
    
    y = np.zeros(n)
    y[:100] = 1 # Fraud
    
    cols = [f"Feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    
    # 2. Train Hybrid Models
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(df, y)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    rf.fit(X_res, y_res)
    
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(df)
    
    models['rf'] = rf
    models['iso'] = iso
    
    # Initialize SHAP explainer
    # We use a try-catch for the explainer initialization to be safe
    try:
        models['explainer'] = shap.TreeExplainer(rf)
    except Exception as e:
        print(f"⚠️ SHAP Explainer warning: {e}")
        models['explainer'] = None
        
    print(f"✅ System Online: Trained on {N_FEATURES} features.")

# --- UTILS ---
class TransactionBatch(BaseModel):
    data: list

def log_to_db(records):
    try:
        df = pd.DataFrame(records)
        df.to_sql('logs', db_engine, if_exists='append', index=False)
    except Exception as e:
        print(f"Database Log Error: {e}")

# --- ENDPOINTS ---
@app.post("/analyze/batch")
async def analyze_batch(batch: TransactionBatch, background_tasks: BackgroundTasks):
    # Convert input to DataFrame
    df = pd.DataFrame(batch.data)
    
    # --- CRASH PREVENTION CHECKS ---
    # 1. Ensure columns match training data
    expected_cols = [f"Feature_{i}" for i in range(N_FEATURES)]
    if len(df.columns) != N_FEATURES:
        # Force rename columns if count matches but names differ
        if len(df.columns) == N_FEATURES:
            df.columns = expected_cols
        else:
            # Emergency fallback: Slice or Pad
            df = df.iloc[:, :N_FEATURES] # Take first 10
            # If too few, pad with 0
            while len(df.columns) < N_FEATURES:
                df[f"Feature_{len(df.columns)}"] = 0

    # Hybrid Inference
    rf_probs = models['rf'].predict_proba(df)[:, 1]
    iso_scores = models['iso'].decision_function(df)
    iso_norm = (iso_scores.max() - iso_scores) / (iso_scores.max() - iso_scores.min())
    
    # Weighted Ensemble Scoring
    risk_scores = (0.65 * rf_probs) + (0.35 * iso_norm)
    final_scores = np.round(risk_scores * 100, 1)
    
    results = []
    db_logs = []
    
    # Safe SHAP Calculation
    try:
        if models['explainer']:
            shap_values = models['explainer'].shap_values(df)
            shap_vals_target = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            shap_vals_target = np.zeros(df.shape)
    except Exception:
        # If SHAP fails, fallback to zeros to prevent crash
        shap_vals_target = np.zeros(df.shape)

    for i, score in enumerate(final_scores):
        # SAFE INDEX LOOKUP
        try:
            top_feat_idx = np.argmax(np.abs(shap_vals_target[i]))
            if top_feat_idx < len(df.columns):
                top_feat = df.columns[top_feat_idx]
            else:
                top_feat = "Unknown"
        except:
            top_feat = "Feature_1" # Default fallback
            
        tid = f"TXN-{np.random.randint(10000, 99999)}"
        
        results.append({
            "id": tid,
            "risk_score": score,
            "factors": top_feat,
            "anomaly_score": float(iso_norm[i])
        })
        
        # Prepare DB Log
        db_logs.append({
            "timestamp": datetime.now(),
            "transaction_id": tid,
            "risk_score": score,
            "model_confidence": float(rf_probs[i]),
            "anomaly_score": float(iso_norm[i]),
            "top_factor": str(top_feat),
            "is_flagged": bool(score > 75)
        })

    # Async Write to DB
    background_tasks.add_task(log_to_db, db_logs)
    
    return {"status": "processed", "results": results}

@app.get("/network-graph")
def get_fraud_ring():
    nodes = [
        {"id": "Suspicious_IP_1", "group": 1},
        {"id": "User_A", "group": 2},
        {"id": "User_B", "group": 2},
        {"id": "Shared_Device_ID", "group": 1},
        {"id": "User_C", "group": 2}
    ]
    links = [
        {"source": "User_A", "target": "Suspicious_IP_1"},
        {"source": "User_B", "target": "Suspicious_IP_1"},
        {"source": "User_B", "target": "Shared_Device_ID"},
        {"source": "User_C", "target": "Shared_Device_ID"}
    ]
    return {"nodes": nodes, "links": links}

@app.get("/system/stats")
def get_stats():
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT AVG(risk_score), COUNT(*) FROM logs")).fetchone()
            # Handle case where DB is empty
            avg_risk = result[0] if result and result[0] is not None else 0.0
            count = result[1] if result and result[1] is not None else 0
        return {"total_processed": count, "average_risk": round(avg_risk, 2)}
    except Exception as e:
        print(f"Stats Error: {e}")
        return {"total_processed": 0, "average_risk": 0.0}