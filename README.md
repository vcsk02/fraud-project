# SentinelAI – Enterprise Fraud Detection Platform

Course: CMPE 256
Date: November 29, 2025

## 1. Executive Summary

SentinelAI is a production-style Fraud Detection and Recommendation Engine designed to bridge the gap between traditional rule-based monitoring and modern AI-driven risk scoring.

Instead of a simple “Fraud / Safe” label, SentinelAI acts as an analyst copilot: it assigns a continuous risk score to each transaction, surfaces explanations, and keeps a full audit trail suitable for real banking workflows.

The system uses a Hybrid Ensemble Architecture deployed via a decoupled microservices stack:

Supervised Learning – Random Forest
Learns known fraud patterns from labeled historical data.

Unsupervised Learning – Isolation Forest
Detects novel anomalies and “zero-day” behaviors in feature space.

Key capabilities:

Real-time risk scoring via FastAPI

Interactive dashboard for fraud ops teams (Streamlit)

Explainability via SHAP

Persistent audit logging and telemetry

## 2. Problem Statement

Financial fraud detection faces three core challenges that SentinelAI targets:

Extreme Class Imbalance
Genuine fraud is rare (often < 0.2%), which makes naive classifiers biased toward “non-fraud” and unreliable in production.

Black-Box Decisioning
Complex ML models often cannot articulate why a transaction was flagged. This hurts analyst trust and fails regulatory expectations.

Static Detection Rules
Attackers adapt; systems that rely only on historical rules/signatures struggle to catch emerging patterns and coordinated fraud rings.

## 3. Technical Solution & Architecture

SentinelAI tackles these issues with a three-pillared design.

### 3.1 Hybrid Detection Engine

A weighted ensemble balances precision and recall:

Supervised Layer – Random Forest

Trained on labeled historical data.

Optimized to capture known fraud behaviors.

Unsupervised Layer – Isolation Forest

Operates on the same feature space.

Flags statistical outliers and suspicious “unknown unknowns.”

Imbalance Handling – SMOTE

Uses Synthetic Minority Over-sampling Technique (SMOTE) to rebalance the training data before fitting the supervised model.

### 3.2 Microservices Architecture

The system is structured to resemble a small, real-world banking stack:

Component	Technology	Role
Backend – “Brain”	FastAPI	Ingestion, model inference, risk scoring, SHAP computation, logging
Frontend – “Face”	Streamlit	Analyst dashboard, live scoring controls, link analysis, KPIs
Persistence Layer	SQLite	Transaction logs, audit trails, and model/telemetry records

This separation keeps the ML logic, UI, and storage clean and independently deployable.

### 3.3 Explainable AI (XAI)

For flagged or high-risk transactions, SentinelAI computes SHAP values (Shapley Additive Explanations):

Quantifies each feature’s contribution to the final risk score
(e.g., Transaction Amount, Time of Day, Merchant Category, Device Fingerprint).

Helps analysts answer:

“What exactly made this look suspicious?”

This turns a raw probability into an actionable narrative.

## 4. Key Features
### A. Risk Scoring & Recommendations

Instead of a binary label, SentinelAI outputs a Risk Score from 0–100:

High-Risk → Prioritized queue for manual review

Medium-Risk → Optional stepped-up verification (2FA, KYC checks)

Low-Risk → Auto-approved, reducing analyst workload

Thresholds can be tuned by risk appetite.

### B. Link Analysis (Graph-Style View)

Fraud is often networked, not isolated. The dashboard’s link-analysis view highlights:

Multiple user accounts sharing a single IP or device ID

Repeated charge attempts across different cards with shared attributes

Suspicious clusters that might indicate fraud rings or account takeovers

This helps analysts move beyond row-by-row inspection into graph-level reasoning.

### C. Model Persistence & MLOps Hooks

Serialization with joblib for both models and preprocessing artifacts

Metric computation (Precision, Recall, F1, etc.) after each training run

Designed to plug into a future CI/CD or MLOps pipeline for:

scheduled retraining

model versioning

monitoring for model drift

## 5. Experimental Results

The models were evaluated on a held-out test set, with a focus on:

High Recall → minimize undetected fraud

Reasonable Precision → avoid drowning analysts in false positives

F1-Score → balance between the two

ROC-AUC / PR-AUC → robust view under heavy class imbalance

Full details (metrics tables, confusion matrices, and SHAP plots) are documented in:

SentinelAI Project Report.pdf

Any associated notebooks in the repository

(The old placeholder “0.00%” metrics have been removed; always refer to the report/notebooks for the true final scores.)

## 6. Installation & Running Locally
### 6.1 System Requirements

Python: 3.9+

Core Libraries (see requirements.txt):

FastAPI

Uvicorn

Streamlit

Scikit-learn

Pandas / NumPy

SHAP

SQLAlchemy

Joblib

Install all dependencies with:

pip install -r requirements.txt

### 6.2 Backend – FastAPI Service

Navigate to the backend folder:

cd backend


Start the FastAPI app with Uvicorn:

Replace `main:app` with your actual entrypoint if different
python -m uvicorn main:app --reload


By default this serves the API at:

http://127.0.0.1:8000


Open the interactive API docs at:

Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc (if enabled)

From here you can send test requests to the prediction endpoint (e.g. /score or /predict, depending on the implementation).

### 6.3 Frontend – Streamlit Dashboard

In a new terminal, from the project root:

cd frontend


Launch the dashboard:

# Replace `app.py` with the actual Streamlit file if needed
python -m streamlit run app.py


Open the Streamlit URL (usually):

http://localhost:8501


You should now see the SentinelAI: Enterprise Fraud Defense Platform UI with:

System Telemetry & Model Health sidebar

Real-Time Transaction Scoring Engine

Tabs for Live Operations, Link Analysis, Model Performance, and Audit Logs

### 6.4 Quick Demo Flow

A typical manual demo:

Ensure backend (FastAPI) is running.

Open the Streamlit dashboard in your browser.

In the Real-Time Transaction Scoring Engine:

Adjust Batch Size.

Click Inject Live Transactions to simulate a batch.

Watch the System Telemetry update:

Total Transactions Analyzed

Current Avg Risk Score

API Status & Model Build Date

Explore:

Link Analysis to see suspicious clusters

Model Performance to review metrics

Audit Logs to inspect individual scored transactions

## 7. Project Structure
fraud-project/
├── backend/                     # FastAPI service: models, scoring, SHAP, logging
├── frontend/                    # Streamlit dashboard (Live Ops, Link Analysis, etc.)
├── SentinelAI Project Report.pdf
├── SentinelAI Fraud Detection.pptx
├── requirements.txt
└── README.md


You can extend the structure section once more modules/files are finalized.

## 8. Dataset

This project uses a highly imbalanced financial transaction dataset with labeled fraudulent vs. legitimate transactions.

High-level properties:

Tabular transaction records (amount, time, derived features, etc.)

Strong class imbalance (fraud << non-fraud)

Preprocessed for ML training and real-time inference

For full dataset details (source, preprocessing pipeline, feature list), refer to:

SentinelAI Project Report.pdf

Data preprocessing section of the notebooks / backend code

## 9. UI Preview

Steps:

Create a docs/ directory.

Save a PNG of the main dashboard as docs/dashboard.png.

Commit the image along with this README.

(When you actually paste this into your README, you can delete the “Add a screenshot…” explanatory text and just keep the image line.)

## 10. Future Work

Planned / possible extensions:

Integrate with a real message queue or event stream (Kafka, Kinesis, etc.)

Add model-drift and data-drift monitoring dashboards

Implement authentication / RBAC for analyst and admin personas

Deploy as containers (Docker + docker-compose / Kubernetes)

Experiment with sequence models (RNNs / Transformers) for temporal fraud patterns
