SentinelAI – Enterprise Fraud Detection Platform
Course: CMPE 256
Submission Type: Final Capstone Project
Date: November 29, 2025
Team Name: [Insert Your Team Name]
1. Executive Summary
SentinelAI is a production-grade Fraud Detection and Recommendation Engine designed to bridge the gap between traditional rule-based financial monitoring and modern AI capabilities. Unlike standard binary classifiers that simply flag transactions as "Fraud" or "Safe," SentinelAI functions as an intelligent decision-support system for financial analysts.
The system utilizes a Hybrid Ensemble Architecture logic deployed via a decoupled Microservices Architecture:
Supervised Learning (Random Forest): Detects known historical fraud patterns.
Unsupervised Learning (Isolation Forest): Identifies novel anomalies and zero-day threats.
Key capabilities include real-time scalability, deep explainability via SHAP, and full operational transparency through audit logging.
2. Problem Statement
Financial fraud detection currently faces three critical challenges that SentinelAI aims to solve:
Imbalanced Data: Genuine fraud is rare (often < 0.2%), making standard model training difficult and leading to biased classifiers.
Black Box Decisioning: Complex models often fail to explain why a transaction was flagged, causing trust issues and regulatory non-compliance.
Static Detection: Fraud patterns evolve rapidly; systems relying solely on historical data struggle to catch emerging threats.
3. Technical Solution & Architecture
SentinelAI addresses these challenges through a three-pillared technical approach.
3.1 Hybrid Detection Engine
To balance precision and recall, the system employs a weighted ensemble model:
Supervised Layer: Random Forest trained on historical labels.
Unsupervised Layer: Isolation Forest analyzes feature space for statistical outliers.
Imbalance Handling: The training pipeline integrates SMOTE (Synthetic Minority Over-sampling Technique) to synthetically balance class distribution before model fitting.
3.2 Microservices Architecture
The system is engineered as a decoupled application to mimic real-world banking infrastructure:
Component
Technology
Function
Backend ("The Brain")
FastAPI
Handles data ingestion, model inference, and database interactions via async background tasks.
Frontend ("The Face")
Streamlit
Interactive dashboard for visualizations, analyst queues, and system controls.
Persistence Layer
SQLite
Stores transaction logs, audit trails, and model drift monitoring data.

3.3 Explainable AI (XAI)
The system integrates SHAP (Shapley Additive Explanations). For every flagged transaction, the API calculates the marginal contribution of each feature (e.g., Transaction Amount, Location) to the final risk score, transforming a probability into an actionable explanation.
4. Key Features
A. Risk Scoring & Recommendation
Instead of a binary output, the system generates a Risk Score (0–100).
High Risk: Prioritized for immediate human review.
Low Risk: Auto-approved to reduce analyst load.
B. Link Analysis (Graph Theory)
The network visualization module identifies hidden connections between seemingly unrelated accounts, such as:
Multiple User IDs accessing the system from a single compromised IP.
Shared device identifiers across different accounts.
C. Model Persistence & MLOps
Serialization: Models saved using joblib.
Metrics: Automated calculation of Precision, Recall, and F1-Score after every build.
Production Ready: Designed for rapid deployment and iterative improvement.
5. Experimental Results
The model was evaluated on a held-out test set. We optimized for Recall to ensure maximum capture of fraudulent activities, while maintaining high Precision to reduce analyst fatigue.
Metric
Score
Description
Precision
0.00%
Optimized to reduce false positives.
Recall
0.00%
Optimized to capture rare fraud cases.
F1-Score
0.00%
Harmonic mean focused on the minority class.

(Note: Replace 0.00% with your final finalized metrics.)
6. Installation & User Guide
6.1 System Requirements
Python: 3.9+
Core Libraries: FastAPI, Streamlit, Scikit-learn, Pandas, SHAP, SQLAlchemy.
Install dependencies:
pip install -r requirements.txt


6.2 Execution Instructions
The system requires two concurrent terminal processes.
1. Start the Backend Service:
Initializes ML models and exposes the API at http://127.0.0.1:8000.
python -m uvicorn main:app --reload


2. Start the Frontend Dashboard:
Launches the UI for analysts.
python -m streamlit run app.py


6.3 Operational Workflow
Simulation: Go to the "Live Operations" tab to inject transaction batches.
Review: High-risk transactions (Risk Score > 75) appear in the "Analyst Queue".
Investigation:
Review the Primary Risk Driver (SHAP analysis).
Check the "Link Analysis" tab for network connections.
Audit: All decisions (flags, approvals, overrides) are permanently recorded in the "Audit Logs".
7. Conclusion
SentinelAI demonstrates that combining hybrid machine learning models with modern software engineering practices (Microservices, XAI) can significantly improve fraud detection capabilities in enterprise environments. By moving beyond opaque detection, SentinelAI functions as a scalable decision-support engine for high-stakes financial operations.
© 2025 [Insert Team Name]. All Rights Reserved.
