SentinelAI – Enterprise Fraud Detection Platform

Course: CMPE 256
Version: 1.0.0 (Production Release)
Date: November 29, 2025

1. Executive Summary

SentinelAI is a production-grade Fraud Detection and Recommendation Engine designed to address the limitations of traditional rule-based financial monitoring.

Unlike standard binary classifiers that simply flag transactions as "Fraud" or "Safe", SentinelAI functions as an intelligent decision-support system for financial analysts.

The system utilizes a Hybrid Ensemble Architecture, combining:

Supervised Learning (Random Forest) to detect known fraud patterns

Unsupervised Learning (Isolation Forest) to identify novel anomalies

This logic is deployed via a decoupled Microservices Architecture:

FastAPI Backend

Streamlit Frontend

Key capabilities include:

Real-time scalability

Explainability via SHAP (Shapley Additive Explanations)

Operational transparency through audit logging

2. Problem Statement

Financial fraud detection faces three critical challenges:

Imbalanced Data
Genuine fraud is rare (often < 0.2%), making standard model training difficult and leading to biased classifiers.

Black Box Decisioning
Complex models often fail to explain why a transaction was flagged, causing trust issues and regulatory non-compliance.

Static Detection
Fraud patterns evolve rapidly; systems relying solely on historical data struggle to catch zero-day attacks and emerging threats.

3. Technical Solution & Architecture

SentinelAI addresses these challenges through a three-pillared technical approach.

3.1 Hybrid Detection Engine

To balance precision and recall, the system employs a weighted ensemble model:

Supervised Layer – Random Forest
Trained on historical fraud labels to catch known attack vectors.

Unsupervised Layer – Isolation Forest
Analyzes the feature space for statistical outliers, catching new or unknown fraud types.

Imbalance Handling – SMOTE
The training pipeline integrates SMOTE (Synthetic Minority Over-sampling Technique) to synthetically balance class distribution before model fitting.

3.2 Microservices Architecture

The system is engineered as a decoupled application to mimic real-world banking infrastructure:

Backend ("The Brain")

RESTful API built with FastAPI

Handles data ingestion, model inference, and database interactions

Uses asynchronous background tasks to log data without increasing transaction latency

Frontend ("The Face")

Interactive dashboard built with Streamlit

Provides visualizations, analyst queues, and system controls

Persistence Layer

SQLite database stores all transaction logs

Enables audit trails and model drift monitoring

3.3 Explainable AI (XAI)

The system integrates SHAP (Shapley Additive Explanations):

For every flagged transaction, the API calculates the marginal contribution of each feature (e.g., Transaction Amount, Location) to the final risk score.

Transforms the output from a simple probability into an actionable explanation, improving analyst trust and regulatory alignment.

4. Key Features
A. Risk Scoring & Recommendation

Instead of a binary output, the system generates a Risk Score (0–100).

High scores indicate high fraud likelihood.

The model functions as a recommender engine:

Prioritizes high-risk cases for immediate human review

Auto-approves low-risk transactions to reduce analyst load

B. Link Analysis (Graph Theory)

The platform includes a network visualization module:

Identifies hidden connections between seemingly unrelated accounts

Examples:

Multiple user IDs accessing the system from a single compromised IP address

Shared device identifiers across different accounts

This enables detection of organized fraud rings and collusive behaviors.

C. Model Persistence & MLOps

The system includes an automated MLOps-oriented pipeline:

Serializes trained models using joblib

Calculates performance metrics after every build:

Precision

Recall

F1-Score

Ensures the system is:

Production ready

Capable of rapid deployment and iterative improvement

5. Experimental Results

The model was evaluated on a held-out test set using the following key metrics:

Precision
Optimized to reduce false positives and reduce analyst fatigue.

Recall
Optimized to ensure maximum capture of fraudulent activities, especially rare cases.

F1-Score
Captures the balance between precision and recall, with a focus on the minority (fraud) class to ensure robust performance where it matters most.

(Exact numeric metrics can be added here once finalized.)

6. Installation & User Guide
6.1 System Requirements

Python: 3.9+

Core Libraries:

FastAPI

Streamlit

Scikit-learn

Pandas

SHAP

SQLAlchemy

Install dependencies (example):

pip install -r requirements.txt


(Assuming a requirements.txt file is provided.)

6.2 Execution Instructions

The system requires two concurrent terminal processes.

1. Backend Service
python -m uvicorn main:app --reload


Initializes the ML models

Exposes the API at: http://127.0.0.1:8000

2. Frontend Dashboard
python -m streamlit run app.py


Launches the user interface for analysts and operators

6.3 Operational Workflow

Simulation

Analysts use the "Live Operations" tab to inject transaction batches.

Review

High-risk transactions (Risk Score > 75) appear in the "Analyst Queue".

Investigation

Analysts:

Review the Primary Risk Driver (top SHAP feature contribution)

Examine the "Link Analysis" tab for relevant network connections.

Audit

All decisions (flags, approvals, overrides) are permanently recorded in the "Audit Logs" tab for traceability and compliance.

7. Conclusion

SentinelAI demonstrates that combining hybrid machine learning models with modern software engineering practices (Microservices, XAI) can significantly improve fraud detection capabilities in enterprise environments.

The system:

Moves beyond simple, opaque detection

Provides a comprehensive, explainable, and scalable solution for enterprise risk management

Supports human analysts rather than replacing them, functioning as a decision-support and recommendation engine for high-stakes financial operations