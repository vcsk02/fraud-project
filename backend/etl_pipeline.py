import sqlite3
import pandas as pd
import os
from datetime import datetime

# CONFIGURATION
DB_PATH = 'fraud_logs.db'
WAREHOUSE_PATH = 'processed_data_warehouse'
REPORT_FILE = 'daily_fraud_report.csv'

def extract_data():
    """EXTRACT: Read raw logs from the transactional database."""
    print("1. [EXTRACT] Connecting to OLTP Database...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database {DB_PATH} not found. Run the API first.")
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM logs"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"   -> Extracted {len(df)} raw transaction logs.")
    return df

def transform_data(df):
    """TRANSFORM: Clean, Aggregate, and Engineer Features."""
    print("2. [TRANSFORM] Processing Data...")
    
    # 1. Type Conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Feature Engineering: Hourly Window
    df['hour'] = df['timestamp'].dt.hour
    
    # 3. Aggregation: Calculate Fraud Velocity (Transactions per Hour)
    velocity_df = df.groupby('hour').size().reset_index(name='txn_count')
    df = df.merge(velocity_df, on='hour', how='left')
    
    # 4. Aggregation: Average Risk per Hour
    risk_profile = df.groupby('hour')['risk_score'].mean().reset_index(name='avg_hourly_risk')
    df = df.merge(risk_profile, on='hour', how='left')
    
    # 5. Filtering: Keep only high-value columns for the Data Warehouse
    warehouse_df = df[[
        'timestamp', 'transaction_id', 'risk_score', 
        'top_factor', 'avg_hourly_risk', 'txn_count', 'is_flagged'
    ]]
    
    print("   -> Calculated Fraud Velocity and Hourly Risk Aggregates.")
    return warehouse_df

def load_data(df):
    """LOAD: Save to Data Warehouse (CSV/Parquet) for Analytics."""
    print("3. [LOAD] Saving to Data Warehouse...")
    
    if not os.path.exists(WAREHOUSE_PATH):
        os.makedirs(WAREHOUSE_PATH)
        
    output_path = os.path.join(WAREHOUSE_PATH, REPORT_FILE)
    
    # Append if exists, else write new
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"   -> Appended data to {output_path}")
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)
        print(f"   -> Created new Warehouse file at {output_path}")

def run_pipeline():
    print(f"--- STARTING ETL JOB: {datetime.now()} ---")
    try:
        raw_data = extract_data()
        if not raw_data.empty:
            clean_data = transform_data(raw_data)
            load_data(clean_data)
            print("--- ETL JOB SUCCESSFUL ---")
        else:
            print("--- ETL SKIPPED (No Data) ---")
    except Exception as e:
        print(f"!!! ETL FAILED: {e}")

if __name__ == "__main__":
    run_pipeline()