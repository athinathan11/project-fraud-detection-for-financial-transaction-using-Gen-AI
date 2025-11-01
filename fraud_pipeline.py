#!/usr/bin/env python3
"""
Skeleton pipeline for fraud detection (prototype).
- offline feature generation
- train a simple LightGBM
- example of generating an LLM explanation (pseudocode)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve, auc
import joblib

# Placeholder imports (install lgb if used)
# import lightgbm as lgb

def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

def feature_engineer(df):
    # Example: rolling counts per account in last 24h (assuming sorted)
    df = df.sort_values(["account_id", "timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["amount_log"] = np.log1p(df["amount"])
    # simple velocity: transactions in last 24 hours per account (approx)
    df["txn_24h_count"] = df.groupby("account_id")["timestamp"].rolling("24H", on="timestamp").count().reset_index(0,drop=True).fillna(0)
    df["txn_24h_sum"] = df.groupby("account_id")["amount"].rolling("24H", on="timestamp").sum().reset_index(0,drop=True).fillna(0)
    df = df.fillna(0)
    return df

def train_model(df, features, label_col="is_fraud"):
    # Time-based split
    df = df.sort_values("timestamp")
    split = int(len(df)*0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    X_train = train[features]; y_train = train[label_col]
    X_test = test[features]; y_test = test[label_col]

    # Placeholder - replace with LightGBM/XGBoost training
    # dtrain = lgb.Dataset(X_train, label=y_train)
    # params = {"objective":"binary", "metric":"auc"}
    # model = lgb.train(params, dtrain, num_boost_round=100)
    # joblib.dump(model, "models/lgb_model.pkl")

    # For prototype, use a naive predictor
    preds = np.full(len(y_test), y_train.mean())
    precision, recall, _ = precision_recall_curve(y_test, preds)
    pr_auc = auc(recall, precision)
    print("Prototype PR-AUC:", pr_auc)
    # Save metadata
    joblib.dump({"features":features}, "models/metadata.pkl")

def explain_with_llm(case_features: dict):
    """
    Example pseudocode to call LLM to generate an explanation.
    WARNING: Do not send raw PII. Hash or redact sensitive fields.
    """
    # Build a safe prompt
    prompt = f\"\"\"You are a fraud analyst assistant. Given these anonymized features:
{case_features}
Provide:
1) A concise explanation of which features most likely contributed to fraud.
2) Suggested next investigation steps (3 bullet points).
3) Confidence level (low/medium/high) and why.
\"\"\"
    # call to LLM provider (openai/vertex/…)
    # response = llm_client.complete(prompt)
    # return response.text
    return "LLM explanation placeholder (implement LLM call separately, ensure PII redaction)"

def main(mode="train"):
    os.makedirs("models", exist_ok=True)
    df = load_data()
    df = feature_engineer(df)
    features = ["amount_log", "hour", "txn_24h_count", "txn_24h_sum"]
    if mode == "train":
        train_model(df, features)
    elif mode == "explain":
        sample = df.iloc[-1]
        case_feats = {f: float(sample[f]) for f in features}
        print(explain_with_llm(case_feats))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","explain"], default="train")
    args = parser.parse_args()
    main(args.mode)

```markdown name=evaluation_checklist.md    
    
