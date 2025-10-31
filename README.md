# project-fraud-detection-for-financial-transaction-using-Gen-AI
# Fraud Detection for Financial Transactions using GenAI + ML

This repository contains a starter blueprint for building a fraud detection system that combines classical ML, graph models, and Generative AI (LLMs) for feature engineering, synthetic data generation, and analyst assistance.

Contents:
- architecture.md: high-level architecture and components.
- fraud_pipeline.py: pipeline skeleton for feature creation, model training, and inference.
- prompt_templates.md: safe prompt templates for LLM-based assistant tasks (explanations, triage, synthetic generation).
- evaluation_checklist.md: evaluation, monitoring, and deployment checklist.

Security note: Do not send raw PII or unredacted identifiers to third-party LLM APIs. Use hashing/pseudonymization.

Getting started:
1. Populate data in data/transactions.csv (example format described in architecture.md).
2. Configure feature store & model infra (see notes in architecture.md).
3. Run `python fraud_pipeline.py --mode train` to run an end-to-end local prototype.
