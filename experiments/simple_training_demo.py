#!/usr/bin/env python3
"""
Simple MLflow training demo for Red Hat OpenShift AI.

This script demonstrates basic MLflow experiment tracking:
- Creating experiments
- Logging parameters and metrics
- Simulating a training loop

Usage:
    python simple_training_demo.py
"""

import random
import time
import mlflow
from datetime import datetime

# Set experiment
mlflow.set_experiment("simple-demo")

# Start run with timestamp
run_name = f"run-{datetime.now().strftime('%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    print(f"Started run: {run.info.run_id}")

    # Log parameters
    mlflow.log_param("model", "simple_classifier")
    mlflow.log_param("epochs", 5)
    mlflow.log_param("lr", 0.01)

    # Simulate training
    print("Training...")
    for epoch in range(5):
        # Realistic improving metrics
        accuracy = 0.7 + (epoch * 0.05) + random.uniform(-0.02, 0.02)
        loss = 0.6 - (epoch * 0.1) + random.uniform(-0.05, 0.05)

        # Ensure bounds
        accuracy = max(0.6, min(0.95, accuracy))
        loss = max(0.1, loss)

        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch)

        print(f"  Epoch {epoch+1}: acc={accuracy:.3f}, loss={loss:.3f}")
        time.sleep(0.5)

    print(f"Done! Final accuracy: {accuracy:.3f}")
    print(f"Run ID: {run.info.run_id}")
