# MLflow Experiments Demo

This folder contains simple examples for **MLflow experiment tracking** on Red Hat OpenShift AI.

## Overview

These demos show basic MLflow functionality:
- Creating and managing experiments
- Logging parameters and hyperparameters
- Tracking metrics during training
- Viewing results in MLflow UI

## Prerequisites

- You are logged in to OpenShift with `oc login`
- Your account has permission to `experiments`, `registered_models`, and `jobs` resources in the `mlflow.kubeflow.org` API group
  - These permissions are automatically granted to the `admin` and `edit` OpenShift roles
- **Note:** Service accounts are not supported due to [RHOAIENG-42019](https://issues.redhat.com/browse/RHOAIENG-42019)

## Installation

Create a new Python virtualenv and install the MLflow SDK:

```bash
python -m venv mlflow-venv
source mlflow-venv/bin/activate

# For RHOAI 3.2 or 3.3
pip install "git+https://github.com/red-hat-data-services/mlflow@rhoai-3.3"
```

## Quick Start

### 1. Set Environment Variables

```bash
# Get your OpenShift token
export MLFLOW_TRACKING_TOKEN=$(oc whoami --show-token)

# Set MLflow tracking URI
DS_GW=$(oc get route data-science-gateway -n openshift-ingress -o template --template='{{.spec.host}}')
export MLFLOW_TRACKING_URI="https://$DS_GW/mlflow"

# Set workspace (namespace)
export MLFLOW_WORKSPACE=ai-bu-shared

# For self-signed certificates
export MLFLOW_TRACKING_INSECURE_TLS=true
```

### 2. Run the Demo

```bash
python simple_training_demo.py
```

### 3. View Results

Open the MLflow UI (available in the RHOAI dashboard Applications menu) to see:
- Experiment runs
- Parameters logged
- Metrics over time
- Run comparisons

## Files

| File | Description |
|------|-------------|
| `simple_training_demo.py` | Basic training loop with metric logging |

## What Gets Logged

### Parameters
- `model`: Model type identifier
- `epochs`: Number of training epochs
- `lr`: Learning rate

### Metrics (per epoch)
- `accuracy`: Model accuracy (simulated)
- `loss`: Training loss (simulated)

## Next Steps

After understanding basic experiment tracking, explore the **agent-tracing** folder for:
- AI agent tracing with MLflow
- LangChain/LangGraph integration
- Automatic trace capture with autolog
