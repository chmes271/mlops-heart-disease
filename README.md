# End-to-End MLOps Pipeline: UCI Heart Disease Risk Prediction - Vishal Bandhu 2025AE0587

![Continuous Integration Pipeline](https://github.com/chmes271/mlops-heart-disease/actions/workflows/ci_cd.yml/badge.svg)
![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2.svg)

An enterprise-grade, cloud-native MLOps pipeline built on the **UCI Heart Disease Cleveland Dataset**. This project encapsulates the complete machine learning lifecycle—from automated data acquisition and statistical feature preprocessing to experiment tracking, API deployment, containerization, CI/CD automation, Kubernetes orchestration, and Prometheus monitoring.

---

## 📽️ Video Walkthrough & Project Demo


* **Report Document:** Available under `MLOps_Heart_Disease_Assignment_Report.docx` and `mlops_submission_report.html`.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------------+
|                                SYSTEM ARCHITECTURE DIAGRAM                              |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [UCI Repo] ---> [src/data_loader.py] ---> [src/preprocess.py Pipeline]                 |
|                                                        |                                |
|                                                        v                                |
|                                            [src/train.py Training Engine]               |
|                                                        |                                |
|                     +----------------------------------+--------------------------------+
|                     |                                  |                                |
|                     v                                  v                                |
|            [MLflow Experiment]                [models/best_model.pkl]                   |
|           (sqlite:///mlflow.db)                        |                                |
|                                                        v                                |
|                                            [FastAPI App (app/main.py)]                  |
|                                                        |                                |
|                     +----------------------------------+--------------------------------+
|                     |                                  |                                |
|                     v                                  v                                |
|            [GitHub Actions CI/CD]            [Docker Container (Port 8000)]             |
|           (Lint -> Test -> Build)                      |                                |
|                                                        v                                |
|                                           [Kubernetes Cluster (k8s/)]                   |
|                                       (Deployment Replicas=2, LoadBalancer)             |
|                                                        |                                |
|                                                        v                                |
|                                           [Prometheus /metrics Telemetry]               |
+-----------------------------------------------------------------------------------------+
```

---

## 🧰 Tech Stack & Tools

* **Data Processing & ML:** Python 3.10, Scikit-Learn, Pandas, NumPy, Joblib, Cloudpickle
* **Experiment Tracking:** MLflow (SQLite backend tracking store & artifact repository)
* **API Development:** FastAPI, Uvicorn, Pydantic v2
* **Unit Testing & Quality:** Pytest, Flake8
* **CI/CD Automation:** GitHub Actions
* **Containerization & Orchestration:** Docker (`python:3.10-slim`), Kubernetes (`kubectl`)
* **Observability:** Prometheus Metrics Telemetry (`prometheus_fastapi_instrumentator`)

---

## 📁 Repository Structure

```text
mlops-heart-disease/
├── .github/
│   └── workflows/
│       └── ci_cd.yml            # GitHub Actions CI/CD pipeline definition
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI REST server & Prometheus instrumentation
│   └── schemas.py               # Pydantic v2 input/output validation models
├── data/
│   └── raw/
│       └── heart.csv            # Auto-downloaded UCI Heart Disease dataset
├── docs/
│   └── screenshots/             # Pipeline execution & dashboard evidence
│       ├── eda_plots.png
│       ├── mlflow_ui.png
│       ├── pytest_results.png
│       ├── docker_run.png
│       ├── github_actions.png
│       ├── k8s_deployment.png
│       └── prometheus_metrics.png
├── k8s/
│   ├── deployment.yaml          # Kubernetes Deployment (2 replicas, health probes)
│   └── service.yaml             # Kubernetes LoadBalancer Service (Port 80 -> 8000)
├── models/
│   └── best_model.pkl           # Best serialized Random Forest pipeline artifact
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Automated dataset downloader
│   ├── preprocess.py            # ColumnTransformer feature preprocessing pipeline
│   └── train.py                 # Multi-model evaluation & MLflow tracking script
├── tests/
│   ├── __init__.py
│   ├── test_api.py              # Pytest unit tests for FastAPI endpoints
│   └── test_model.py            # Pytest unit tests for data fetching & transformations
├── Dockerfile                   # Multi-stage container build spec
├── generate_screenshots.py     # Asset generator script for documentation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/chmes271/mlops-heart-disease.git
cd mlops-heart-disease

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Acquisition, Training & MLflow Tracking

Run model training to download raw data, run cross-validation, and log experiments to MLflow:

```bash
python -m src.train
```

To launch the MLflow UI and view model metrics:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
*Access MLflow at `http://localhost:5000`.*

### 3. Run Unit Test Suite

Execute automated unit tests to verify data pipelines and API contracts:

```bash
pytest tests/ -v
```

---

## 🐳 Containerization & Local Deployment

### Build & Run Docker Image

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container locally
docker run -d -p 8000:8000 --name heart-api-test heart-disease-api:latest
```

### Test Local Container Endpoint

```bash
# Health Check
curl http://localhost:8000/health

# Prediction Test
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250, 
       "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0, 
       "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 3
     }'
```

---

## ☸️ Kubernetes Orchestration

Deploy the containerized model to a local Kubernetes cluster (Docker Desktop Kubernetes or Minikube):

```bash
# Apply deployment and service manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify cluster resources
kubectl get pods
kubectl get svc
```

### Test Kubernetes Service

```bash
curl -X POST "http://localhost:80/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250, 
       "fbs": 0, "restecg": 1, "thalach": 140, "exang": 0, 
       "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 3
     }'
```

---

## 📊 API Endpoints & Observability

| Environment | Base URL | Health Check (`GET`) | Inference (`POST`) | Prometheus Telemetry (`GET`) |
| :--- | :--- | :--- | :--- | :--- |
| **Kubernetes Cluster** | `http://localhost:80` | `/health` | `/predict` | `/metrics` |
| **Docker Container** | `http://localhost:8000` | `/health` | `/predict` | `/metrics` |

### Sample Prediction Payload & Response

**Request (`POST /predict`):**
```json
{
  "age": 55.0,
  "sex": 1.0,
  "cp": 3.0,
  "trestbps": 130.0,
  "chol": 250.0,
  "fbs": 0.0,
  "restecg": 1.0,
  "thalach": 140.0,
  "exang": 0.0,
  "oldpeak": 1.2,
  "slope": 2.0,
  "ca": 0.0,
  "thal": 3.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.19761012528139382,
  "risk_level": "Low"
}
```

---

## 🔄 Continuous Integration (CI/CD)

The repository includes an automated GitHub Actions workflow (`.github/workflows/ci_cd.yml`) executing:
1. **Linting:** Flake8 code quality checks.
2. **Model Pipeline:** Retrains and validates ML pipeline execution.
3. **Unit Tests:** Runs full Pytest suite for API & model logic.
4. **Container Build:** Compiles and validates Docker container image build.

---

## 📈 Model Performance Benchmark

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 85.25% | 0.8214 | 0.8519 | 0.8364 | 0.9120 |
| **Random Forest (Champion)** | **86.89%** | **0.8333** | **0.8929** | **0.8621** | **0.9470** |

---
