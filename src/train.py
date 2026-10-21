import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import fetch_data
from src.preprocess import build_preprocessor, prepare_data

def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

def train_and_track():
    # Direct MLflow to write into the local SQLite database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Heart_Disease_Classification")
    
 
    
    df = fetch_data()
    X, y = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": (
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            {"C": 1.0, "max_iter": 1000}
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"n_estimators": 100, "max_depth": 5}
        )
    }

    best_f1 = 0.0
    best_pipeline = None

    for model_name, (clf, params) in models.items():
        with mlflow.start_run(run_name=model_name):
            preprocessor = build_preprocessor()
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

            full_pipeline.fit(X_train, y_train)

            y_pred = full_pipeline.predict(X_test)
            y_proba = full_pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_proba)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # skops_trusted_types=True fixes skops serialization exception
            mlflow.sklearn.log_model(
    sk_model=full_pipeline,
    artifact_path="model",
    serialization_format="cloudpickle"
)

            print(f"=== {model_name} Metrics ===")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_pipeline = full_pipeline

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.pkl")
    print("\nBest model successfully saved to models/best_model.pkl")

if __name__ == "__main__":
    train_and_track()