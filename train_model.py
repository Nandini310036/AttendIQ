from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
try:
    from .data_processing import load_data, build_student_features, export_cleaned_data
except ImportError:
    from data_processing import load_data, build_student_features, export_cleaned_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def train():
    students, attendance = load_data()
    export_cleaned_data(students, attendance)
    features, cutoff_date = build_student_features(students, attendance)
    feature_cols = ["department", "year", "early_classes", "early_present", "early_absent", "early_attendance_pct"]
    X = features[feature_cols]
    y = features["at_risk"]

    if y.nunique() < 2:
        raise ValueError("The generated dataset needs examples of both at-risk and not-at-risk students.")

    categorical = ["department"]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    }
    results = []
    best_name, best_pipe, best_f1 = None, None, -1
    report_text = []
    for name, classifier in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", classifier)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4)
        }
        results.append(metrics)
        report_text.append(f"{name}\n{classification_report(y_test, pred, zero_division=0)}\nConfusion matrix:\n{confusion_matrix(y_test, pred)}\n")
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name, best_pipe = name, pipe

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    (OUT_DIR / "model_evaluation.txt").write_text(
        f"Prediction setup\n================\nEarly attendance features are calculated before cutoff date {pd.Timestamp(cutoff_date).date()}.\n"
        "Target: whether attendance in the later period is below 75%.\n"
        "Data is synthetic and these metrics are for software demonstration only.\n\n"
        + "\n".join(report_text), encoding="utf-8"
    )
    joblib.dump(best_pipe, MODEL_DIR / "attendance_risk_model.joblib")
    metadata = {
        "best_model": best_name,
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "feature_columns": feature_cols,
        "threshold": 75,
        "note": "Synthetic demonstration dataset; do not use for real student decisions."
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Prediction file: features available at the cutoff; predicted risk from selected model.
    features["predicted_risk_probability"] = best_pipe.predict_proba(X)[:, 1]
    features["predicted_risk"] = (features["predicted_risk_probability"] >= 0.5).astype(int)
    features["predicted_risk_label"] = features["predicted_risk"].map({1: "At risk", 0: "Lower risk"})
    features.to_csv(OUT_DIR / "student_risk_predictions.csv", index=False)
    print("Training complete.")
    print(f"Cutoff date: {pd.Timestamp(cutoff_date).date()}")
    print(f"Best model by test F1-score: {best_name}")
    print(results_df.to_string(index=False))
    print(f"Outputs saved in: {OUT_DIR}")
    return results_df

if __name__ == "__main__":
    train()
