
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

try:
    from .data_processing import (
        load_data,
        build_student_features,
        export_cleaned_data
    )
except ImportError:
    from data_processing import (
        load_data,
        build_student_features,
        export_cleaned_data
    )


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 75
PROBABILITY_THRESHOLD = 0.5


# --------------------------------------------------
# 2. CREATE DATA PREPROCESSOR
# --------------------------------------------------

def create_preprocessor():

    categorical = ["department"]

    numeric = [
        "year",
        "early_classes",
        "early_present",
        "early_absent",
        "early_attendance_pct"
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            numeric
        )
    ])

    return preprocessor


# --------------------------------------------------
# 3. MAIN TRAINING FUNCTION
# --------------------------------------------------

def train():

    print("Starting model training...")

    # Step 1: Load and prepare data

    students, attendance = load_data()

    export_cleaned_data(students, attendance)

    features, cutoff_date = build_student_features(
        students,
        attendance,
        threshold=THRESHOLD
    )

    feature_cols = [
        "department",
        "year",
        "early_classes",
        "early_present",
        "early_absent",
        "early_attendance_pct"
    ]

    X = features[feature_cols]
    y = features["at_risk"]

    # Step 2: Validate dataset

    if len(features) < 8:
        raise ValueError(
            "Not enough student records for train/test splitting."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain both at-risk and lower-risk students."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each risk class must contain at least 2 students."
        )

    # Step 3: Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Step 4: Define machine learning algorithms

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced"
        )
    }

    results = []
    reports = []

    best_name = None
    best_f1 = -1
    best_pipeline = None

    # Step 5: Train and evaluate both models

    for name, classifier in candidates.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "model": name,

            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),

            "precision": round(
                float(precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )), 4
            ),

            "recall": round(
                float(recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )), 4
            ),

            "f1_score": round(
                float(f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )), 4
            )
        }

        results.append(metrics)

        reports.append(
            f"{name}\n"
            f"{classification_report(y_test, predictions, zero_division=0)}\n"
            f"Confusion Matrix:\n"
            f"{confusion_matrix(y_test, predictions)}\n"
        )

        # Select the best model using F1-score

        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]
            best_name = name
            best_pipeline = pipeline

    # Step 6: Save model comparison results

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False
    )

    results_df.to_csv(
        OUT_DIR / "model_comparison.csv",
        index=False
    )

    # Step 7: Save predictions for unseen test students
    # Use best_pipeline trained only on X_train.

    test_features = features.loc[X_test.index].copy()

    test_features["predicted_risk_probability"] = (
        best_pipeline.predict_proba(X_test)[:, 1]
    )

    test_features["predicted_risk"] = (
        test_features["predicted_risk_probability"]
        >= PROBABILITY_THRESHOLD
    ).astype(int)

    test_features["predicted_risk_label"] = (
        test_features["predicted_risk"].map({
            1: "At risk",
            0: "Lower risk"
        })
    )

    # Do not expose actual later-period outcome in prediction CSV.

    test_features = test_features.drop(
        columns=[
            "later_classes",
            "later_attendance_pct",
            "at_risk"
        ],
        errors="ignore"
    )

    test_features.to_csv(
        OUT_DIR / "student_risk_predictions.csv",
        index=False
    )

    # Step 8: Train final model using all available records.
    # This saved model is intended for future unseen student data.

    final_model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", candidates[best_name])
    ])

    final_model.fit(X, y)

    joblib.dump(
        final_model,
        MODEL_DIR / "attendance_risk_model.joblib"
    )

    # Step 9: Save model metadata

    metadata = {
        "best_model": best_name,
        "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "feature_columns": feature_cols,
        "threshold": THRESHOLD,
        "prediction_probability_threshold": PROBABILITY_THRESHOLD,
        "note": (
            "Synthetic demonstration dataset. "
            "Do not use for real student decisions."
        )
    }

    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    # Step 10: Save evaluation report

    report = (
        "ATTENDANCE RISK MODEL EVALUATION\n"
        "================================\n\n"
        f"Cutoff date: {pd.Timestamp(cutoff_date).date()}\n"
        f"Target: Later-period attendance below {THRESHOLD}%\n"
        "Features: Early-period attendance and student details\n"
        "Dataset: Synthetic demonstration data\n\n"
        + "\n".join(reports)
    )

    (OUT_DIR / "model_evaluation.txt").write_text(
        report,
        encoding="utf-8"
    )

    # Step 11: Display training summary

    print("\nTraining completed successfully!")

    print("Cutoff date:", pd.Timestamp(cutoff_date).date())

    print("Selected model:", best_name)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nTest prediction records:", len(test_features))

    print("\nSaved files:")
    print("- models/attendance_risk_model.joblib")
    print("- models/model_metadata.json")
    print("- outputs/model_comparison.csv")
    print("- outputs/model_evaluation.txt")
    print("- outputs/student_risk_predictions.csv")


# --------------------------------------------------
# 4. RUN THE PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    train()
