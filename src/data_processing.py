
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_data():
    students_file = DATA_DIR / "students.csv"
    attendance_file = DATA_DIR / "attendance.csv"

    if not students_file.exists():
        raise FileNotFoundError("students.csv file not found in data folder.")

    if not attendance_file.exists():
        raise FileNotFoundError("attendance.csv file not found in data folder.")

    students = pd.read_csv(students_file)
    attendance = pd.read_csv(attendance_file, parse_dates=["date"])

    required_student_cols = ["student_id", "department", "year"]
    required_attendance_cols = ["student_id", "date", "status"]

    for col in required_student_cols:
        if col not in students.columns:
            raise ValueError(f"Missing column in students.csv: {col}")

    for col in required_attendance_cols:
        if col not in attendance.columns:
            raise ValueError(f"Missing column in attendance.csv: {col}")

    students = students.drop_duplicates(subset=["student_id"])
    attendance = attendance.drop_duplicates(
        subset=["student_id", "date"]
    )

    attendance["status"] = attendance["status"].astype(str).str.strip().str.title()

    attendance = attendance[
        attendance["status"].isin(["Present", "Absent"])
    ].copy()

    attendance = attendance.dropna(subset=["student_id", "date"])

    attendance["is_present"] = (
        attendance["status"] == "Present"
    ).astype(int)

    return students, attendance


def build_student_features(
    students, attendance, cutoff_fraction=0.5, threshold=75
):
    if not 0 < cutoff_fraction < 1:
        raise ValueError("cutoff_fraction must be between 0 and 1.")

    dates = sorted(attendance["date"].dropna().unique())

    if len(dates) < 4:
        raise ValueError(
            "At least 4 distinct attendance dates are required."
        )

    cutoff_idx = max(
        1, min(len(dates) - 1, int(len(dates) * cutoff_fraction))
    )

    cutoff_date = dates[cutoff_idx]

    early = attendance[attendance["date"] < cutoff_date]
    later = attendance[attendance["date"] >= cutoff_date]

    early_summary = early.groupby("student_id").agg(
        early_classes=("status", "count"),
        early_present=("is_present", "sum"),
        early_absent=("is_present", lambda s: int((s == 0).sum())),
        early_attendance_pct=(
            "is_present",
            lambda s: round(float(s.mean() * 100), 2)
        ),
    ).reset_index()

    later_summary = later.groupby("student_id").agg(
        later_classes=("status", "count"),
        later_attendance_pct=(
            "is_present",
            lambda s: float(s.mean() * 100)
        ),
    ).reset_index()

    later_summary["at_risk"] = (
        later_summary["later_attendance_pct"] < threshold
    ).astype(int)

    features = students.merge(
        early_summary, on="student_id", how="left"
    )

    features = features.merge(
        later_summary[
            ["student_id", "later_classes",
             "later_attendance_pct", "at_risk"]
        ],
        on="student_id",
        how="left"
    )

    # Students without enough records should not be silently
    # treated as students with zero attendance.
    features["has_complete_period_data"] = (
        features["early_classes"].notna()
        & features["later_classes"].notna()
    )

    # Keep only students with valid early and later data
    # for ML training.
    features = features[
        features["has_complete_period_data"]
    ].copy()

    features["year"] = pd.to_numeric(
        features["year"], errors="coerce"
    )

    features = features.dropna(
        subset=["year", "early_attendance_pct", "at_risk"]
    )

    features["year"] = features["year"].astype(int)

    return features, cutoff_date


def export_cleaned_data(students, attendance):
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    students.to_csv(
        out / "clean_students.csv", index=False
    )

    attendance.drop(
        columns=["is_present"], errors="ignore"
    ).to_csv(
        out / "clean_attendance.csv", index=False
    )
