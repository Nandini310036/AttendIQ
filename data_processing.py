from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def load_data():
    students = pd.read_csv(DATA_DIR / "students.csv")
    attendance = pd.read_csv(DATA_DIR / "attendance.csv", parse_dates=["date"])
    students = students.drop_duplicates(subset=["student_id"])
    attendance = attendance.drop_duplicates(subset=["student_id", "date"])
    attendance = attendance[attendance["status"].isin(["Present", "Absent"])].copy()
    attendance["is_present"] = (attendance["status"] == "Present").astype(int)
    return students, attendance

def build_student_features(students, attendance, cutoff_fraction=0.5, threshold=75):
    """Use only early-period attendance to predict final-period low attendance.
    Label is based on the later period; early features do not include the label period.
    """
    dates = sorted(attendance["date"].dropna().unique())
    if len(dates) < 4:
        raise ValueError("At least 4 distinct attendance dates are required.")
    cutoff_idx = max(1, min(len(dates)-1, int(len(dates) * cutoff_fraction)))
    cutoff_date = dates[cutoff_idx]
    early = attendance[attendance["date"] < cutoff_date]
    later = attendance[attendance["date"] >= cutoff_date]

    early_summary = early.groupby("student_id").agg(
        early_classes=("status", "count"),
        early_present=("is_present", "sum"),
        early_absent=("is_present", lambda s: int((s == 0).sum())),
        early_attendance_pct=("is_present", lambda s: round(float(s.mean() * 100), 2)),
    ).reset_index()

    later_summary = later.groupby("student_id").agg(
        later_classes=("status", "count"),
        later_attendance_pct=("is_present", lambda s: float(s.mean() * 100)),
    ).reset_index()
    later_summary["at_risk"] = (later_summary["later_attendance_pct"] < threshold).astype(int)

    features = students.merge(early_summary, on="student_id", how="inner")
    features = features.merge(later_summary[["student_id", "later_classes", "later_attendance_pct", "at_risk"]],
                              on="student_id", how="inner")
    features["year"] = features["year"].astype(int)
    return features, cutoff_date

def export_cleaned_data(students, attendance):
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    students.to_csv(out / "clean_students.csv", index=False)
    attendance.drop(columns=["is_present"]).to_csv(out / "clean_attendance.csv", index=False)
