from pathlib import Path
import sys
import html
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# AttendIQ - Student Attendance Analytics & Risk Prediction
# Frontend only. Existing src/, data/, models/ and outputs/ are used.
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_processing import load_data, build_student_features
from train_model import train

st.set_page_config(
    page_title="AttendIQ | Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ DEMO LOGIN -----------------------------------
# Demo credentials only. Do not use hardcoded credentials in public deployment.
DEMO_USERS = {
    "Admin": {"username": "admin", "password": "admin123"},
    "Faculty": {"username": "faculty", "password": "faculty123"},
}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "logged_in_role" not in st.session_state:
    st.session_state.logged_in_role = ""
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = ""
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "📊 Overview"

if not st.session_state.authenticated:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
        .stApp {
            background: radial-gradient(circle at 8% 12%, rgba(90,132,255,.16), transparent 28%),
                        radial-gradient(circle at 92% 88%, rgba(34,197,186,.10), transparent 25%), #f3f6fc;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }
        .block-container { max-width: 1120px; padding: 2.5rem 1.5rem 2rem; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .auth-brand { display:flex; align-items:center; gap:12px; margin:0 0 25px 3px; }
        .auth-logo { width:46px; height:46px; display:flex; align-items:center; justify-content:center;
            border-radius:14px; color:#fff; font-size:25px; font-weight:800;
            background:linear-gradient(140deg,#527dff,#2349b9); box-shadow:0 9px 22px rgba(48,91,210,.25); }
        .auth-brand-name { color:#14213b; font:800 1.42rem 'Manrope',sans-serif; letter-spacing:-.045em; }
        .auth-brand-tag { color:#8794aa; font-size:.65rem; font-weight:800; letter-spacing:.16em; margin-top:1px; }
        .auth-hero { min-height:510px; height:100%; box-sizing:border-box; position:relative; overflow:hidden;
            padding:48px 42px; border-radius:26px; color:white;
            background:linear-gradient(145deg,#101d38 0%,#183567 58%,#315fe0 100%);
            box-shadow:0 22px 55px rgba(25,49,100,.17); }
        .auth-hero:before, .auth-hero:after { content:''; position:absolute; border:1px solid rgba(255,255,255,.13);
            border-radius:50%; pointer-events:none; }
        .auth-hero:before { width:390px;height:390px;right:-170px;top:-155px; }
        .auth-hero:after { width:285px;height:285px;right:-115px;top:-102px; }
        .auth-kicker { color:#b9ccff; font-size:.7rem; font-weight:800; letter-spacing:.17em; }
        .auth-heading { color:#fff; font:800 2.45rem/1.16 'Manrope',sans-serif; letter-spacing:-.06em; margin:22px 0 15px; }
        .auth-copy { color:#d7e2ff; font-size:.94rem; line-height:1.8; max-width:390px; }
        .auth-points { display:grid; gap:17px; margin-top:38px; }
        .auth-point { display:flex; align-items:center; gap:13px; color:#edf2ff; font-size:.88rem; }
        .auth-point-icon { width:34px;height:34px;display:flex;align-items:center;justify-content:center;
            flex:0 0 34px; border-radius:11px; background:rgba(255,255,255,.13); color:#fff; font-size:1rem; }
        .auth-foot { position:absolute; bottom:28px; left:42px; color:#b9c9ee; font-size:.72rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.97);
            border:1px solid #e5eaf3 !important; border-radius:25px !important;
            box-shadow:0 18px 50px rgba(30,49,82,.09); padding:30px 30px 24px !important; }
        .auth-form-kicker { color:#5274cf; font-size:.69rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
        .auth-form-title { color:#17243a; font:800 1.8rem 'Manrope',sans-serif; letter-spacing:-.05em; margin:7px 0 5px; }
        .auth-form-sub { color:#8390a5; font-size:.86rem; margin-bottom:22px; }
        [data-testid="stTextInput"] label, [data-testid="stSelectbox"] label {
            color:#43516a !important; font-weight:700 !important; font-size:.82rem !important; }
        [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border:1px solid #dce4f0 !important; border-radius:11px !important;
            background:#f9fbff !important; min-height:45px; }
        [data-testid="stTextInput"] input:focus { border-color:#5278ed !important; box-shadow:0 0 0 3px #e9efff !important; }
        .stButton > button[kind="primary"] { background:linear-gradient(100deg,#426ff2,#2c52ca) !important;
            color:#fff !important; border:0 !important; border-radius:11px !important; min-height:48px;
            font-weight:800 !important; box-shadow:0 8px 18px rgba(56,103,244,.2); transition:transform .2s ease, box-shadow .2s ease; }
        .stButton > button[kind="primary"]:hover { transform:translateY(-2px); box-shadow:0 12px 22px rgba(56,103,244,.28); }
        [data-testid="stExpander"] { border:1px solid #e7ecf5 !important; border-radius:12px !important; }
        @media (max-width: 850px) { .block-container { padding:1.2rem .8rem; }
            .auth-hero { min-height:360px; padding:30px 25px; } .auth-heading { font-size:1.9rem; }
            .auth-foot { left:25px; } [data-testid="stVerticalBlockBorderWrapper"] { padding:23px 20px !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="auth-brand">
            <div class="auth-logo">◈</div>
            <div><div class="auth-brand-name">AttendIQ</div>
            <div class="auth-brand-tag">STUDENT ANALYTICS PLATFORM</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.04, 0.96], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="auth-hero">
                <div class="auth-kicker">SMARTER ACADEMIC INSIGHTS</div>
                <div class="auth-heading">Every student.<br>Every signal.<br>One clear view.</div>
                <div class="auth-copy">Track attendance, understand student engagement and spot possible
                attendance risks early with a simple analytics workspace.</div>
                <div class="auth-points">
                    <div class="auth-point"><span class="auth-point-icon">▦</span> Attendance trends at a glance</div>
                    <div class="auth-point"><span class="auth-point-icon">◎</span> Early risk prediction insights</div>
                    <div class="auth-point"><span class="auth-point-icon">↗</span> Model performance overview</div>
                </div>
                <div class="auth-foot">A smarter way to understand student attendance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown('<div class="auth-form-kicker">Secure workspace access</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-title">Welcome back</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-sub">Enter your details to continue to AttendIQ.</div>', unsafe_allow_html=True)
            role = st.selectbox("Login as", ["Admin", "Faculty"], key="login_role")
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

            if st.button("Sign in  →", type="primary", use_container_width=True):
                user = DEMO_USERS[role]
                if username == user["username"] and password == user["password"]:
                    st.session_state.authenticated = True
                    st.session_state.logged_in_role = role
                    st.session_state.logged_in_user = username
                    st.session_state.selected_page = "📊 Overview"
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

            with st.expander("Demo login credentials"):
                st.markdown("**Admin**  \nUsername: `admin`  \nPassword: `admin123`")
                st.markdown("**Faculty**  \nUsername: `faculty`  \nPassword: `faculty123`")
            st.caption("Demo access only · Not intended for production authentication.")
    st.stop()

# ------------------------------ DESIGN SYSTEM -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #17243a; --muted: #718096; --blue: #3867f4;
        --line: #e8edf5; --canvas: #f6f8fc;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { padding: 1.65rem 2.3rem 2.5rem; max-width: 1600px; }
    [data-testid="stHeader"] { background: rgba(246,248,252,.9); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: -.035em; }
    h1 { font-size: 2rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 800 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 800 !important; }
    p, label, .stCaption { color: var(--muted); }

    [data-testid="stSidebar"] { background: #101c35; border-right: 1px solid #1d2a45; }
   /* Sidebar normal text */
[data-testid="stSidebar"] {
    color: #e8eefb;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9eacc5 !important;
}

/* Dropdown selected value - BLACK */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

/* Dropdown arrow */
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #000000 !important;
}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #9eacc5; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] label { color: #dbe5f7; font-weight: 600; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 10px;
    }
    /* Keep the selected Department / Academic year value readable */
   
/* FIX SELECTED DROPDOWN TEXT COLOR */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="singleValue"],
[data-testid="stSidebar"] div[data-baseweb="select"] div[class*="valueContainer"],
[data-testid="stSidebar"] div[data-baseweb="select"] div[role="combobox"],
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .4rem; }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px; border: 1px solid #dce4f1; font-weight: 700;
        padding: .55rem 1rem; min-height: 2.65rem; transition: all .18s ease;
    }
    .stButton > button:hover {
        border-color: #7e9cff !important; color: #3867f4 !important;
        background: #edf2ff !important; transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stButton > button {
        text-align: left; justify-content: flex-start; width: 100%;
        background: #1b2945; color: #e8eefb; border: 1px solid #2b3b5b;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #263d68 !important; color: #ffffff !important;
        border-color: #5d83df !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #315b91 !important; color: #ffffff !important;
        border-color: #5d83df !important;
    }
    .stDownloadButton > button { background: white; color: var(--blue); }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid var(--line); border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 5px 18px rgba(26,45,82,.035);
    }
    div[data-testid="stMetricLabel"] { color: #718096; font-weight: 700; font-size: .82rem; }
    div[data-testid="stMetricValue"] { color: #17243a; font-family: 'Manrope',sans-serif; font-weight: 800; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stPlotlyChart"] {
        background: white; border: 1px solid var(--line); border-radius: 16px;
        padding: 10px 10px 2px; box-shadow: 0 5px 18px rgba(26,45,82,.025);
    }
    hr { border-color: var(--line); }
    .hero {
        background: linear-gradient(112deg, #142342 0%, #1d3e79 58%, #3867f4 100%);
        border-radius: 20px; padding: 27px 30px; color: white;
        box-shadow: 0 12px 28px rgba(40,83,170,.16); margin-bottom: 22px;
    }
    .hero-kicker { color: #b9cbff; font-size: .76rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    .hero-title { color: white; font: 800 1.75rem 'Manrope',sans-serif; margin: 7px 0 5px; letter-spacing: -.04em; }
    .hero-copy { color: #d7e2ff; font-size: .94rem; margin: 0; }
    .hero-pill { display:inline-block; margin-top:15px; padding:6px 11px; border:1px solid #7795db;
        border-radius:30px; color:#e7edff; font-size:.75rem; font-weight:700; }
    .section-head { display:flex; align-items:center; justify-content:space-between; margin: 10px 0 13px; }
    .section-eyebrow { color:#8190a8; font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .section-title { color:#17243a; font:800 1.17rem 'Manrope',sans-serif; margin-top:3px; }
    .panel-note { color:#7c8aa1; font-size:.82rem; margin-top:-5px; margin-bottom:12px; }
    .mini-card { background:white; border:1px solid var(--line); border-radius:15px; padding:17px 18px;
        min-height:115px; box-shadow:0 5px 18px rgba(26,45,82,.025); }
    .mini-label { color:#78869c; font-size:.77rem; font-weight:700; }
    .mini-value { color:#17243a; font:800 1.55rem 'Manrope',sans-serif; margin-top:9px; }
    .mini-foot { color:#8b98aa; font-size:.74rem; margin-top:3px; }
    .status-chip { display:inline-block; padding:5px 9px; border-radius:20px; font-size:.73rem; font-weight:800; }
    .status-risk { color:#b42318; background:#fff0ee; }
    .status-safe { color:#13764b; background:#eaf8f0; }
    .info-strip { background:#edf3ff; border:1px solid #dce7ff; color:#35558e; padding:12px 15px; border-radius:12px; font-size:.84rem; }
    .warning-strip { background:#fff8e8; border:1px solid #f5e5b8; color:#88621b; padding:12px 15px; border-radius:12px; font-size:.83rem; }
    .sidebar-brand { font:800 1.35rem 'Manrope',sans-serif; color:white; letter-spacing:-.04em; }
    .sidebar-sub { color:#9eacc5; font-size:.75rem; margin-top:2px; }
    .sidebar-rule { height:1px; background:#2b3955; margin:18px 0; }
    @media (max-width: 800px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 22px; }
        .hero-title { font-size:1.4rem; }
    }
    header[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def get_dashboard_data():
    students, attendance = load_data()
    features, cutoff_date = build_student_features(students, attendance)

    attendance_pct = (
        attendance.groupby("student_id")["is_present"]
        .mean()
        .mul(100)
        .round(2)
        .reset_index(name="attendance_percentage")
    )

    summary = students.merge(
        attendance_pct,
        on="student_id",
        how="left"
    )

    summary["attendance_percentage"] = (
        summary["attendance_percentage"].fillna(0)
    )
    summary["year"] = summary["year"].astype(str)

    return students, attendance, features, cutoff_date, summary


# -------------------------------- DATA ---------------------------------------
try:
    students, attendance, features, cutoff_date, summary = get_dashboard_data()
except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()

# ------------------------------- SIDEBAR -------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">◈ AttendIQ</div>'
        '<div class="sidebar-sub">STUDENT ANALYTICS PLATFORM</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown("**WORKSPACE**")

    # These are clickable menu buttons, not radio buttons or tabs.
    if st.button(
        "📊  Dashboard overview", key="nav_overview", use_container_width=True,
        type="primary" if st.session_state.selected_page == "📊 Overview" else "secondary",
    ):
        st.session_state.selected_page = "📊 Overview"
        st.rerun()

    if st.button(
        "👥  Student directory", key="nav_students", use_container_width=True,
        type="primary" if st.session_state.selected_page == "👥 Student directory" else "secondary",
    ):
        st.session_state.selected_page = "👥 Student directory"
        st.rerun()

    if st.button(
        "🎯  Risk intelligence", key="nav_risk", use_container_width=True,
        type="primary" if st.session_state.selected_page == "🎯 Risk prediction" else "secondary",
    ):
        st.session_state.selected_page = "🎯 Risk prediction"
        st.rerun()

    if st.button(
        "🧠  Model performance", key="nav_model", use_container_width=True,
        type="primary" if st.session_state.selected_page == "🧠 Model evaluation" else "secondary",
    ):
        st.session_state.selected_page = "🧠 Model evaluation"
        st.rerun()

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown("**FILTERS**")
    departments = ["All departments"] + sorted(
        summary["department"].dropna().astype(str).unique().tolist()
    )
    years = ["All years"] + sorted(summary["year"].dropna().astype(str).unique().tolist())
    selected_dept = st.selectbox("Department", departments)
    selected_year = st.selectbox("Academic year", years)
    threshold = st.slider("Attendance threshold", 50, 90, 75, format="%d%%")

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown("**DATA SNAPSHOT**")
    st.caption(
        f"📅 {attendance['date'].min().strftime('%d %b %Y')} — "
        f"{attendance['date'].max().strftime('%d %b %Y')}"
    )
    st.caption(f"🗂️ {len(students):,} student records")
    st.caption("🔒 Synthetic demonstration dataset")
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.caption(f"Signed in: {st.session_state.logged_in_user} ({st.session_state.logged_in_role})")
    if st.button("↩ Logout", key="logout_button", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.logged_in_role = ""
        st.session_state.logged_in_user = ""
        st.session_state.selected_page = "📊 Overview"
        st.rerun()
    st.caption("AttendIQ · Academic analytics")

selected_page = st.session_state.selected_page
# ------------------------------ FILTERED DATA --------------------------------
filtered = summary.copy()
if selected_dept != "All departments":
    filtered = filtered[filtered["department"].astype(str) == selected_dept]
if selected_year != "All years":
    filtered = filtered[filtered["year"].astype(str) == selected_year]

filtered_ids = filtered["student_id"].tolist()
filtered_attendance = attendance[attendance["student_id"].isin(filtered_ids)].copy()
overall = float(filtered["attendance_percentage"].mean()) if not filtered.empty else 0.0
below = int((filtered["attendance_percentage"] < threshold).sum())
record_count = len(filtered_attendance)

# ------------------------------ SHARED HELPERS -------------------------------
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

def polish(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#52627a", size=12),
        title=dict(font=dict(family="Manrope, sans-serif", size=15, color="#17243a"), x=0.02),
        margin=dict(l=22, r=18, t=55, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e8edf5", tickfont=dict(color="#718096"))
    fig.update_yaxes(gridcolor="#edf1f7", zeroline=False, tickfont=dict(color="#718096"))
    return fig


def section(eyebrow, title, note=None):
    st.markdown(
        f'<div class="section-head"><div><div class="section-eyebrow">{eyebrow}</div>'
        f'<div class="section-title">{title}</div></div></div>',
        unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="panel-note">{note}</div>', unsafe_allow_html=True)

# -------------------------------- OVERVIEW ------------------------------------
if selected_page == "📊 Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Academic intelligence · Overview</div>
            <div class="hero-title">Student Attendance Analytics</div>
            <p class="hero-copy">Monitor attendance patterns, explore student performance, and identify potential attendance risks.</p>
            <span class="hero-pill">● LIVE PROJECT DASHBOARD</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards - these appear only on the Overview page.
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("TOTAL STUDENTS", f"{len(filtered):,}", help="Students in the selected filters")
    with k2:
        st.metric("AVERAGE ATTENDANCE", f"{overall:.1f}%", help="Average attendance across filtered students")
    with k3:
        st.metric(f"BELOW {threshold}% THRESHOLD", f"{below:,}", help="Students below the selected attendance threshold")
    with k4:
        st.metric("ATTENDANCE RECORDS", f"{record_count:,}", help="Records belonging to filtered students")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    section(
        "ATTENDANCE INSIGHTS",
        "Understand the attendance picture",
        "Explore the distribution and compare attendance across departments.",
    )
    left, right = st.columns(2)
    with left:
        if not filtered.empty:
            fig = px.histogram(
                filtered, x="attendance_percentage", nbins=14,
                labels={"attendance_percentage": "Attendance (%)", "count": "Students"},
                color_discrete_sequence=["#4775f5"],
            )
            fig.add_vline(
                x=threshold, line_dash="dash", line_color="#e39a36",
                annotation_text=f"{threshold}% target", annotation_position="top",
            )
            fig.update_traces(marker_line_width=0, opacity=.9)
            st.plotly_chart(polish(fig, 345), use_container_width=True, config=PLOT_CONFIG)
        else:
            st.info("No students match the selected filters.")

    with right:
        dept = filtered.groupby("department", as_index=False)["attendance_percentage"].mean()
        if not dept.empty:
            fig = px.bar(
                dept, x="department", y="attendance_percentage",
                labels={"department": "Department", "attendance_percentage": "Average attendance (%)"},
                color="attendance_percentage", color_continuous_scale=["#b9caff", "#3867f4"],
                range_y=[0, 100],
            )
            fig.add_hline(y=threshold, line_dash="dash", line_color="#e39a36")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(polish(fig, 345), use_container_width=True, config=PLOT_CONFIG)
        else:
            st.info("Department comparison is not available for these filters.")

    section(
        "DAILY MOVEMENT",
        "Attendance trend over time",
        "Daily percentage of present records for the students in the current view.",
    )
    if not filtered_attendance.empty:
        daily = filtered_attendance.groupby("date", as_index=False)["is_present"].mean()
        daily["attendance_percentage"] = daily["is_present"] * 100
        fig = px.area(
            daily, x="date", y="attendance_percentage",
            labels={"date": "Date", "attendance_percentage": "Attendance (%)"},
            markers=True, color_discrete_sequence=["#3867f4"],
        )
        fig.update_traces(
            fill="tozeroy", fillcolor="rgba(56,103,244,.10)",
            line=dict(width=3), marker=dict(size=6),
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(polish(fig, 330), use_container_width=True, config=PLOT_CONFIG)
    else:
        st.info("No attendance records are available for this selection.")

    a, b, c = st.columns(3)
    with a:
        highest_name = (
            html.escape(str(filtered.loc[filtered["attendance_percentage"].idxmax(), "student_name"]))
            if not filtered.empty else "No student data"
        )
        highest_value = f"{filtered['attendance_percentage'].max():.1f}" if not filtered.empty else "0.0"
        st.markdown(
            f'<div class="mini-card"><div class="mini-label">HIGHEST ATTENDANCE</div>'
            f'<div class="mini-value">{highest_value}%</div><div class="mini-foot">{highest_name}</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        lowest_name = (
            html.escape(str(filtered.loc[filtered["attendance_percentage"].idxmin(), "student_name"]))
            if not filtered.empty else "No student data"
        )
        lowest_value = f"{filtered['attendance_percentage'].min():.1f}" if not filtered.empty else "0.0"
        st.markdown(
            f'<div class="mini-card"><div class="mini-label">LOWEST ATTENDANCE</div>'
            f'<div class="mini-value">{lowest_value}%</div><div class="mini-foot">{lowest_name}</div></div>',
            unsafe_allow_html=True,
        )
    with c:
        safe_count = int((filtered["attendance_percentage"] >= threshold).sum())
        st.markdown(
            f'<div class="mini-card"><div class="mini-label">MEETING THRESHOLD</div>'
            f'<div class="mini-value">{safe_count:,}</div>'
            f'<div class="mini-foot">Students at or above {threshold}%</div></div>',
            unsafe_allow_html=True,
        )


# ----------------------------- STUDENT DIRECTORY -----------------------------
elif selected_page == "👥 Student directory":

    section(
        "STUDENT DIRECTORY",
        "Find and review student records",
        "Search by student name or ID. The table follows the department and year filters.",
    )

    # -------------------------- ADD STUDENT FORM --------------------------
    if st.session_state.logged_in_role == "Admin":

        with st.expander("➕ Add New Student", expanded=False):

            with st.form("add_student_form"):

                st.subheader("Add Student Details")

                student_id = st.text_input("Student ID")
                student_name = st.text_input("Student Name")

                department = st.selectbox(
                    "Department",
                    ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT"],
                )

                year = st.selectbox(
                    "Academic Year",
                    ["1", "2", "3", "4"],
                )

                submitted = st.form_submit_button(
                    "Add Student",
                    use_container_width=True,
                )

                if submitted:

                    student_id = student_id.strip()
                    student_name = student_name.strip()

                    if not student_id or not student_name:
                        st.error("Please enter Student ID and Student Name.")

                    elif students["student_id"].astype(str).str.strip().str.lower().eq(
                        student_id.lower()
                    ).any():
                        st.error("Student ID already exists.")

                    else:
                        # Preserve all existing CSV columns
                        new_record = {column: "" for column in students.columns}

                        new_record["student_id"] = student_id
                        new_record["student_name"] = student_name
                        new_record["department"] = department
                        new_record["year"] = year

                        new_student = pd.DataFrame([new_record])

                        updated_students = pd.concat(
                            [students, new_student],
                            ignore_index=True,
                        )

                        updated_students.to_csv(
                            ROOT / "data" / "students.csv",
                            index=False,
                        )

                        get_dashboard_data.clear()

                        st.success("Student added successfully!")
                        st.rerun()

    else:
        st.info("Faculty access is read-only. Only Admin can add students.")

    # -------------------------- SEARCH STUDENTS --------------------------
    search = st.text_input(
        "Search students",
        placeholder="Enter student name or ID...",
        key="student_search",
    )

    directory = filtered.copy()

    if search.strip():
        term = search.strip().lower()

        directory = directory[
            directory["student_name"].astype(str).str.lower().str.contains(
                term, na=False
            )
            | directory["student_id"].astype(str).str.lower().str.contains(
                term, na=False
            )
        ]

    directory["attendance_status"] = directory["attendance_percentage"].apply(
        lambda value: "Below threshold"
        if value < threshold
        else "Meeting threshold"
    )

    directory = directory.sort_values("attendance_percentage")

    show_cols = [
        "student_id",
        "student_name",
        "department",
        "year",
        "attendance_percentage",
        "attendance_status",
    ]

    st.caption(f"Showing {len(directory):,} of {len(filtered):,} students")

    st.dataframe(
        directory[show_cols].rename(
            columns={
                "student_id": "Student ID",
                "student_name": "Student name",
                "department": "Department",
                "year": "Year",
                "attendance_percentage": "Attendance (%)",
                "attendance_status": "Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Attendance (%)": st.column_config.ProgressColumn(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    d1, d2 = st.columns([1, 3])

    with d1:
        st.download_button(
            "⬇ Download student CSV",
            data=directory.to_csv(index=False).encode("utf-8"),
            file_name="filtered_student_summary.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=directory.empty,
        )

    with d2:
        st.markdown(
            '<div class="info-strip">Student attendance is calculated as present classes ÷ total recorded classes × 100.</div>',
            unsafe_allow_html=True,
        )

    # -------------------------- STUDENT PROFILE --------------------------
    if not directory.empty:

        section("STUDENT PROFILE", "Individual attendance snapshot")

        profile_options = directory.apply(
            lambda row: f"{row['student_id']} · {row['student_name']}",
            axis=1,
        ).tolist()

        selected_profile = st.selectbox(
            "Choose a student",
            profile_options,
            key="profile_student",
        )

        profile_id = selected_profile.split(" · ", 1)[0]

        profile = directory[
            directory["student_id"].astype(str) == profile_id
        ].iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("Student", str(profile["student_name"]))
        p2.metric("Department", str(profile["department"]))
        p3.metric("Academic year", str(profile["year"]))
        p4.metric("Attendance", f"{profile['attendance_percentage']:.1f}%")

        st.progress(
            min(max(float(profile["attendance_percentage"]) / 100, 0), 1)
        )

        if profile["attendance_percentage"] < threshold:
            st.markdown(
                f'<span class="status-chip status-risk">Below {threshold}% threshold</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-chip status-safe">Meeting attendance threshold</span>',
                unsafe_allow_html=True,
            )

# ------------------------------- RISK PREDICTION -----------------------------

# ------------------------------- RISK PREDICTION -----------------------------
elif selected_page == "🎯 Risk prediction":
    section(
        "RISK INTELLIGENCE", "Early attendance risk prediction",
        "The model uses first-period attendance information to estimate the likelihood of low attendance in the later period.",
    )
    st.markdown(
        f'<div class="info-strip">Prediction cutoff date: <b>{pd.Timestamp(cutoff_date).date()}</b> · '
        'Target: identify whether later-period attendance is below 75%.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    model_path = ROOT / "models" / "attendance_risk_model.joblib"
    pred_path = ROOT / "outputs" / "student_risk_predictions.csv"

    if not model_path.exists():
        st.warning("No trained model file was found. An Admin can train the models.")

    if st.session_state.logged_in_role == "Admin":
        train_col, status_col = st.columns([1, 2])
        with train_col:
            run_training = st.button(
                "⚡ Train / refresh models", type="primary", use_container_width=True
            )
        with status_col:
            st.caption(
                "Training compares Logistic Regression and Random Forest, then saves the best model and prediction outputs."
            )
            if run_training:
                try:
                    with st.spinner("Training models and preparing predictions..."):
                       results = train()
                    st.success("Training completed. Model metrics and risk predictions have been refreshed.")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Model training could not be completed: {exc}")
    else:
        st.info("Faculty access is read-only. Contact an Admin to train or refresh the models.")

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        pred["year"] = pred["year"].astype(str)
        if selected_dept != "All departments":
            pred = pred[pred["department"].astype(str) == selected_dept]
        if selected_year != "All years":
            pred = pred[pred["year"] == selected_year]
        pred["risk_probability_pct"] = pred["predicted_risk_probability"] * 100
        pred = pred.sort_values("risk_probability_pct", ascending=False)
        risk_count = int((pred["predicted_risk_label"] == "At risk").sum())
        r1, r2, r3 = st.columns(3)
        r1.metric("Students assessed", len(pred))
        r2.metric("Predicted at risk", risk_count)
        r3.metric(
            "Average risk probability",
            f"{pred['risk_probability_pct'].mean():.1f}%" if not pred.empty else "0.0%",
        )
        section("MODEL OUTPUT", "Student risk register")
        risk_view = pred[[
            "student_id", "student_name", "department", "year",
            "early_attendance_pct", "risk_probability_pct", "predicted_risk_label",
        ]].rename(columns={
            "student_id": "Student ID", "student_name": "Student name",
            "department": "Department", "year": "Year",
            "early_attendance_pct": "Early attendance (%)",
            "risk_probability_pct": "Risk probability (%)",
            "predicted_risk_label": "Predicted status",
        })
        st.dataframe(
            risk_view, use_container_width=True, hide_index=True,
            column_config={
                "Early attendance (%)": st.column_config.ProgressColumn(
                    "Early attendance (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Risk probability (%)": st.column_config.ProgressColumn(
                    "Risk probability (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
        st.download_button(
            "⬇ Download risk predictions",
            data=pred.to_csv(index=False).encode("utf-8"),
            file_name="student_risk_predictions.csv", mime="text/csv",
        )
        st.markdown(
            '<div class="warning-strip">Important: This is a synthetic-data demonstration. Risk scores are not a judgement of a student and must never be used alone for academic penalties or decisions.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Prediction file is not available yet. An Admin can train the models to generate it.")

# ------------------------------ MODEL EVALUATION -----------------------------
elif selected_page == "🧠 Model evaluation":
    section(
        "MODEL QUALITY", "Compare machine-learning performance",
        "Evaluation metrics are calculated on a held-out test split.",
    )
    metrics_path = ROOT / "outputs" / "model_comparison.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metric_cols = [
            column for column in ["accuracy", "precision", "recall", "f1_score"]
            if column in metrics.columns
        ]
        if not metrics.empty:
            best_row = (
                metrics.sort_values("f1_score", ascending=False).iloc[0]
                if "f1_score" in metrics.columns else metrics.iloc[0]
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BEST MODEL", str(best_row["model"]))
            m2.metric("ACCURACY", f"{best_row['accuracy']:.1%}" if "accuracy" in metrics else "—")
            m3.metric("PRECISION", f"{best_row['precision']:.1%}" if "precision" in metrics else "—")
            m4.metric("F1 SCORE", f"{best_row['f1_score']:.1%}" if "f1_score" in metrics else "—")

        section("PERFORMANCE COMPARISON", "Model metrics")
        if metric_cols:
            long_metrics = metrics.melt(
                id_vars="model", value_vars=metric_cols,
                var_name="metric", value_name="score",
            )
            fig = px.bar(
                long_metrics, x="model", y="score", color="metric", barmode="group",
                range_y=[0, 1], text_auto=".0%",
                color_discrete_sequence=["#3867f4", "#16a67a", "#f0a43a", "#8b6ce8"],
                labels={"model": "Model", "score": "Score", "metric": "Metric"},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(polish(fig, 370), use_container_width=True, config=PLOT_CONFIG)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download model comparison",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv", mime="text/csv",
        )
        report_path = ROOT / "outputs" / "model_evaluation.txt"
        if report_path.exists():
            with st.expander("View detailed evaluation report"):
                st.code(report_path.read_text(encoding="utf-8"), language="text")
        st.markdown(
            '<div class="warning-strip">These scores describe performance on the supplied synthetic dataset only. They do not establish real-world predictive validity.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Model evaluation is not available yet. Open Risk prediction and train the models first.")

# -------------------------------- FOOTER --------------------------------------
