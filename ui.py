"""Shared VoiceAttend AI visual system, branding, routing helpers, and auth UI."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import streamlit as st

import config

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "voiceattend_logo.svg"


def _logo_uri() -> str:
    encoded = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { --va-bg:#080d18; --va-panel:#111b2e; --va-elevated:#172338; --va-border:#263653;
          --va-text:#f4f6fc; --va-muted:#98a5bb; --va-indigo:#7167f8; --va-cyan:#35d5e8; }
        .stApp { background:var(--va-bg); color:var(--va-text); }
        [data-testid="stSidebar"] { background:#0d1424; border-right:1px solid #1e2b45; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:var(--va-muted); }
        [data-testid="stSidebarNav"] { padding-top:1rem; }
        [data-testid="stSidebarNav"] li a { border-radius:9px; margin:2px 8px; }
        [data-testid="stSidebarNav"] li a:hover { background:#172338; }
        [data-testid="stSidebarNav"] li a[aria-current="page"] { background:#25264e; color:#fff; border-left:3px solid #7167f8; }
        h1,h2,h3,h4 { letter-spacing:-.025em; color:var(--va-text); }
        h1 { font-size:2.1rem !important; }
        [data-testid="stMetric"] { background:linear-gradient(145deg,#111b2e,#0f1829); border:1px solid var(--va-border); border-radius:16px; padding:18px 20px; }
        [data-testid="stMetricLabel"] { color:var(--va-muted); }
        [data-testid="stMetricValue"] { color:var(--va-text); }
        .va-brand { padding:5px 0 18px; border-bottom:1px solid #263653; margin-bottom:12px; }
        .va-brand-row { display:flex; align-items:center; gap:9px; }
        .va-brand-logo { width:34px; height:34px; }
        .va-brand-name { color:#f5f7fb; font-weight:800; font-size:1.08rem; }
        .va-brand-sub { color:#8392aa; font-size:.72rem; margin-left:43px; margin-top:-2px; }
        .va-eyebrow { color:var(--va-cyan); text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; margin-bottom:6px; }
        .va-panel { background:var(--va-panel); border:1px solid var(--va-border); border-radius:18px; padding:20px; height:100%; }
        .va-panel-title { font-size:1.05rem; font-weight:750; color:var(--va-text); margin-bottom:4px; }
        .va-muted { color:var(--va-muted); font-size:.86rem; }
        .va-status { display:inline-flex; align-items:center; gap:7px; padding:5px 10px; border:1px solid #2d4562; border-radius:999px; color:#b9c6d9; font-size:.78rem; }
        .va-dot { width:7px; height:7px; border-radius:50%; background:#43d19a; }
        div.stButton > button { border-radius:10px; border:1px solid #334866; }
        div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#7167f8,#35a9e8); border-color:#7167f8; }
        [data-testid="stDataFrame"] { border:1px solid var(--va-border); border-radius:12px; overflow:hidden; }
        .va-login-shell { min-height:calc(100vh - 3rem); display:flex; align-items:center; }
        .va-login-left { padding:34px 46px 34px 4px; }
        .va-login-right { background:linear-gradient(145deg,#111b2e,#0f1829); border:1px solid #2a3b5e; border-radius:22px; padding:34px; box-shadow:0 24px 70px rgba(0,0,0,.28); }
        .va-login-logo { width:86px; height:86px; }
        .va-wordmark { font-size:1.5rem; font-weight:800; color:#f4f6fc; margin-top:10px; }
        .va-tagline { color:#98a5bb; font-size:.9rem; }
        .va-login-headline { font-size:3rem; line-height:1.05; font-weight:800; letter-spacing:-.05em; margin:32px 0 16px; }
        .va-gradient-text { background:linear-gradient(100deg,#a59fff,#7cecf5); -webkit-background-clip:text; background-clip:text; color:transparent; }
        .va-feature { display:flex; gap:12px; align-items:center; margin:15px 0; color:#dce4f2; }
        .va-feature-icon { color:#35d5e8; font-size:1.1rem; }
        .va-wave { height:100px; margin:22px 0 24px; border-radius:18px; background:radial-gradient(circle at 20% 40%,rgba(113,103,248,.32),transparent 42%),radial-gradient(circle at 80% 60%,rgba(53,213,232,.20),transparent 42%),#0e172a; border:1px solid #263653; position:relative; overflow:hidden; }
        .va-wave:after { content:""; position:absolute; inset:49% 5% auto; height:2px; background:linear-gradient(90deg,transparent,#7167f8,#35d5e8,transparent); box-shadow:0 -13px 0 rgba(113,103,248,.55),0 13px 0 rgba(53,213,232,.35); transform:skewY(-4deg); }
        .va-security { color:#98a5bb; font-size:.78rem; text-align:center; margin-top:18px; }
        </style>
        """, unsafe_allow_html=True,
    )


def render_sidebar(active: Optional[str] = None, pages: Optional[dict] = None) -> None:
    """Render the single branded navigation system."""
    uri = _logo_uri()
    st.sidebar.markdown(
        f'<div class="va-brand"><div class="va-brand-row"><img class="va-brand-logo" src="{uri}"><span class="va-brand-name">VoiceAttend AI</span></div><div class="va-brand-sub">Your Voice. Your Identity. Verified.</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("OVERVIEW")
    if pages:
        st.sidebar.page_link(pages["OVERVIEW"][0], label="Dashboard", icon=":material/dashboard:")
    else:
        st.sidebar.page_link("app.py", label="Dashboard", icon=":material/dashboard:")
    st.sidebar.caption("ATTENDANCE")
    if pages:
        st.sidebar.page_link(pages["ATTENDANCE"][0], label="Live Attendance", icon=":material/mic:")
        st.sidebar.page_link(pages["ATTENDANCE"][1], label="Attendance Reports", icon=":material/insights:")
    else:
        st.sidebar.page_link("pages/4_Live_Attendance.py", label="Live Attendance", icon=":material/mic:")
        st.sidebar.page_link("pages/5_Attendance_Reports.py", label="Attendance Reports", icon=":material/insights:")
    st.sidebar.caption("MANAGEMENT")
    if pages:
        st.sidebar.page_link(pages["MANAGEMENT"][0], label="Students", icon=":material/group:")
        st.sidebar.page_link(pages["MANAGEMENT"][1], label="Voice Enrollment", icon=":material/record_voice_over:")
    else:
        st.sidebar.page_link("pages/2_Students.py", label="Students", icon=":material/group:")
        st.sidebar.page_link("pages/3_Voice_Enrollment.py", label="Voice Enrollment", icon=":material/record_voice_over:")
    st.sidebar.caption("INTELLIGENCE")
    if pages:
        st.sidebar.page_link(pages["INTELLIGENCE"][0], label="AI System Health", icon=":material/monitor_heart:")
    else:
        st.sidebar.page_link("pages/6_System_Status.py", label="AI System Health", icon=":material/monitor_heart:")
    st.sidebar.divider()
    st.sidebar.caption("VERIFICATION SETTINGS")
    st.sidebar.slider(
        "Biometric similarity threshold",
        min_value=0.50,
        max_value=0.95,
        value=float(config.DEFAULT_SIMILARITY_THRESHOLD),
        step=0.01,
        key="similarity_threshold",
        help="Minimum cosine similarity required to accept a speaker match. Higher values are stricter.",
    )
    st.sidebar.caption(f"Current threshold: {st.session_state['similarity_threshold']:.2f}")
    st.sidebar.divider()
    st.sidebar.markdown('<span class="va-status"><span class="va-dot"></span> Local system online</span>', unsafe_allow_html=True)


def render_login(auth: Any) -> None:
    """Render branded first-time setup or login without exposing secrets."""
    uri = _logo_uri()
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.markdown('<div class="va-login-left">', unsafe_allow_html=True)
        st.markdown(f'<img class="va-login-logo" src="{uri}"><div class="va-wordmark">VoiceAttend AI</div><div class="va-tagline">Your Voice. Your Identity. Verified.</div>', unsafe_allow_html=True)
        st.markdown('<div class="va-login-headline">Attendance,<br><span class="va-gradient-text">verified by voice.</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="va-muted">Speaker recognition, speech-command verification and liveness signals in one focused attendance workflow.</div>', unsafe_allow_html=True)
        st.markdown('<div class="va-wave"></div>', unsafe_allow_html=True)
        for icon, label, text in [("◉", "Speaker Intelligence", "Identity matching built from enrolled voice profiles."), ("⌁", "Secure Verification", "Independent command and liveness checks."), ("◷", "Real-Time Attendance", "Fast local processing for classroom sessions.")]:
            st.markdown(f'<div class="va-feature"><span class="va-feature-icon">{icon}</span><span><b>{label}</b><br><span class="va-muted">{text}</span></span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="va-muted" style="margin-top:28px">VoiceAttend AI · v2.0.0</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="va-login-right">', unsafe_allow_html=True)
        if not auth.has_any_admin():
            st.markdown('<div class="va-eyebrow">INITIAL SETUP</div><h2>Create your admin workspace</h2><p class="va-muted">Set up the first administrator to protect attendance operations.</p>', unsafe_allow_html=True)
            with st.form("premium_first_time_setup"):
                username = st.text_input("Username", autocomplete="username")
                password = st.text_input("Password", type="password", autocomplete="new-password")
                confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
                submitted = st.form_submit_button("Create administrator", type="primary", use_container_width=True)
                if submitted:
                    if not username.strip() or len(password) < 8 or password != confirm:
                        st.error("Enter a username, an 8+ character password, and matching confirmation.")
                    elif auth.create_admin_user(username, password, role="admin"):
                        st.success("Administrator created. You can now sign in.")
                        st.rerun()
                    else:
                        st.error("That username is unavailable.")
        else:
            st.markdown('<div class="va-eyebrow">SECURE ACCESS</div><h2>Welcome back</h2><p class="va-muted">Sign in to continue to your voice intelligence workspace.</p>', unsafe_allow_html=True)
            sign_in_tab, create_tab, reset_tab = st.tabs(["Sign in", "Create account", "Reset password"])

            with sign_in_tab:
                if auth.is_locked_out():
                    st.error(f"Too many attempts. Try again in {auth.remaining_lockout_seconds():.0f} seconds.")
                else:
                    show_password = st.checkbox("Show password", value=False, key="login_show_password")
                    with st.form("premium_login_form"):
                        username = st.text_input("Username", autocomplete="username")
                        password = st.text_input("Password", type="default" if show_password else "password", autocomplete="current-password")
                        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
                        if submitted:
                            if auth.login(username, password):
                                st.success("Signed in successfully.")
                                st.rerun()
                            else:
                                st.error("Sign-in failed. Check your credentials and try again.")

            with create_tab:
                st.caption("Create a faculty account for attendance operations. New accounts cannot grant administrator access.")
                with st.form("premium_create_user_form"):
                    new_username = st.text_input("New username", autocomplete="username")
                    new_password = st.text_input("Password", type="password", autocomplete="new-password")
                    new_confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
                    create_submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
                    if create_submitted:
                        if not new_username.strip() or len(new_password) < 8 or new_password != new_confirm:
                            st.error("Enter a username, an 8+ character password, and matching confirmation.")
                        elif auth.create_admin_user(new_username, new_password, role="faculty"):
                            st.success("Account created. You can now sign in.")
                        else:
                            st.error("That username is unavailable.")

            with reset_tab:
                st.caption("For safety, an active administrator must authorize every password reset.")
                with st.form("premium_reset_password_form"):
                    target_username = st.text_input("Account to reset", autocomplete="username")
                    admin_username = st.text_input("Administrator username", autocomplete="username")
                    admin_password = st.text_input("Administrator password", type="password", autocomplete="current-password")
                    reset_password = st.text_input("New password", type="password", autocomplete="new-password")
                    reset_confirm = st.text_input("Confirm new password", type="password", autocomplete="new-password")
                    reset_submitted = st.form_submit_button("Reset password", type="primary", use_container_width=True)
                    if reset_submitted:
                        if not target_username.strip() or not admin_username.strip() or len(reset_password) < 8 or reset_password != reset_confirm:
                            st.error("Enter all account details, an 8+ character password, and matching confirmation.")
                        elif auth.reset_password_by_admin(target_username, reset_password, admin_username, admin_password):
                            st.success("Password reset completed. You can now sign in with the new password.")
                        else:
                            st.error("Password reset was not authorized or the account was not found.")
            st.markdown('<div class="va-security">◈ Local authentication · Session protected</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def page_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="va-eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)
