"""VoiceAttend AI entry point with a single explicit navigation tree."""

import streamlit as st

from dashboard_view import render_dashboard
from database import db
from services import auth
from ui import apply_theme, render_login, render_sidebar

st.set_page_config(page_title="VoiceAttend AI", page_icon="assets/voiceattend_logo.svg", layout="wide", initial_sidebar_state="expanded")
apply_theme()
db.init_db()


def home_page() -> None:
    if not auth.has_any_admin() or not auth.is_authenticated():
        render_login(auth)
        return
    render_dashboard()


pages = {
    "OVERVIEW": [st.Page(home_page, title="Dashboard", default=True, icon=":material/dashboard:")],
    "ATTENDANCE": [
        st.Page("pages/4_Live_Attendance.py", title="Live Attendance", icon=":material/mic:"),
        st.Page("pages/5_Attendance_Reports.py", title="Attendance Reports", icon=":material/insights:"),
    ],
    "MANAGEMENT": [
        st.Page("pages/2_Students.py", title="Students", icon=":material/group:"),
        st.Page("pages/3_Voice_Enrollment.py", title="Voice Enrollment", icon=":material/record_voice_over:"),
    ],
    "INTELLIGENCE": [st.Page("pages/6_System_Status.py", title="AI System Health", icon=":material/monitor_heart:")],
}

render_sidebar("Dashboard", pages)
auth.render_user_info_sidebar()
navigation = st.navigation(pages, position="hidden")
navigation.run()
