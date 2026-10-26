"""Dashboard page alias for VoiceAttend AI."""

import streamlit as st

from dashboard_view import render_dashboard
from database import db
from services import auth
from ui import apply_theme, render_sidebar

st.set_page_config(page_title="Dashboard - VoiceAttend AI", page_icon="V", layout="wide")
apply_theme()
db.init_db()
render_sidebar("Dashboard")
auth.render_user_info_sidebar()

if not auth.has_any_admin():
    st.info("Create the first administrator account from the home page.")
    st.stop()
if not auth.is_authenticated():
    auth._render_login_form()
    st.stop()

render_dashboard()
