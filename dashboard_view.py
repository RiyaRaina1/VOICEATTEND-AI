"""Shared VoiceAttend AI dashboard view backed by real database records."""

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from database import db
from services import anti_spoofing
from speech import command_verifier
from speaker import model_service
from ui import page_intro


def _theme(fig: go.Figure, height: int = 290) -> go.Figure:
    fig.update_layout(height=height, margin=dict(t=12, b=12, l=8, r=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ba9bf"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#20304a", zeroline=False)
    return fig


def render_dashboard() -> None:
    students = db.get_all_students()
    records = db.get_attendance_records()
    active_session = db.get_active_session()
    total = len(students)
    ready = sum(s.get("sample_count", 0) >= config.DEFAULT_ENROLLMENT_TARGET for s in students)
    partial = sum(0 < s.get("sample_count", 0) < config.DEFAULT_ENROLLMENT_TARGET for s in students)
    none = sum(s.get("sample_count", 0) == 0 for s in students)
    today = datetime.now().strftime("%Y-%m-%d")
    today_records = [r for r in records if str(r.get("marked_at", ""))[:10] == today]

    page_intro("VOICE INTELLIGENCE", "Voice Intelligence Dashboard", "Real-time attendance intelligence and biometric verification")
    a, b = st.columns([4, 1])
    with a:
        st.caption(datetime.now().strftime("%A, %d %B %Y"))
    with b:
        if st.button("Refresh data", use_container_width=True):
            st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total students", total)
    m2.metric("Voice profiles ready", ready, f"{ready}/{total}" if total else "0")
    m3.metric("Attendance today", len(today_records), f"of {ready} eligible" if ready else "No eligible profiles")
    m4.metric("Accepted records", len(records))

    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="va-panel"><div class="va-panel-title">Attendance activity</div><div class="va-muted">Accepted attendance records · last 7 days</div>', unsafe_allow_html=True)
        dates = [datetime.now().date() - timedelta(days=i) for i in range(6, -1, -1)]
        counts = {d.strftime("%Y-%m-%d"): 0 for d in dates}
        for record in records:
            key = str(record.get("marked_at", ""))[:10]
            if key in counts:
                counts[key] += 1
        df = pd.DataFrame({"Date": [d.strftime("%a") for d in dates], "Attendance": list(counts.values())})
        fig = _theme(px.bar(df, x="Date", y="Attendance", color_discrete_sequence=["#7167f8"]))
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Accepted records")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="va-panel"><div class="va-panel-title">Enrollment health</div><div class="va-muted">Target: {config.DEFAULT_ENROLLMENT_TARGET} samples per student</div>', unsafe_allow_html=True)
        if total:
            fig = go.Figure(go.Pie(labels=["Ready", "Partial", "Not enrolled"], values=[ready, partial, none], hole=.68, marker_colors=["#43d19a", "#f4b860", "#ff7b82"], textinfo="none"))
            fig = _theme(fig, 240)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-.05), annotations=[dict(text=f"{ready}/{total}", x=.5, y=.5, font_size=22, showarrow=False, font_color="#f5f7fb")])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No students registered yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.1, .9])
    with left:
        st.markdown('<div class="va-panel"><div class="va-panel-title">Live attendance session</div><div class="va-muted">Current session state from SQLite</div>', unsafe_allow_html=True)
        if active_session:
            session_records = db.get_attendance_records(session_id=active_session["id"])
            remaining = db.get_unmarked_students(active_session["id"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Session", active_session["session_name"])
            c2.metric("Attendees", len(session_records))
            c3.metric("Remaining", len(remaining))
            st.success(f"Active since {active_session['created_at']}")
        else:
            st.info("No active session. Create one from Live Attendance.")
        if st.button("Open live attendance", type="primary", use_container_width=True):
            st.switch_page("pages/4_Live_Attendance.py")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="va-panel"><div class="va-panel-title">System health</div><div class="va-muted">Live component availability</div>', unsafe_allow_html=True)
        health = [("SQLite database", True), ("Speaker recognition", model_service.get_model_info().get("available", False)), ("Speech command model", command_verifier.get_model_status().get("available", False)), ("Anti-spoofing heuristics", anti_spoofing.get_anti_spoofing_status().get("available", False))]
        for name, ok in health:
            color = "#43d19a" if ok else "#ff7b82"
            state = "Operational" if ok else "Unavailable"
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #20304a"><span>{name}</span><span style="color:{color};font-weight:700">{state}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="va-panel"><div class="va-panel-title">Recent attendance activity</div><div class="va-muted">Stored attendance records only; rejected attempts are not persisted by the current schema.</div>', unsafe_allow_html=True)
    if records:
        recent = pd.DataFrame(records[:8])[["student_code", "student_name", "session_name", "marked_at", "similarity_score", "decision"]]
        recent.columns = ["Student ID", "Student", "Session", "Timestamp", "Similarity", "Outcome"]
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance activity has been recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)
