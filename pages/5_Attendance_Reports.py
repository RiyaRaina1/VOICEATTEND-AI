"""VoiceAttend - Attendance Reports & Analytics.

Provides historical logs, multi-filter capabilities (session, student, date),
student-level attendance percentages with low-attendance warnings (<75%),
and one-click CSV and PDF export.
"""

from datetime import date
from io import BytesIO
import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from database import db
from services import auth
st.set_page_config(page_title="Attendance Reports - VoiceAttend", page_icon="📑", layout="wide")
db.init_db()

st.title("📑 Attendance Reports & Analytics")
st.caption("Review historical attendance logs, analyze participation rates, and export CSV/PDF reports")


def generate_pdf_report(records_df: pd.DataFrame, title: str = "VoiceAttend Attendance Report") -> bytes:
    """Generate a clean PDF report using ReportLab."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    elements = []
    styles = getSampleStyleSheet()

    # Header title
    title_style = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
    )
    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(f"Generated on: {date.today().strftime('%B %d, %Y')}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    # Convert dataframe to table data
    headers = ["Session", "ID", "Name", "Score", "Command", "Status", "Timestamp"]
    data = [headers]

    for _, row in records_df.iterrows():
        data.append([
            str(row.get("Session", ""))[:18],
            str(row.get("Student Code", "")),
            str(row.get("Name", ""))[:20],
            str(row.get("Similarity Score", "")),
            str(row.get("Command Verified", "")),
            str(row.get("Decision", "")),
            str(row.get("Timestamp", ""))[:19],
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


tab_logs, tab_analytics = st.tabs(["📋 Attendance Records & Export", "📊 Student-wise Attendance %"])

with tab_logs:
    # Filter Section
    with st.expander("🔍 Filter Options", expanded=True):
        col1, col2, col3 = st.columns(3)

        sessions = db.get_all_sessions()
        session_map = {"All Sessions": None}
        for s in sessions:
            session_map[f"{s['session_name']} (ID: {s['id']})"] = s["id"]

        with col1:
            selected_session = st.selectbox("Filter by Session", options=list(session_map.keys()))
            selected_session_id = session_map[selected_session]

        students = db.get_all_students()
        student_map = {"All Students": None}
        for st_item in students:
            student_map[f"{st_item['student_code']} - {st_item['name']}"] = st_item["id"]

        with col2:
            selected_student = st.selectbox("Filter by Student", options=list(student_map.keys()))
            selected_student_id = student_map[selected_student]

        with col3:
            use_date_filter = st.checkbox("Filter by Date")
            filter_date = st.date_input("Date", value=date.today()) if use_date_filter else None

    date_str = filter_date.strftime("%Y-%m-%d") if (use_date_filter and filter_date) else None

    # Fetch Records
    records = db.get_attendance_records(
        session_id=selected_session_id,
        student_id=selected_student_id,
        date_str=date_str,
    )

    if not records:
        st.info("No attendance records match the selected criteria.")
    else:
        df = pd.DataFrame(records)
        display_cols = [
            "session_name",
            "student_code",
            "student_name",
            "class_section",
            "similarity_score",
            "command_verified",
            "decision",
            "marked_at",
        ]
        for c in display_cols:
            if c not in df.columns:
                df[c] = ""

        df_view = df[display_cols].copy()
        df_view["command_verified"] = df_view["command_verified"].apply(lambda v: "Yes" if v else "No")
        df_view["similarity_score"] = df_view["similarity_score"].apply(lambda s: f"{s:.4f}")
        df_view.columns = [
            "Session",
            "Student Code",
            "Name",
            "Class/Section",
            "Similarity Score",
            "Command Verified",
            "Decision",
            "Timestamp",
        ]

        # Metrics Summary
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Records", len(df_view))
        with stat_col2:
            unique_students = df["student_id"].nunique()
            st.metric("Unique Students", unique_students)
        with stat_col3:
            avg_score = df["similarity_score"].astype(float).mean()
            st.metric("Average Similarity", f"{avg_score:.4f}")

        st.markdown("##### Attendance Records Table")
        st.dataframe(df_view, use_container_width=True, hide_index=True)

        # Export Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            csv_bytes = df_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Report to CSV",
                data=csv_bytes,
                file_name=f"VoiceAttend_Report_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        with btn_col2:
            try:
                pdf_bytes = generate_pdf_report(df_view)
                st.download_button(
                    label="📄 Export Report to PDF",
                    data=pdf_bytes,
                    file_name=f"VoiceAttend_Report_{date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="secondary",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.error(f"PDF generation error: {pdf_err}")

with tab_analytics:
    st.subheader("📊 Student Participation & Low Attendance Warnings")
    all_students = db.get_all_students()
    all_sessions = db.get_all_sessions()
    total_sess_count = len(all_sessions)

    if total_sess_count == 0 or not all_students:
        st.info("Insufficient session or student data to calculate attendance percentages.")
    else:
        student_stats = []
        low_attendance_count = 0

        for st_item in all_students:
            s_id = st_item["id"]
            attended = db.get_attendance_records(student_id=s_id)
            attended_sessions = set(r["session_id"] for r in attended)
            att_count = len(attended_sessions)
            percentage = round((att_count / total_sess_count) * 100, 1)

            if percentage < 75.0:
                low_attendance_count += 1
                status = "⚠️ Warning (<75%)"
            else:
                status = "✅ Normal"

            student_stats.append({
                "Student ID": st_item["student_code"],
                "Name": st_item["name"],
                "Class": st_item["class_section"] or "—",
                "Attended Sessions": f"{att_count}/{total_sess_count}",
                "Attendance %": percentage,
                "Status": status,
            })

        df_stats = pd.DataFrame(student_stats)

        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.metric("Total Sessions Conducted", total_sess_count)
        with stat2:
            st.metric("Total Enrolled Students", len(all_students))
        with stat3:
            st.metric("Students Below 75% Attendance", low_attendance_count, delta=-low_attendance_count if low_attendance_count else 0, delta_color="inverse")

        st.dataframe(
            df_stats.sort_values(by="Attendance %", ascending=True),
            use_container_width=True,
            hide_index=True,
        )
