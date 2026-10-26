"""VoiceAttend - Student Management.

Enables administrators to register students with unique codes, search,
edit student details, and remove students along with voice samples.
"""

import sqlite3
import streamlit as st
import pandas as pd
import config
from database import db
from services import auth
st.set_page_config(page_title="Students - VoiceAttend", page_icon="👥", layout="wide")
db.init_db()

st.title("👥 Student Management")
st.caption("Register, update, and manage student records and enrollment profiles")

tab_list, tab_add = st.tabs(["📋 Registered Students", "➕ Add New Student"])

with tab_add:
    st.subheader("Register a New Student")
    with st.form("add_student_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            student_code = st.text_input("Student ID / Roll Number *", placeholder="e.g. MCA2026-001")
            name = st.text_input("Full Name *", placeholder="e.g. Rahul Sharma")
        with col2:
            class_section = st.text_input("Class / Section (Optional)", placeholder="e.g. MCA Sem 3 - Section A")

        submitted = st.form_submit_button("Register Student", type="primary")
        if submitted:
            if not student_code.strip():
                st.error("Student ID is required.")
            elif not name.strip():
                st.error("Student Name is required.")
            else:
                try:
                    student_id = db.create_student(
                        student_code=student_code,
                        name=name,
                        class_section=class_section,
                    )
                    st.success(f"Student '{name}' ({student_code}) registered successfully! (ID: {student_id})")
                except sqlite3.IntegrityError:
                    st.error(f"Duplicate Error: Student ID '{student_code}' is already registered.")
                except Exception as e:
                    st.error(f"Error registering student: {e}")

with tab_list:
    search_query = st.text_input("🔍 Search by Name or Student ID", placeholder="Type name or roll number...")
    
    if search_query.strip():
        students = db.search_students(search_query.strip())
    else:
        students = db.get_all_students()

    if not students:
        st.info("No students found. Use the 'Add New Student' tab to register students.")
    else:
        st.write(f"Showing **{len(students)}** student(s):")
        
        for student in students:
            sid = student["id"]
            code = student["student_code"]
            sname = student["name"]
            sclass = student["class_section"] or "—"
            sample_count = student.get("sample_count", 0)

            # Determine badge
            if sample_count >= config.DEFAULT_ENROLLMENT_TARGET:
                badge = f"🟢 Enrolled ({sample_count}/{config.DEFAULT_ENROLLMENT_TARGET} samples)"
            elif sample_count > 0:
                badge = f"🟡 Partial ({sample_count}/{config.DEFAULT_ENROLLMENT_TARGET} samples)"
            else:
                badge = "🔴 Not Enrolled"

            with st.expander(f"**{code}** — {sname} [{badge}]"):
                c1, c2, c3 = st.columns([2, 2, 2])
                with c1:
                    st.write(f"**Class/Section:** {sclass}")
                    st.write(f"**Registered At:** {student['created_at']}")
                with c2:
                    st.write(f"**Voice Samples:** {sample_count}")
                    st.write(f"**Status:** {badge}")
                with c3:
                    if st.button(f"🎙️ Enroll Voice", key=f"enroll_btn_{sid}"):
                        st.session_state["selected_student_id"] = sid
                        st.switch_page("pages/3_Voice_Enrollment.py")

                st.markdown("---")
                edit_col, del_col = st.columns(2)

                # Edit Student
                with edit_col:
                    st.markdown("##### ✏️ Edit Student Info")
                    with st.form(f"edit_form_{sid}"):
                        new_name = st.text_input("Name", value=sname)
                        new_class = st.text_input("Class/Section", value=student["class_section"] or "")
                        if st.form_submit_button("Update Info"):
                            try:
                                db.update_student(sid, new_name, new_class)
                                st.success("Updated student details.")
                                st.rerun()
                            except Exception as err:
                                st.error(f"Update failed: {err}")

                # Delete Student
                with del_col:
                    st.markdown("##### 🗑️ Delete Student")
                    if auth.has_role("admin"):
                        st.caption("Permanently removes student record, voice embeddings, and saved audio recordings.")
                        confirm = st.checkbox("Confirm permanent deletion", key=f"confirm_del_{sid}")
                        if st.button("Delete Student", key=f"del_btn_{sid}", type="primary", disabled=not confirm):
                            success = db.delete_student(sid, delete_audio_files=True)
                            if success:
                                auth.log_audit_event("STUDENT_DELETED", f"Deleted student {code} ({sname})")
                                st.success(f"Student {sname} ({code}) deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete student.")
                    else:
                        st.caption("🔒 Administrator privilege required to delete student profiles.")
