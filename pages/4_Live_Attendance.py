"""VoiceAttend - Live Attendance.

Executes real-time attendance verification:
1. Audio quality check (silence, clipping, SNR)
2. Anti-spoofing / liveness heuristics (replay and synthetic signal detection)
3. Speech command verification ("Present")
4. Deep learning speaker identification (ECAPA-TDNN)
5. Confidence thresholding, ambiguous match rejection, and duplicate prevention
6. Multi-stage latency tracking and structured verification feedback
"""

import streamlit as st
import pandas as pd
import config
from database import db
from services import attendance_service, auth
st.set_page_config(page_title="Live Attendance - VoiceAttend", page_icon="⚡", layout="wide")
db.init_db()

st.title("⚡ Live Voice Attendance")
st.caption("Multi-stage verification: Audio Quality ➔ Anti-Spoofing ➔ Speech Command ➔ Biometric Identification")

# Sidebar: Session Management & Threshold Calibration
with st.sidebar:
    st.subheader("⚙️ Verification Settings")

    similarity_threshold = float(
        st.session_state.get("similarity_threshold", config.DEFAULT_SIMILARITY_THRESHOLD)
    )

    enable_anti_spoofing = st.toggle("Enable Anti-Spoofing Heuristics", value=True)

    st.divider()

    # Active Session Management
    st.subheader("📅 Attendance Session")
    sessions = db.get_all_sessions()
    active_session = db.get_active_session()

    if sessions:
        session_names = {f"{s['session_name']} (ID: {s['id']}) - {s['status']}": s["id"] for s in sessions}
        active_idx = 0
        if active_session:
            for idx, sid in enumerate(session_names.values()):
                if sid == active_session["id"]:
                    active_idx = idx
                    break
        selected_session_label = st.selectbox("Current Session", options=list(session_names.keys()), index=active_idx)
        current_session_id = session_names[selected_session_label]
        current_session = db.get_session_by_id(current_session_id)
    else:
        current_session = None
        current_session_id = None

    # Option to create a new session
    st.markdown("##### ➕ New Session")
    new_session_name = st.text_input("New Session Name", placeholder="e.g. MCA_Sem3_Class1")
    if st.button("Create Session", type="secondary", use_container_width=True):
        if new_session_name.strip():
            new_id = db.create_attendance_session(new_session_name.strip())
            st.success(f"Session '{new_session_name.strip()}' created!")
            st.rerun()
        else:
            st.error("Please enter a session name.")

if not current_session:
    st.warning("No attendance sessions exist yet. Please create a session in the left sidebar to begin.")
    st.stop()

# Session Status Banner
banner_col1, banner_col2, banner_col3 = st.columns([3, 1, 1])
with banner_col1:
    st.markdown(f"### Active Session: **{current_session['session_name']}**")
    st.caption(f"Created: {current_session['created_at']} | Status: {current_session['status'].upper()}")

records = db.get_attendance_records(session_id=current_session_id)
unmarked = db.get_unmarked_students(session_id=current_session_id)

with banner_col2:
    st.metric("Present", len(records))
with banner_col3:
    st.metric("Remaining", len(unmarked))

st.divider()

# Main Attendance Section
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("🎙️ Student Voice Verification")
    st.info("Instruct the student to click record and say **\"Present\"** clearly into the microphone.")

    attendance_audio = st.audio_input("Record Attendance Utterance", key="live_attendance_audio")

    if attendance_audio is not None:
        st.write("##### 🎧 Utterance Preview")
        st.audio(attendance_audio)

        if st.button("🔍 Verify & Mark Attendance", type="primary", use_container_width=True):
            with st.spinner("Executing verification pipeline (Audio Check ➔ Anti-Spoof ➔ ASR ➔ Biometric Match)..."):
                audio_bytes = attendance_audio.getvalue()
                result = attendance_service.process_live_attendance(
                    audio_source=audio_bytes,
                    session_id=current_session_id,
                    similarity_threshold=similarity_threshold,
                    enable_anti_spoofing=enable_anti_spoofing,
                )

                decision = result.get("decision", "rejected")
                pipeline = result.get("pipeline") or {}
                latencies = result.get("latencies_ms") or {}
                total_latency = result.get("total_latency_ms") or 0.0

                # Pipeline Step Status Badges
                st.markdown("#### 🔍 Verification Pipeline Breakdown")
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)

                # The service names this stage audio_preprocessing.  Reading
                # audio_validation made every valid recording display FAIL.
                audio_step = pipeline.get("audio_preprocessing") or {}
                audio_ok = audio_step.get("ok", False)

                spoof_step = pipeline.get("anti_spoofing") or {}
                spoof_status = spoof_step.get("status", "skipped")
                spoof_ok = spoof_status in ("clear", "suspicious", "skipped")

                cmd_step = pipeline.get("command_verification") or {}
                cmd_ok = cmd_step.get("verified", False)

                ident_step = pipeline.get("speaker_identification") or {}
                # Speaker identification reports `matched`, not `ok`.
                ident_ok = ident_step.get("matched", False) and decision == "accepted"

                with p_col1:
                    if audio_ok:
                        st.success("1. Audio Quality\n\n**PASS** ✅")
                    else:
                        st.error("1. Audio Quality\n\n**FAIL** ❌")

                with p_col2:
                    if spoof_status == "clear":
                        st.success("2. Anti-Spoofing\n\n**GENUINE** 🛡️")
                    elif spoof_status == "suspicious":
                        st.warning("2. Anti-Spoofing\n\n**SUSPICIOUS** ⚠️")
                    elif spoof_status == "spoof_detected":
                        st.error("2. Anti-Spoofing\n\n**SPOOF** 🚫")
                    else:
                        st.info("2. Anti-Spoofing\n\n**SKIPPED** ⚪")

                with p_col3:
                    if cmd_ok:
                        st.success("3. Command (NLP)\n\n**VERIFIED** 🗣️")
                    else:
                        st.error("3. Command (NLP)\n\n**FAILED** ❌")

                with p_col4:
                    if ident_ok:
                        st.success("4. Biometric ID\n\n**MATCHED** 👤")
                    elif decision == "duplicate":
                        st.warning("4. Biometric ID\n\n**DUPLICATE** ⚠️")
                    else:
                        st.error("4. Biometric ID\n\n**REJECTED** ❌")

                # Decision Display
                st.markdown("---")
                if decision == "accepted":
                    student = result.get("student") or {}
                    cmd = result.get("command_result") or {}
                    similarity = result.get("similarity_score") or 0.0
                    session_name = result.get("session_name", "")
                    st.success("### ✅ ATTENDANCE GRANTED")
                    st.markdown(
                        f"""
                        - **Student:** **{student.get('name', 'Student')}** (`{student.get('student_code', '')}`)
                        - **Class / Section:** {student.get('class_section') or 'N/A'}
                        - **Cosine Similarity:** `{similarity:.4f}` (Threshold: `{similarity_threshold:.2f}`)
                        - **Speech Command:** Transcribed `"{cmd.get('normalized_transcription', '')}"` -> **VERIFIED**
                        - **Session:** {session_name}
                        """
                    )
                    st.balloons()
                elif decision == "duplicate":
                    student = result.get("student") or {}
                    st.warning("### ⚠️ ALREADY MARKED PRESENT")
                    st.markdown(
                        f"**{student.get('name', 'Student')}** (`{student.get('student_code', '')}`) "
                        f"has already been marked present in this session."
                    )
                else:
                    st.error("### ❌ ATTENDANCE REJECTED")
                    st.markdown(f"**Reason:** {result.get('message', 'Verification failed.')}")
                    identity = result.get("identification") or {}
                    best_candidate = identity.get("best_candidate") or {}
                    if best_candidate:
                        st.info(
                            f"Top biometric candidate: **{best_candidate.get('name', 'Unknown')}** "
                            f"with score **{best_candidate.get('score', 0.0):.4f}** "
                            f"(required: **{similarity_threshold:.2f}**). "
                            "Re-enroll the correct student if this is not the expected profile."
                        )
                    elif identity.get("reason"):
                        st.info(f"Biometric diagnostics: {identity['reason']}")

                # Latency Breakdown
                st.markdown("##### ⏱️ Pipeline Latencies")
                l_col1, l_col2, l_col3, l_col4, l_col5 = st.columns(5)
                with l_col1:
                    st.metric("Audio Check", f"{float(latencies.get('audio_validation') or 0):.1f} ms")
                with l_col2:
                    st.metric("Anti-Spoofing", f"{float(latencies.get('anti_spoofing') or 0):.1f} ms")
                with l_col3:
                    st.metric("Whisper ASR", f"{float(latencies.get('command_verification') or 0):.1f} ms")
                with l_col4:
                    spk_ms = float(latencies.get('speaker_embedding') or 0) + float(latencies.get('speaker_identification') or 0)
                    st.metric("Speaker ID", f"{spk_ms:.1f} ms")
                with l_col5:
                    st.metric("Total Latency", f"{float(total_latency):.1f} ms")

                # Diagnostic details
                with st.expander("🔎 Full Diagnostics & Ranking"):
                    cmd = result.get("command_result") or {}
                    if cmd:
                        st.write(f"- **Transcribed Phrase:** `\"{cmd.get('raw_transcription', '')}\"`")
                        st.write(f"- **Normalized Phrase:** `\"{cmd.get('normalized_transcription', '')}\"`")
                    asp = pipeline.get("anti_spoofing") or {}
                    if asp:
                        st.write(f"- **Anti-Spoofing Status:** `{asp.get('status')}` (Reason: {asp.get('reason')})")
                        if "checks" in asp and asp["checks"]:
                            st.json(asp["checks"])
                    ident = result.get("identification") or {}
                    if ident and ident.get("ranked_candidates"):
                        st.write(f"- **Biometric Decision:** `{ident.get('reason', 'Unavailable')}`")
                        st.write("- **Ranked Biometric Candidates:**")
                        st.dataframe(pd.DataFrame(ident["ranked_candidates"]), hide_index=True)

with right_col:
    st.subheader("📋 Session Attendee Log")
    if records:
        df_records = pd.DataFrame(records)[["student_code", "student_name", "similarity_score", "marked_at"]]
        df_records.columns = ["Student ID", "Name", "Score", "Marked Time"]
        st.dataframe(df_records, use_container_width=True, hide_index=True)
    else:
        st.info("No attendance recorded in this session yet.")

    st.markdown("##### ⏳ Unmarked Students")
    if unmarked:
        df_unmarked = pd.DataFrame(unmarked)[["student_code", "name"]]
        df_unmarked.columns = ["Student ID", "Name"]
        st.dataframe(df_unmarked, use_container_width=True, hide_index=True)
    else:
        st.success("All registered students are marked present!")
