"""VoiceAttend - Model & System Status.

Diagnostics screen displaying the availability, caching status, and health
of the speaker recognition model, speech command model, anti-spoofing heuristics,
database storage, and administrative audit logs.
"""

import sys
import streamlit as st
import pandas as pd
import config
from database import db
from services import anti_spoofing, auth
from speaker import model_service
from speech import command_verifier
st.set_page_config(page_title="System Status - VoiceAttend", page_icon="⚙️", layout="wide")
db.init_db()

st.title("⚙️ Model & System Diagnostics")
st.caption("Inspect deep learning models, heuristic pipelines, database storage, and audit trails")

# Action: Reload / Preload Models
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🔄 Test / Load Models", type="primary", use_container_width=True):
        with st.spinner("Initializing models..."):
            model_service.load_model()
            command_verifier.load_command_model()
            st.rerun()

st.divider()

# 1. Speaker Recognition Model Status
st.subheader("🗣️ Speaker Recognition Model (Deep Learning)")
spk_info = model_service.get_model_info()

spk_col1, spk_col2 = st.columns(2)
with spk_col1:
    st.write(f"**Model Architecture:** SpeechBrain ECAPA-TDNN")
    st.write(f"**Pretrained Weights:** `{spk_info['source']}`")
    st.write(f"**Inference Device:** `{spk_info['device']}`")
    st.write(f"**Embedding Dimension:** `{spk_info['embedding_dim']}`")
    st.write(f"**Local Cache Dir:** `{spk_info['cache_dir']}`")

with spk_col2:
    if spk_info["available"]:
        st.success("✅ **Model Status: AVAILABLE**\n\nReady for enrollment and real-time attendance verification.")
    else:
        st.error(f"❌ **Model Status: UNAVAILABLE**\n\n{spk_info.get('error', 'Model not initialized.')}")
        st.info("Ensure `speechbrain` and `torch` are installed in the active environment.")

st.divider()

# 2. Speech Command Verification Model Status
st.subheader("💬 Speech Command Verification (NLP)")
cmd_info = command_verifier.get_model_status()

cmd_col1, cmd_col2 = st.columns(2)
with cmd_col1:
    st.write(f"**Engine:** faster-whisper")
    st.write(f"**Model Size:** `{cmd_info['model_name']}`")
    st.write(f"**Execution:** `{cmd_info['device']}`")
    st.write(f"**Target Command:** `{config.DEFAULT_ATTENDANCE_COMMAND}`")
    st.write(f"**Approved Variants:** `{', '.join(config.APPROVED_COMMAND_VARIANTS[:4])}...`")

with cmd_col2:
    if cmd_info["available"]:
        st.success("✅ **Model Status: AVAILABLE**\n\nReady to transcribe and verify attendance commands.")
    else:
        st.error(f"❌ **Model Status: UNAVAILABLE**\n\n{cmd_info.get('error', 'Model not loaded.')}")

st.divider()

# 3. Anti-Spoofing & Liveness Heuristics
st.subheader("🛡️ Anti-Spoofing & Liveness Pipeline")
spoof_info = anti_spoofing.get_anti_spoofing_status()
sp_col1, sp_col2 = st.columns(2)

with sp_col1:
    st.write(f"**Mode:** `{spoof_info['mode'].upper()}`")
    st.write(f"**Status:** `{'AVAILABLE' if spoof_info['available'] else 'UNAVAILABLE'}`")
    st.write(f"**Clipping Threshold:** `{anti_spoofing.CLIP_FRACTION_THRESHOLD * 100:.1f}%`")
    st.write(f"**Silence RMS Floor:** `{anti_spoofing.MIN_ENERGY_RMS}`")
    st.write(
        f"**Spectral Flatness Window:** `[{anti_spoofing.SPECTRAL_FLATNESS_MIN:.2f}, "
        f"{anti_spoofing.SPECTRAL_FLATNESS_MAX:.2f}]`"
    )

with sp_col2:
    st.info(
        f"💡 **Pipeline Architecture:**\n\n"
        f"{spoof_info['warning']}\n\n"
        f"Checks: Hard clipping, DC bias drift, spectral flatness anomalies, silence detection, and SNR."
    )

st.divider()

# 4. Database & Storage Health
st.subheader("💾 Database & Storage")
db_col1, db_col2 = st.columns(2)

with db_col1:
    st.write(f"**Database File:** `{config.DB_PATH}`")
    st.write(f"**Enrollment Audio Dir:** `{config.ENROLLMENT_AUDIO_DIR}`")
    st.write(f"**Profiles Dir:** `{config.PROFILES_DIR}`")

with db_col2:
    try:
        all_students = db.get_all_students()
        all_samples = db.get_all_active_voice_samples()
        all_records = db.get_attendance_records()

        st.metric("Total Students in DB", len(all_students))
        st.metric("Total Active Voice Samples", len(all_samples))
        st.metric("Total Attendance Records", len(all_records))
        st.success("✅ SQLite Database connection healthy and responsive.")
    except Exception as db_err:
        st.error(f"❌ Database Error: {db_err}")

st.divider()

# 5. System Audit Trail (Admin Only)
st.subheader("📜 System Audit Log")
if auth.has_role("admin"):
    try:
        audit_logs = auth.get_recent_audit_logs(limit=20)
        if audit_logs:
            df_audit = pd.DataFrame(audit_logs)[["id", "timestamp", "username", "action", "details"]]
            df_audit.columns = ["ID", "Timestamp", "User", "Action", "Details"]
            st.dataframe(df_audit, use_container_width=True, hide_index=True)
        else:
            st.info("No audit log events recorded yet.")
    except Exception as log_err:
        st.error(f"Could not load audit logs: {log_err}")
else:
    st.caption("🔒 Audit logs require Administrator privileges to view.")

st.divider()

# 6. Runtime Environment
st.subheader("💻 Environment Diagnostics")
env_col1, env_col2 = st.columns(2)
with env_col1:
    st.write(f"**Python Version:** {sys.version.split()[0]}")
    st.write(f"**Platform:** {sys.platform}")
    st.write(f"**Audio Sample Rate:** {config.AUDIO_SAMPLE_RATE} Hz (16 kHz mono)")
with env_col2:
    st.write(f"**Default Cosine Threshold:** {config.DEFAULT_SIMILARITY_THRESHOLD}")
    st.write(f"**Enrollment Target:** {config.DEFAULT_ENROLLMENT_TARGET} samples (max {config.MAX_ENROLLMENT_SAMPLES})")
