"""VoiceAttend - Student-Specific Voice Enrollment.

CRITICAL WORKFLOW:
Administrators select an existing student, record 2-4 second speech samples directly
via the microphone, validate audio quality, extract ECAPA-TDNN speaker embeddings,
and store the sample associated strictly with the selected student's primary key.
"""

from pathlib import Path
import streamlit as st
import config
from audio.preprocessing import AudioValidationError
from database import db
from services import attendance_service, auth
from speaker import model_service
st.set_page_config(page_title="Voice Enrollment - VoiceAttend", page_icon="🎙️", layout="wide")
db.init_db()

st.title("🎙️ Student Voice Enrollment")
st.caption("Direct microphone enrollment — build speaker voice profiles with 3–5 utterance samples")

# Check Model Availability
model_info = model_service.get_model_info()
if not model_info["available"]:
    st.warning(
        f"⚠️ Speaker Recognition Model is not ready: {model_info.get('error') or 'Initializing...'}\n"
        "Voice samples can only be enrolled when the speaker model is loaded."
    )

# 1. Student Selection
students = db.get_all_students()
if not students:
    st.info("No students registered yet. Please add students first.")
    if st.button("Go to Student Management"):
        st.switch_page("pages/2_Students.py")
    st.stop()

# Build student selector map
student_options = {
    f"{s['student_code']} — {s['name']} ({s.get('sample_count', 0)}/{config.DEFAULT_ENROLLMENT_TARGET} samples)": s["id"]
    for s in students
}

# Pre-select student if navigated from Students page
default_index = 0
preselected_id = st.session_state.get("selected_student_id")
if preselected_id:
    for idx, (label, sid) in enumerate(student_options.items()):
        if sid == preselected_id:
            default_index = idx
            break

selected_label = st.selectbox(
    "Select Student to Enroll *",
    options=list(student_options.keys()),
    index=default_index,
)
selected_student_id = student_options[selected_label]
selected_student = db.get_student_by_id(selected_student_id)

if not selected_student:
    st.error("Student not found.")
    st.stop()

# 2. Display Student Details Card
sample_count = selected_student.get("sample_count", 0)
card_col1, card_col2, card_col3 = st.columns(3)

with card_col1:
    st.markdown(f"### **{selected_student['name']}**")
    st.write(f"**Student ID:** `{selected_student['student_code']}`")
    st.write(f"**Class/Section:** {selected_student['class_section'] or 'Not Specified'}")

with card_col2:
    st.markdown("### **Enrollment Status**")
    if sample_count >= config.DEFAULT_ENROLLMENT_TARGET:
        st.success(f"✅ Complete Profile ({sample_count}/{config.DEFAULT_ENROLLMENT_TARGET} samples)")
    elif sample_count > 0:
        st.warning(f"⚠️ Partial Profile ({sample_count}/{config.DEFAULT_ENROLLMENT_TARGET} samples)")
    else:
        st.error("❌ Not Enrolled (0 samples)")

with card_col3:
    st.markdown("### **Sample Progress**")
    progress_val = min(1.0, float(sample_count) / config.DEFAULT_ENROLLMENT_TARGET)
    st.progress(progress_val, text=f"{sample_count} of {config.DEFAULT_ENROLLMENT_TARGET} target samples (max: {config.MAX_ENROLLMENT_SAMPLES})")

st.divider()

# 3. Voice Recording Interface
st.subheader("🎤 Record Enrollment Sample")
st.info(
    "🗣️ **Prompt the student:** 'Please speak clearly into the microphone and say **\"Present\"** naturally.'\n\n"
    "Target duration: **2 to 4 seconds**. Ensure low ambient background noise."
)

if sample_count >= config.MAX_ENROLLMENT_SAMPLES:
    st.warning(
        f"Maximum enrollment limit reached ({config.MAX_ENROLLMENT_SAMPLES} samples). "
        "Remove an existing sample below if you wish to record a replacement."
    )
else:
    # Native Streamlit audio recorder (browser-based microphone capture)
    audio_data = st.audio_input("Record Voice Sample", key=f"audio_input_{selected_student_id}_{sample_count}")

    if audio_data is not None:
        st.write("##### 🎧 Preview Recording")
        st.audio(audio_data)

        # Accept / Discard Actions
        act_col1, act_col2 = st.columns([1, 4])
        with act_col1:
            if st.button("✅ Accept & Enroll Sample", type="primary", use_container_width=True):
                with st.spinner("Validating audio and extracting speaker embedding..."):
                    try:
                        audio_bytes = audio_data.getvalue()
                        result = attendance_service.enroll_student_sample(
                            student_id=selected_student_id,
                            audio_source=audio_bytes,
                        )
                        st.success(f"🎉 {result['message']}")
                        st.rerun()
                    except AudioValidationError as ave:
                        st.error(f"Validation Failed: {ave}")
                    except attendance_service.ServiceError as se:
                        st.error(f"Enrollment Error: {se}")
                    except Exception as err:
                        st.error(f"Unexpected Error: {err}")

        with act_col2:
            st.caption("Click 'Record' again on the audio widget above if you wish to discard and re-record.")

st.divider()

# 4. Existing Voice Samples List
st.subheader(f"📁 Enrolled Samples ({sample_count})")
existing_samples = db.get_voice_samples_for_student(selected_student_id)

if not existing_samples:
    st.caption("No samples enrolled for this student yet.")
else:
    for sample in existing_samples:
        samp_id = sample["id"]
        samp_num = sample["sample_number"]
        samp_path = Path(sample["audio_path"])
        samp_date = sample["created_at"]
        model_id = sample["model_identifier"]

        with st.container():
            s_col1, s_col2, s_col3 = st.columns([2, 3, 1])
            with s_col1:
                st.write(f"**Sample #{samp_num}** (Recorded: {samp_date})")
                st.caption(f"Model: {model_id}")
            with s_col2:
                if samp_path.exists():
                    st.audio(str(samp_path))
                else:
                    st.warning("Audio file missing on disk.")
            with s_col3:
                if st.button(f"🗑️ Delete", key=f"del_sample_{samp_id}"):
                    attendance_service.delete_student_sample(samp_id)
                    st.success(f"Sample #{samp_num} deleted.")
                    st.rerun()
            st.markdown("---")
