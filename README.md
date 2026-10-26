# VoiceAttend: A Few-Shot Voice-Based Attendance System

**VoiceAttend** is a local, privacy-preserving, few-shot voice biometric attendance application designed for Windows laptops with CPU-only execution. Developed as an MCA Deep Learning + NLP project, it pairs **SpeechBrain ECAPA-TDNN** speaker recognition embeddings with **faster-whisper** speech-command verification to authenticate students when they say **"Present"**.

---

## 🌟 Key Features

1. **Student-Specific Direct Voice Enrollment**:
   - Administrators select an enrolled student and record 2–4 second voice samples directly from the web microphone interface.
   - Every audio recording and speaker embedding is strictly bound to the student's unique database primary key.
   - Supports 3–5 utterance samples per student to construct an aggregated, L2-normalized centroid voice profile.
   - Individual samples can be reviewed, played back, or deleted.

2. **Pretrained Deep Learning Speaker Recognition**:
   - Uses **SpeechBrain ECAPA-TDNN** (pretrained on VoxCeleb) to extract robust 192-dimensional speaker embeddings.
   - Computes cosine similarity against enrolled student profiles.
   - Rejects impostors and uncertain voices using a configurable similarity threshold (default: `0.72`).
   - Pure CPU inference with local caching — no external cloud API calls or subscriptions.

3. **NLP Speech-Command Verification**:
   - Transcribes speech locally via **faster-whisper** (quantized `tiny` model on CPU).
   - Normalizes spoken text and enforces that the student says **"Present"** (or approved variations like *"Present Sir"*, *"Present Ma'am"*, *"Here"*).
   - Operates independently from the speaker recognition model to prevent attendance marking on non-attendance utterances.

4. **Session-Based Attendance & Duplicate Prevention**:
   - Prevents marking the same student twice in the same attendance session using SQLite integrity constraints and transaction checks.
   - Real-time attendee log and dynamic remaining student tracker.

5. **Local SQLite Persistence**:
   - Parameterized queries with `PRAGMA foreign_keys = ON;`.
   - Embeddings stored safely as raw binary `float32` BLOBs (zero `eval()` or unpickling risks).
   - Audio files stored on disk under `data/enrollment/` with cascading cleanup upon student deletion.

6. **Interactive Streamlit Multi-Page UI**:
   - **Dashboard**: High-level enrollment health, active session stats, and student progress.
   - **Students**: Searchable student registry, addition, editing, and deletion.
   - **Voice Enrollment**: Direct microphone capture, audio waveform preview, and sample management.
   - **Live Attendance**: Real-time voice verification with immediate decision pills (Accepted / Rejected / Duplicate).
   - **Attendance Reports**: Multi-filter logs (session, student, date) and one-click CSV export.
   - **Model & System Status**: Health checks for speech models, SQLite database, and local environment.

7. **Comprehensive Evaluation Suite**:
   - `evaluation/evaluate_speakers.py` measures identification accuracy, false acceptance rate (FAR), false rejection rate (FRR), unknown rejection rate, command accuracy, and inference latency.
   - Benchmarks profile quality across **1, 3, and 5 enrollment samples**.

---

## 🏛️ System Architecture

```
Voice Utterance (Microphone)
       │
       ├──► Audio Preprocessing (16 kHz mono, duration & RMS energy validation)
       │           │
       │           ├──► [NLP Component] faster-whisper (tiny int8 CPU)
       │           │         │
       │           │         └──► Transcription Normalization & "Present" Verification
       │           │
       │           └──► [Deep Learning] SpeechBrain ECAPA-TDNN (VoxCeleb)
       │                     │
       │                     └──► 192-dim Unit Vector Embedding Extraction
       │                               │
       │                               ▼
       │                    Cosine Similarity Matching against Enrolled Profiles
       │                               │
       │                               ▼
       └────────────────► Decision Logic:
                           1. Is speech command valid?
                           2. Does top cosine similarity >= threshold (0.72)?
                           3. Has student already attended this session?
                                       │
                                       ▼
                       [Persist in SQLite Attendance Records]
```

---

## 📁 Project Structure

```
VOICEATTEND/
├── app.py                      # Streamlit application entrypoint
├── config.py                   # Centralized configuration and path management
├── requirements.txt            # Dependency specification
├── .gitignore                  # Git exclusions for models, databases, and audio
├── README.md                   # System documentation and user manual
│
├── audio/                      # Audio capture, validation, and preprocessing
│   ├── __init__.py
│   ├── preprocessing.py        # 16kHz resampling, RMS energy, and WAV saving
│   └── recorder.py             # Streamlit buffer handling and filename generator
│
├── speaker/                    # Deep Learning Speaker Recognition
│   ├── __init__.py
│   ├── model_service.py        # SpeechBrain ECAPA-TDNN adapter
│   ├── identification.py       # Cosine similarity and threshold matching
│   └── embedding_store.py      # SQLite BLOB serialization and profile aggregation
│
├── speech/                     # NLP Speech Command Verification
│   ├── __init__.py
│   └── command_verifier.py     # faster-whisper transcription & text matching
│
├── database/                   # SQLite Storage Layer
│   ├── __init__.py
│   ├── db.py                   # CRUD operations with foreign key enforcement
│   └── schema.sql              # Database schema definition
│
├── services/                   # High-Level Business Orchestration
│   ├── __init__.py
│   └── attendance_service.py   # Enrollment and live attendance pipelines
│
├── pages/                      # Streamlit Multi-Page Views
│   ├── 1_Dashboard.py          # Metrics, session monitor, and enrollment summary
│   ├── 2_Students.py           # Student registry and profile overview
│   ├── 3_Voice_Enrollment.py   # Direct microphone recording & sample manager
│   ├── 4_Live_Attendance.py    # Real-time attendance verification interface
│   ├── 5_Attendance_Reports.py # Attendance logs, filters, and CSV export
│   └── 6_System_Status.py      # Pretrained model status and diagnostics
│
├── evaluation/                 # Benchmark & Evaluation Harness
│   ├── evaluate_speakers.py    # FAR, FRR, accuracy, and sample-count comparison
│   └── README.md               # Manifest format and benchmark instructions
│
├── data/                       # Local application storage (excluded from git)
│   ├── enrollment/             # Student WAV files
│   ├── profiles/               # Cached embeddings
│   └── pretrained_models/      # Downloaded model weights
│
└── tests/                      # Automated Test Suite
    ├── test_database.py        # SQLite schema, CRUD, and duplicate constraints
    ├── test_similarity.py      # Cosine similarity, normalization, and ranking
    └── test_attendance.py      # Preprocessing, command verification, and service checks
```

---

## ⚙️ Installation & Setup (Windows PowerShell)

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.10 – 3.12 (Python 3.12 recommended)
- Microphone connected to your PC
- Internet connection for initial package and model weights download (offline afterwards)

### 1. Clone & Navigate to Project Directory
```powershell
cd "E:\MCA SEM3 PROJECT\VOICEATTEND"
```

### 2. Create and Activate Virtual Environment
```powershell
# Using uv (fastest)
uv venv --python 3.12 .venv
.\.venv\Scripts\activate

# Or using standard python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
# Install PyTorch CPU first (ensures low footprint without CUDA overhead)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📖 Step-by-Step Demonstration Walkthrough

### Step 1: Register Students
1. Navigate to **Students** (`pages/2_Students.py`) in the left sidebar.
2. Select the **➕ Add New Student** tab.
3. Register 2–3 test students:
   - Student ID: `MCA2026-001`, Name: `Rahul Sharma`, Class: `MCA Sem 3`
   - Student ID: `MCA2026-002`, Name: `Priya Patel`, Class: `MCA Sem 3`
   - Student ID: `MCA2026-003`, Name: `Amit Kumar`, Class: `MCA Sem 3`

### Step 2: Enroll Voice Profiles (Direct Microphone Recording)
1. Navigate to **Voice Enrollment** (`pages/3_Voice_Enrollment.py`).
2. Select a student (e.g. `Rahul Sharma`).
3. Click the **Record** button on the microphone input widget.
4. Have the student clearly say: **"Present"** (approx. 2–3 seconds).
5. Stop the recording and listen to the preview audio.
6. Click **✅ Accept & Enroll Sample**.
7. Repeat the process to collect **3 samples** for Rahul.
8. The student's status badge will update from **🔴 Not Enrolled** to **🟢 Enrolled (3/3 samples)**.
9. Enroll 3 samples for `Priya Patel` as well.

### Step 3: Start an Attendance Session
1. Navigate to **Live Attendance** (`pages/4_Live_Attendance.py`).
2. In the left sidebar under **New Session**, enter a session name (e.g. `MCA_DL_Lecture_01`) and click **Create Session**.

### Step 4: Verify Live Attendance
1. Have `Rahul Sharma` click the microphone record button and say: **"Present"**.
2. Click **🔍 Verify & Mark Attendance**.
3. **Outcome**:
   - The NLP verifier transcribes and approves `"present"`.
   - The ECAPA-TDNN model compares the embedding against enrolled profiles.
   - Decision: **✅ ATTENDANCE GRANTED** with student name, ID, and similarity score (~`0.80+`).
4. **Duplicate Attendance Test**:
   - Have Rahul record and submit again in the same session.
   - Decision: **⚠️ ALREADY MARKED PRESENT** — duplicate entry is prevented.
5. **Unknown Speaker Rejection Test**:
   - Have an unenrolled friend or test speaker record saying `"Present"`.
   - Decision: **❌ ATTENDANCE REJECTED** — similarity score is below the threshold (`< 0.72`).
6. **Wrong Command Test**:
   - Have an enrolled student say `"Good morning"`.
   - Decision: **❌ ATTENDANCE REJECTED** — speech command verification fails.

### Step 5: View and Export Attendance Reports
1. Navigate to **Attendance Reports** (`pages/5_Attendance_Reports.py`).
2. Filter records by session (`MCA_DL_Lecture_01`).
3. View the attendance table with timestamps, similarity scores, and command verification flags.
4. Click **📥 Export Report to CSV** to download the official attendance sheet.

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:

```powershell
pytest -v tests/
```

Test coverage includes:
- `test_database.py`: Verifies foreign key constraints, unique student codes, sample persistence, cascade deletion, and duplicate session rejection.
- `test_similarity.py`: Tests cosine similarity mathematical properties, L2 vector normalization, profile centroid aggregation, and top-candidate ranking.
- `test_attendance.py`: Validates audio length and RMS silence rejection, command text normalization, and error handling for missing sessions.

---

## 📊 Benchmarking & Model Evaluation

Run the evaluation module to measure system metrics:

```powershell
python evaluation/evaluate_speakers.py --dataset_csv evaluation/dataset_manifest.csv --threshold 0.72 --sample_splits 1 3 5
```

The script evaluates:
- Identification Accuracy (%)
- False Acceptance Rate (FAR %)
- False Rejection Rate (FRR %)
- Accuracy progression across 1, 3, and 5 enrollment samples
- Average inference latency (ms) for Whisper and ECAPA-TDNN

---

## 🔒 Privacy, Security, & Limitations

- **Local Storage Only**: All audio recordings and voice embeddings are saved exclusively on the local machine (`data/`). No biometric data is sent to external servers or cloud services.
- **Data Subject Rights**: Deleting a student via the management screen automatically deletes their database records, embedding vectors, and associated audio files on disk.
- **Biometric Security Limitations**:
  - Voice biometrics alone does not provide physical presence proof against high-fidelity audio replay attacks (e.g. pre-recorded voice playback on loudspeakers).
  - Background acoustic noise and microphone variations may impact similarity scores; keep microphone hardware consistent during enrollment and attendance.

---

## VoiceAttend AI UI structure

The premium visual system is centralized in [`ui.py`](./ui.py), with the local
brand mark in [`assets/voiceattend_logo.svg`](./assets/voiceattend_logo.svg).
The application entry point uses one explicit Streamlit navigation tree with
grouped Dashboard, Attendance, Management, and Intelligence links.

The split-screen login gate and shared dashboard renderer in
[`dashboard_view.py`](./dashboard_view.py) use the existing authentication and
SQLite data layers. Missing verification history is shown as unavailable rather
than replaced with fabricated values.
