# Running VoiceAttend AI

This guide explains how to install, start, test, and use the VoiceAttend AI Streamlit application on Windows.

## Requirements

- Windows 10 or Windows 11, 64-bit
- Python 3.10, 3.11, or 3.12
- A working microphone for enrollment and attendance verification
- Internet access for installing Python packages and downloading missing model files

The application is designed for CPU execution. A GPU is not required.

## 1. Open the project folder

Open PowerShell and move to the folder containing `app.py`:

```powershell
cd "D:\VOICEATTEND\VOICEATTEND - Copy"
```

If your project is stored somewhere else, replace the path with your own project path.

## 2. Create a virtual environment

Create the environment once:

```powershell
py -3.12 -m venv .venv
```

If Python 3.12 is not installed, use another supported version:

```powershell
py -3.11 -m venv .venv
```

Activate it for the current PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## 3. Install dependencies

Upgrade packaging tools and install the project requirements:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The project intentionally uses a compatible PyAV range for `faster-whisper`:

```text
av>=11,<19
```

Do not manually upgrade PyAV to version 19 or newer unless the speech-processing compatibility code is also updated.

## 4. Start the application

From the project folder, with `.venv` activated, run:

```powershell
python -m streamlit run app.py --server.port 8501
```

Open this address in your browser:

```text
http://localhost:8501
```

To stop the application, return to the PowerShell window and press `Ctrl+C`.

## 5. First login and account management

On the first launch, create the initial administrator account using the setup form.

After an administrator exists, the login page provides:

- **Sign in** — log in with an existing account.
- **Create account** — create a faculty account. Public registration cannot create administrators.
- **Reset password** — reset an existing account only after entering an active administrator username and password.

Passwords must contain at least 8 characters. Passwords are stored as hashes rather than plain text.

## 6. Basic application workflow

1. Open **Students** and add each student.
2. Open **Voice Enrollment** and record at least one clear sample for each student. Three samples are recommended.
3. Open **Dashboard** and create or open an attendance session.
4. Open **Live Attendance**.
5. Ask the enrolled student to say **“Present”** clearly.
6. Review the verification stages and attendance result.

The **Biometric similarity threshold** is available in the sidebar. The default is `0.72`:

- Lower values may reduce false rejections but can increase false acceptances.
- Higher values are stricter and may reject legitimate recordings affected by noise or microphone changes.
- Change the value before verification and use separately recorded evaluation audio when calibrating it.

## 7. Run the automated tests

With the virtual environment activated:

```powershell
python -m pytest -q
```

The tests cover authentication, database behavior, attendance rules, audio processing, and speaker similarity logic.

## 8. Optional speaker evaluation

Run the included evaluation dataset with the default threshold:

```powershell
python evaluation\evaluate_speakers.py --dataset_csv evaluation\dataset_manifest.csv
```

Evaluate a different threshold and enrollment sample counts:

```powershell
python evaluation\evaluate_speakers.py `
  --dataset_csv evaluation\dataset_manifest.csv `
  --threshold 0.75 `
  --sample_splits 1 3 5
```

Keep enrollment and test recordings separate when evaluating recognition performance.

## 9. Configuration overrides

The default configuration is in `config.py`. You can override common settings for a PowerShell session:

```powershell
$env:VOICEATTEND_SIMILARITY_THRESHOLD = "0.72"
$env:WHISPER_MODEL_SIZE = "tiny"
$env:VOICEATTEND_DB_PATH = "D:\VOICEATTEND\VOICEATTEND - Copy\data\voiceattend.db"
```

Then start Streamlit in the same PowerShell window. The sidebar threshold control can adjust the active session value without editing source code.

## Troubleshooting

### `python` or `py` is not recognized

Install Python 3.10–3.12 from python.org and enable **Add Python to PATH** during installation. Reopen PowerShell afterward.

### Microphone recording does not work

Check Windows microphone permissions, select the correct input device, and reload the browser page. Use a quiet environment and speak for approximately 1–6 seconds.

### Speech recognition reports an `metadata_errors` argument error

Confirm that the virtual environment is active and reinstall the compatible audio dependency:

```powershell
python -m pip install --upgrade "av>=11,<19"
```

### Biometric ID is rejected

Make sure the student has voice samples enrolled, record the live sample using the same microphone when possible, reduce background noise, and review the sidebar threshold. A rejected result can be correct when the speaker is not enrolled or the similarity score is below the threshold.

### The database or model files are missing

Start the application once from the project root. The application creates required data directories and initializes SQLite. Missing pretrained model files may be downloaded on first use, so internet access may be required.

### Port 8501 is already in use

Start on another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

Then open `http://localhost:8502`.

