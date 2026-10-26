"""Database access layer for VoiceAttend.

Provides SQLite persistence with foreign key enforcement, parameterized queries,
safe connection handling, and comprehensive CRUD for students, voice samples,
attendance sessions, and attendance records.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import config

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and row mapping."""
    target_path = Path(db_path) if db_path else config.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields connection and guarantees connection closure."""
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes from schema.sql."""
    with get_db(db_path) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


# ==============================================================================
# Student Operations
# ==============================================================================


def create_student(
    student_code: str, name: str, class_section: Optional[str] = None, db_path: Optional[Path] = None
) -> int:
    """Register a new student with unique student code."""
    student_code = student_code.strip()
    name = name.strip()
    class_section = class_section.strip() if class_section else None

    if not student_code:
        raise ValueError("Student code / ID cannot be empty.")
    if not name:
        raise ValueError("Student name cannot be empty.")

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (student_code, name, class_section)
            VALUES (?, ?, ?)
            """,
            (student_code, name, class_section),
        )
        conn.commit()
        return cursor.lastrowid


def get_student_by_id(student_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve student dictionary by primary key."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.*, 
                   COUNT(v.id) AS sample_count
            FROM students s
            LEFT JOIN voice_samples v ON s.id = v.student_id AND v.embedding_status = 'active'
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (student_id,),
        ).fetchone()
        return dict(row) if row else None


def get_student_by_code(student_code: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve student dictionary by unique student_code."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.*,
                   COUNT(v.id) AS sample_count
            FROM students s
            LEFT JOIN voice_samples v ON s.id = v.student_id AND v.embedding_status = 'active'
            WHERE s.student_code = ?
            GROUP BY s.id
            """,
            (student_code.strip(),),
        ).fetchone()
        return dict(row) if row else None


def get_all_students(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch all students ordered by student_code with sample count."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*,
                   COUNT(v.id) AS sample_count
            FROM students s
            LEFT JOIN voice_samples v ON s.id = v.student_id AND v.embedding_status = 'active'
            GROUP BY s.id
            ORDER BY s.student_code ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def search_students(query: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Search students by matching name or student_code."""
    pattern = f"%{query.strip()}%"
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*,
                   COUNT(v.id) AS sample_count
            FROM students s
            LEFT JOIN voice_samples v ON s.id = v.student_id AND v.embedding_status = 'active'
            WHERE s.name LIKE ? OR s.student_code LIKE ?
            GROUP BY s.id
            ORDER BY s.name ASC
            """,
            (pattern, pattern),
        ).fetchall()
        return [dict(r) for r in rows]


def update_student(
    student_id: int, name: str, class_section: Optional[str] = None, db_path: Optional[Path] = None
) -> bool:
    """Update name and class/section for an existing student."""
    name = name.strip()
    class_section = class_section.strip() if class_section else None
    if not name:
        raise ValueError("Student name cannot be empty.")

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE students
            SET name = ?, class_section = ?
            WHERE id = ?
            """,
            (name, class_section, student_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_student(
    student_id: int, delete_audio_files: bool = True, db_path: Optional[Path] = None
) -> bool:
    """Delete student, cascades to voice_samples and attendance_records, cleans up audio files."""
    audio_paths_to_delete: List[str] = []
    with get_db(db_path) as conn:
        if delete_audio_files:
            rows = conn.execute(
                "SELECT audio_path FROM voice_samples WHERE student_id = ?",
                (student_id,),
            ).fetchall()
            audio_paths_to_delete = [r["audio_path"] for r in rows if r["audio_path"]]

        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

    if deleted and delete_audio_files:
        for file_path in audio_paths_to_delete:
            try:
                p = Path(file_path)
                if p.exists() and p.is_file():
                    p.unlink()
            except OSError:
                pass

    return deleted


# ==============================================================================
# Voice Sample Operations
# ==============================================================================


def add_voice_sample(
    student_id: int,
    audio_path: str,
    embedding_blob: bytes,
    sample_number: int,
    model_identifier: str,
    db_path: Optional[Path] = None,
) -> int:
    """Store an enrollment sample and its embedding associated with the student."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO voice_samples (
                student_id, audio_path, embedding_blob,
                sample_number, model_identifier, embedding_status
            )
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            (student_id, audio_path, embedding_blob, sample_number, model_identifier),
        )
        conn.commit()
        return cursor.lastrowid


def get_voice_samples_for_student(
    student_id: int, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieve all active voice samples for a specific student."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM voice_samples
            WHERE student_id = ? AND embedding_status = 'active'
            ORDER BY sample_number ASC
            """,
            (student_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_active_voice_samples(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all active voice samples joined with student info."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT v.*, s.student_code, s.name as student_name
            FROM voice_samples v
            JOIN students s ON v.student_id = s.id
            WHERE v.embedding_status = 'active'
            ORDER BY v.student_id, v.sample_number ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def delete_voice_sample(
    sample_id: int, delete_file: bool = True, db_path: Optional[Path] = None
) -> bool:
    """Delete a single voice sample and its associated audio file."""
    audio_path = None
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT audio_path FROM voice_samples WHERE id = ?", (sample_id,)
        ).fetchone()
        if row and row["audio_path"]:
            audio_path = row["audio_path"]

        cursor = conn.cursor()
        cursor.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

    if deleted and delete_file and audio_path:
        try:
            p = Path(audio_path)
            if p.exists() and p.is_file():
                p.unlink()
        except OSError:
            pass

    return deleted


# ==============================================================================
# Attendance Session Operations
# ==============================================================================


def create_attendance_session(session_name: str, db_path: Optional[Path] = None) -> int:
    """Create a new attendance session."""
    session_name = session_name.strip()
    if not session_name:
        raise ValueError("Session name cannot be empty.")

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance_sessions (session_name, status)
            VALUES (?, 'active')
            """,
            (session_name,),
        )
        conn.commit()
        return cursor.lastrowid


def get_active_session(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent active attendance session."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM attendance_sessions
            WHERE status = 'active'
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None


def get_all_sessions(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all attendance sessions with attendee count."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, COUNT(r.id) as attendance_count
            FROM attendance_sessions s
            LEFT JOIN attendance_records r ON s.id = r.session_id
            GROUP BY s.id
            ORDER BY s.created_at DESC, s.id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_by_id(session_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a session by its ID."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM attendance_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def close_session(session_id: int, db_path: Optional[Path] = None) -> bool:
    """Mark an attendance session as closed."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE attendance_sessions SET status = 'closed' WHERE id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


# ==============================================================================
# Attendance Record Operations
# ==============================================================================


def mark_attendance(
    student_id: int,
    session_id: int,
    similarity_score: float,
    command_verified: bool,
    decision: str = "accepted",
    db_path: Optional[Path] = None,
) -> int:
    """Record verified attendance for a student in a session.
    
    Raises sqlite3.IntegrityError if the student is already marked in this session.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance_records (
                student_id, session_id, similarity_score,
                command_verified, decision
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                student_id,
                session_id,
                float(similarity_score),
                1 if command_verified else 0,
                decision,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def is_student_marked(student_id: int, session_id: int, db_path: Optional[Path] = None) -> bool:
    """Check if student has already been marked present in the session."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT id FROM attendance_records
            WHERE student_id = ? AND session_id = ?
            """,
            (student_id, session_id),
        ).fetchone()
        return row is not None


def get_attendance_records(
    session_id: Optional[int] = None,
    student_id: Optional[int] = None,
    date_str: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Query attendance records with optional filtering."""
    query = """
        SELECT r.*,
               s.student_code, s.name as student_name, s.class_section,
               sess.session_name
        FROM attendance_records r
        JOIN students s ON r.student_id = s.id
        JOIN attendance_sessions sess ON r.session_id = sess.id
        WHERE 1=1
    """
    params: List[Any] = []

    if session_id is not None:
        query += " AND r.session_id = ?"
        params.append(session_id)

    if student_id is not None:
        query += " AND r.student_id = ?"
        params.append(student_id)

    if date_str:
        query += " AND DATE(r.marked_at) = DATE(?)"
        params.append(date_str)

    query += " ORDER BY r.marked_at DESC"

    with get_db(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_unmarked_students(session_id: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List students who have NOT yet been marked present in the session."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, COUNT(v.id) as sample_count
            FROM students s
            LEFT JOIN voice_samples v ON s.id = v.student_id AND v.embedding_status = 'active'
            WHERE s.id NOT IN (
                SELECT student_id FROM attendance_records WHERE session_id = ?
            )
            GROUP BY s.id
            ORDER BY s.student_code ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
