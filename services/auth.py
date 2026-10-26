"""Administrator authentication and role-based access for VoiceAttend.

Security design:
  - Passwords are stored as bcrypt hashes — never plaintext.
  - Admin credentials are stored in the SQLite database (not config files).
  - Sessions use Streamlit session_state with a server-side token.
  - No hardcoded production passwords. An initial setup wizard creates the first admin.
  - Login abuse protection: simple in-memory failed attempt tracking per session.

Roles:
  - 'admin'  : Full access — can create/delete students, manage enrollment, view audit log.
  - 'faculty' : Can start/close sessions and view reports. Cannot delete students.

Audit log: All security-sensitive operations are written to the audit_log DB table.
"""

import hashlib
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from database import db

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
MAX_LOGIN_ATTEMPTS = 5          # Lock after this many failures in one session
LOCKOUT_DURATION_SEC = 300      # 5 minutes lock-out
SESSION_TOKEN_BYTES = 32        # 256-bit session token

# Try to import bcrypt; fall back to a SHA-256 scheme if not installed
try:
    import bcrypt
    _HAS_BCRYPT = True
except ImportError:
    _HAS_BCRYPT = False
    logger.warning(
        "bcrypt is not installed. Falling back to SHA-256 password hashing. "
        "Run: pip install bcrypt   for stronger password security."
    )


# ─── Password hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    if _HAS_BCRYPT:
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
        return hashed.decode("utf-8")
    else:
        # Fallback: SHA-256 with a random salt
        salt = secrets.token_hex(32)
        digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return f"sha256:{salt}:{digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    try:
        if stored_hash.startswith("sha256:"):
            _, salt, digest = stored_hash.split(":")
            return secrets.compare_digest(
                hashlib.sha256((salt + password).encode("utf-8")).hexdigest(),
                digest,
            )
        elif _HAS_BCRYPT:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        else:
            return False
    except Exception:
        return False


# ─── Database helpers ──────────────────────────────────────────────────────────

def _ensure_admin_tables(db_path: Optional[Path] = None) -> None:
    """Create admin_users and audit_log tables if they don't exist."""
    with db.get_db(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'faculty',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login_at TIMESTAMP,
                is_active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                username TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_hint TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
            """
        )
        conn.commit()


def create_admin_user(
    username: str,
    password: str,
    role: str = "admin",
    db_path: Optional[Path] = None,
) -> bool:
    """Create a new admin user. Returns True on success, False if username exists."""
    _ensure_admin_tables(db_path)
    if not username.strip() or not password:
        raise ValueError("Username and password cannot be empty.")
    if role not in ("admin", "faculty"):
        raise ValueError("Role must be 'admin' or 'faculty'.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    password_hash = hash_password(password)
    try:
        with db.get_db(db_path) as conn:
            conn.execute(
                "INSERT INTO admin_users (username, password_hash, role) VALUES (?, ?, ?)",
                (username.strip().lower(), password_hash, role),
            )
            conn.commit()
        return True
    except Exception:
        return False


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Authenticate a user and return their info dict or None."""
    _ensure_admin_tables(db_path)
    with db.get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM admin_users WHERE username = ? AND is_active = 1",
            (username.strip().lower(),),
        ).fetchone()

    if not row:
        return None

    user = dict(row)
    if not verify_password(password, user["password_hash"]):
        return None

    # Update last login time
    with db.get_db(db_path) as conn:
        conn.execute(
            "UPDATE admin_users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user["id"],),
        )
        conn.commit()

    return user


def has_any_admin(db_path: Optional[Path] = None) -> bool:
    """Return True if at least one admin user exists in the database."""
    _ensure_admin_tables(db_path)
    with db.get_db(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM admin_users WHERE is_active = 1"
        ).fetchone()
    return row["cnt"] > 0


def get_all_users(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all admin users (without password hashes)."""
    _ensure_admin_tables(db_path)
    with db.get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, username, role, created_at, last_login_at, is_active FROM admin_users"
        ).fetchall()
    return [dict(r) for r in rows]


def deactivate_user(user_id: int, db_path: Optional[Path] = None) -> bool:
    """Soft-delete an admin user."""
    with db.get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE admin_users SET is_active = 0 WHERE id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def change_password(
    user_id: int,
    new_password: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Update a user's password hash."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    new_hash = hash_password(new_password)
    with db.get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE admin_users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def reset_password_by_admin(
    target_username: str,
    new_password: str,
    admin_username: str,
    admin_password: str,
    db_path: Optional[Path] = None,
) -> bool:
    """Reset an account password after verifying an active administrator."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    _ensure_admin_tables(db_path)
    with db.get_db(db_path) as conn:
        admin = conn.execute(
            "SELECT username, role, password_hash FROM admin_users "
            "WHERE username = ? AND is_active = 1",
            (admin_username.strip().lower(),),
        ).fetchone()
        if not admin or admin["role"] != "admin" or not verify_password(admin_password, admin["password_hash"]):
            return False

        cursor = conn.execute(
            "UPDATE admin_users SET password_hash = ? WHERE username = ? AND is_active = 1",
            (hash_password(new_password), target_username.strip().lower()),
        )
        conn.commit()
        changed = cursor.rowcount > 0

    if changed:
        log_audit_event(
            "PASSWORD_RESET",
            f"Password reset for account: {target_username.strip().lower()}",
            username=admin["username"],
            db_path=db_path,
        )
    return changed


# ─── Audit logging ─────────────────────────────────────────────────────────────

def log_audit_event(
    action: str,
    details: str = "",
    username: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Append an entry to the audit log. Errors are swallowed to avoid cascading failures."""
    try:
        _ensure_admin_tables(db_path)
        if username is None:
            # Attempt to get from session state (safe fallback)
            try:
                username = st.session_state.get("auth_username", "unknown")
            except Exception:
                username = "unknown"
        with db.get_db(db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (username, action, details) VALUES (?, ?, ?)",
                (username, action, details),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to write audit log: {e}")


def get_audit_log(
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent audit log entries."""
    _ensure_admin_tables(db_path)
    with db.get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Streamlit session management ─────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise session-state keys for auth."""
    defaults = {
        "auth_logged_in": False,
        "auth_username": None,
        "auth_role": None,
        "auth_token": None,
        "auth_failed_attempts": 0,
        "auth_lockout_until": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_locked_out() -> bool:
    _init_session_state()
    lockout_until = st.session_state.get("auth_lockout_until", 0.0)
    return time.time() < lockout_until


def remaining_lockout_seconds() -> float:
    lockout_until = st.session_state.get("auth_lockout_until", 0.0)
    return max(0.0, lockout_until - time.time())


def is_authenticated() -> bool:
    _init_session_state()
    return bool(st.session_state.get("auth_logged_in", False))


def current_user() -> Optional[str]:
    return st.session_state.get("auth_username")


def current_role() -> Optional[str]:
    return st.session_state.get("auth_role")


def has_role(required_role: str) -> bool:
    """Check if the current user has the required role or higher.
    Role hierarchy: admin > faculty
    """
    role = current_role()
    if role == "admin":
        return True
    if required_role == "faculty" and role == "faculty":
        return True
    return False


def login(username: str, password: str, db_path: Optional[Path] = None) -> bool:
    """Attempt login. Returns True on success."""
    _init_session_state()

    if is_locked_out():
        return False

    user = authenticate_user(username, password, db_path=db_path)
    if user:
        st.session_state.auth_logged_in = True
        st.session_state.auth_username = user["username"]
        st.session_state.auth_role = user["role"]
        st.session_state.auth_token = secrets.token_hex(SESSION_TOKEN_BYTES)
        st.session_state.auth_failed_attempts = 0
        st.session_state.auth_lockout_until = 0.0
        log_audit_event("LOGIN_SUCCESS", f"Role: {user['role']}", username=user["username"])
        return True
    else:
        st.session_state.auth_failed_attempts = (
            st.session_state.get("auth_failed_attempts", 0) + 1
        )
        if st.session_state.auth_failed_attempts >= MAX_LOGIN_ATTEMPTS:
            st.session_state.auth_lockout_until = time.time() + LOCKOUT_DURATION_SEC
            log_audit_event(
                "LOGIN_LOCKOUT",
                f"After {MAX_LOGIN_ATTEMPTS} failed attempts",
                username=username,
            )
        else:
            log_audit_event(
                "LOGIN_FAILURE",
                f"Attempt {st.session_state.auth_failed_attempts}/{MAX_LOGIN_ATTEMPTS}",
                username=username,
            )
        return False


def logout() -> None:
    """Clear authentication from session state."""
    username = current_user()
    _init_session_state()
    st.session_state.auth_logged_in = False
    st.session_state.auth_username = None
    st.session_state.auth_role = None
    st.session_state.auth_token = None
    log_audit_event("LOGOUT", username=username)


def require_login(required_role: str = "faculty", db_path: Optional[Path] = None) -> bool:
    """Render login UI if not authenticated. Returns True if user is authenticated with
    the required role. Call this at the top of any protected Streamlit page.

    Usage:
        if not auth.require_login(required_role='admin'):
            st.stop()
    """
    _init_session_state()

    # Check if first-time setup is needed
    if not has_any_admin(db_path=db_path):
        _render_first_time_setup(db_path=db_path)
        return False

    if is_authenticated() and has_role(required_role):
        return True

    _render_login_form(db_path=db_path)
    return False


def _render_first_time_setup(db_path: Optional[Path] = None) -> None:
    """Show initial admin setup wizard when no admin accounts exist."""
    st.warning("⚠️ **First-time setup:** No administrator accounts found.")
    st.info(
        "Create the first administrator account to secure the VoiceAttend system. "
        "This account will have full system access."
    )

    with st.form("first_time_setup_form"):
        st.subheader("🔐 Create Administrator Account")
        username = st.text_input("Username", placeholder="e.g. admin")
        password = st.text_input("Password (min 8 characters)", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.form_submit_button("Create Administrator Account", type="primary"):
            if not username.strip():
                st.error("Username cannot be empty.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    create_admin_user(username, password, role="admin", db_path=db_path)
                    st.success(
                        f"✅ Administrator account '{username}' created. Please log in."
                    )
                    log_audit_event(
                        "ADMIN_CREATED",
                        f"First admin setup: {username}",
                        username="system",
                    )
                    st.rerun()
                except ValueError as e:
                    st.error(f"Setup failed: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


def _render_login_form(db_path: Optional[Path] = None) -> None:
    """Render the Streamlit login UI."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 VoiceAttend Login")

        if is_locked_out():
            remaining = remaining_lockout_seconds()
            st.error(
                f"🔒 Account temporarily locked after {MAX_LOGIN_ATTEMPTS} failed attempts. "
                f"Please wait {remaining:.0f} more seconds."
            )
            return

        with st.form("login_form"):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                if login(username, password, db_path=db_path):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    attempts = st.session_state.get("auth_failed_attempts", 0)
                    remaining = MAX_LOGIN_ATTEMPTS - attempts
                    if remaining > 0:
                        st.error(
                            f"Invalid username or password. "
                            f"{remaining} attempt(s) remaining before lockout."
                        )
                    else:
                        st.error("Account locked. Please wait before trying again.")


def render_user_info_sidebar() -> None:
    """Render current user info and logout button in Streamlit sidebar."""
    if is_authenticated():
        st.sidebar.markdown(f"👤 **{current_user()}** ({current_role()})")
        if st.sidebar.button("🚪 Logout", key="sidebar_logout_btn"):
            logout()
            st.rerun()
