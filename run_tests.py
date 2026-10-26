"""VoiceAttend Automated Test Runner.

Executes all unit tests in tests/ using standard library or pytest,
reporting detailed pass/fail status and assertion verification.
"""

import sys
import tempfile
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    print("=" * 60)
    print(" VoiceAttend Test Suite Runner")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    # ---------------------------------------------------------
    # 1. Database Tests
    # ---------------------------------------------------------
    print("\n[Running Database Tests]")
    from tests import test_database

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # test_student_crud
        total_tests += 1
        try:
            db_file = tmp_path / "test_crud.db"
            test_database.db.init_db(db_path=db_file)
            test_database.test_student_crud(db_file)
            print("  [PASS] test_student_crud")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] test_student_crud ({e})")
            failed_tests.append(("test_student_crud", traceback.format_exc()))

        # test_duplicate_student_code_rejection
        total_tests += 1
        try:
            db_file = tmp_path / "test_dup.db"
            test_database.db.init_db(db_path=db_file)
            test_database.test_duplicate_student_code_rejection(db_file)
            print("  [PASS] test_duplicate_student_code_rejection")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] test_duplicate_student_code_rejection ({e})")
            failed_tests.append(("test_duplicate_student_code_rejection", traceback.format_exc()))

        # test_voice_samples_storage_and_cascade_delete
        total_tests += 1
        try:
            db_file = tmp_path / "test_cascade.db"
            test_database.db.init_db(db_path=db_file)
            test_database.test_voice_samples_storage_and_cascade_delete(db_file, tmp_path)
            print("  [PASS] test_voice_samples_storage_and_cascade_delete")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] test_voice_samples_storage_and_cascade_delete ({e})")
            failed_tests.append(("test_voice_samples_storage_and_cascade_delete", traceback.format_exc()))

        # test_attendance_sessions_and_duplicate_prevention
        total_tests += 1
        try:
            db_file = tmp_path / "test_dup_session.db"
            test_database.db.init_db(db_path=db_file)
            test_database.test_attendance_sessions_and_duplicate_prevention(db_file)
            print("  [PASS] test_attendance_sessions_and_duplicate_prevention")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] test_attendance_sessions_and_duplicate_prevention ({e})")
            failed_tests.append(("test_attendance_sessions_and_duplicate_prevention", traceback.format_exc()))

    # ---------------------------------------------------------
    # 2. Similarity & Speaker Identification Tests
    # ---------------------------------------------------------
    print("\n[Running Similarity & Speaker Identification Tests]")
    from tests import test_similarity

    total_tests += 1
    try:
        test_similarity.test_cosine_similarity_properties()
        print("  [PASS] test_cosine_similarity_properties")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_cosine_similarity_properties ({e})")
        failed_tests.append(("test_cosine_similarity_properties", traceback.format_exc()))

    total_tests += 1
    try:
        test_similarity.test_vector_normalization()
        print("  [PASS] test_vector_normalization")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_vector_normalization ({e})")
        failed_tests.append(("test_vector_normalization", traceback.format_exc()))

    total_tests += 1
    try:
        test_similarity.test_profile_averaging()
        print("  [PASS] test_profile_averaging")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_profile_averaging ({e})")
        failed_tests.append(("test_profile_averaging", traceback.format_exc()))

    total_tests += 1
    try:
        test_similarity.test_speaker_identification_acceptance_and_rejection()
        print("  [PASS] test_speaker_identification_acceptance_and_rejection")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_speaker_identification_acceptance_and_rejection ({e})")
        failed_tests.append(("test_speaker_identification_acceptance_and_rejection", traceback.format_exc()))

    # ---------------------------------------------------------
    # 3. Audio & Attendance Service Tests
    # ---------------------------------------------------------
    print("\n[Running Audio & Attendance Service Tests]")
    from tests import test_attendance

    total_tests += 1
    try:
        test_attendance.test_command_text_normalization()
        print("  [PASS] test_command_text_normalization")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_command_text_normalization ({e})")
        failed_tests.append(("test_command_text_normalization", traceback.format_exc()))

    total_tests += 1
    try:
        test_attendance.test_audio_validation_constraints()
        print("  [PASS] test_audio_validation_constraints")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_audio_validation_constraints ({e})")
        failed_tests.append(("test_audio_validation_constraints", traceback.format_exc()))

    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test_svc.db"
            test_attendance.db.init_db(db_path=db_file)
            test_attendance.test_live_attendance_validation_and_session_rejection(db_file)
        print("  [PASS] test_live_attendance_validation_and_session_rejection")
        passed_tests += 1
    except Exception as e:
        print(f"  [FAIL] test_live_attendance_validation_and_session_rejection ({e})")
        failed_tests.append(("test_live_attendance_validation_and_session_rejection", traceback.format_exc()))

    # ---------------------------------------------------------
    # 4. Security, Auth & Anti-Spoofing Tests
    # ---------------------------------------------------------
    print("\n[Running Security, Auth & Anti-Spoofing Tests]")
    from tests import test_security

    sec_tester = test_security.TestAntiSpoofing()
    for method_name in [
        "test_valid_speech_passes",
        "test_empty_audio_rejected",
        "test_silent_audio_rejected",
        "test_heavily_clipped_audio_suspicious",
        "test_result_structure",
        "test_model_status",
    ]:
        total_tests += 1
        try:
            getattr(sec_tester, method_name)()
            print(f"  [PASS] {method_name}")
            passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {method_name} ({e})")
            failed_tests.append((method_name, traceback.format_exc()))

    auth_tester = test_security.TestAuthentication()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_base = Path(tmp_dir)
        for method_name, takes_db in [
            ("test_password_hash_not_plaintext", False),
            ("test_password_verify_correct", False),
            ("test_password_verify_incorrect", False),
            ("test_short_password_rejected", True),
            ("test_invalid_role_rejected", True),
            ("test_create_and_authenticate_user", True),
            ("test_wrong_password_returns_none", True),
            ("test_duplicate_username_rejected", True),
            ("test_has_any_admin_false_initially", True),
            ("test_has_any_admin_true_after_creation", True),
            ("test_audit_log_written", True),
            ("test_deactivate_user", True),
        ]:
            total_tests += 1
            try:
                fn = getattr(auth_tester, method_name)
                if takes_db:
                    test_db = tmp_base / f"{method_name}.db"
                    fn(test_db)
                else:
                    fn()
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {method_name} ({e})")
                failed_tests.append((method_name, traceback.format_exc()))

    # Summary
    print("\n" + "=" * 60)
    print(f" Test Results: {passed_tests}/{total_tests} PASSED")
    print("=" * 60)

    if failed_tests:
        print("\nFailures:")
        for name, err in failed_tests:
            print(f"\n--- {name} ---")
            print(err)
        sys.exit(1)
    else:
        print("\nAll critical unit and integration tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
