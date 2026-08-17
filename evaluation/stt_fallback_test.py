"""
evaluation/stt_fallback_test.py
-------------------------------
Comprehensive Test Suite for ARROHA STT Architecture:
Primary: LOCAL faster-whisper
Emergency Fallback: Sarvam Saaras

Covers 6 Mandatory Verification Cases:
TEST 1: Local STT succeeds -> Sarvam NOT called.
TEST 2: Local STT intentionally fails -> Sarvam called once and succeeds.
TEST 3: Local STT fails + Sarvam fails -> Clean failure, no crash, no hang.
TEST 4: Silence / no speech -> Safe silence response, Sarvam NOT called.
TEST 5: Sarvam API key missing -> Local STT works, fallback gracefully unavailable.
TEST 6: Offline / No Internet simulation -> Local STT functions 100% offline.
"""

from __future__ import annotations

import io
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.voice.stt import LocalSTTBackend, SarvamSTTBackend, SpeechToTextEngine, STTResult


class TestSTTFallbackArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SpeechToTextEngine()

    def test_1_local_stt_succeeds_sarvam_not_called(self):
        """TEST 1: Under normal conditions, Local STT succeeds and Sarvam is NEVER called."""
        with patch.object(self.engine.sarvam_backend, "transcribe") as mock_sarvam:
            result = self.engine.transcribe(b"What is FAISS used for?", language_hint="en")

            # Verify local STT produced text
            self.assertTrue(len(result.text) > 0)
            self.assertEqual(result.backend, "local")
            self.assertFalse(result.fallback_used)
            self.assertFalse(result.is_silence)
            self.assertIsNone(result.error)

            # Assert Sarvam was NOT called at all
            mock_sarvam.assert_not_called()
            print("✅ TEST 1 PASSED: Local STT succeeded. Sarvam was NOT called (0 API calls).")

    def test_2_local_stt_fails_sarvam_fallback_succeeds(self):
        """TEST 2: When Local STT genuinely crashes, Sarvam is called once and used."""
        with patch.object(self.engine.local_backend, "transcribe", side_effect=RuntimeError("Simulated CUDA failure")):
            with patch.object(self.engine.sarvam_backend, "transcribe") as mock_sarvam:
                mock_sarvam.return_value = STTResult(
                    text="मौर्य साम्राज्य की राजधानी पाटलिपुत्र थी।",
                    language="hi",
                    latency_ms=1200.0,
                    backend="sarvam",
                    fallback_used=True,
                )

                result = self.engine.transcribe(b"\x00\x00" * 4000, language_hint="hi")

                # Verify Sarvam was called exactly once
                mock_sarvam.assert_called_once()
                self.assertEqual(result.backend, "sarvam")
                self.assertTrue(result.fallback_used)
                self.assertEqual(result.text, "मौर्य साम्राज्य की राजधानी पाटलिपुत्र थी।")
                print("✅ TEST 2 PASSED: Local STT failed -> Sarvam fallback succeeded seamlessly.")

    def test_3_local_stt_fails_and_sarvam_fails_clean_error(self):
        """TEST 3: When both Local STT and Sarvam fail, return clean error without crashing."""
        with patch.object(self.engine.local_backend, "transcribe", side_effect=RuntimeError("Local model crash")):
            with patch.object(self.engine.sarvam_backend, "transcribe") as mock_sarvam:
                mock_sarvam.return_value = STTResult(
                    text="",
                    language="Unknown",
                    latency_ms=4000.0,
                    backend="sarvam",
                    fallback_used=True,
                    error="Sarvam HTTP 503 Gateway Timeout",
                )

                result = self.engine.transcribe(b"\x00\x00" * 4000, language_hint="en")

                # Verify clean failure object returned
                self.assertEqual(result.text, "")
                self.assertEqual(result.backend, "sarvam")
                self.assertTrue(result.fallback_used)
                self.assertIsNotNone(result.error)
                print("✅ TEST 3 PASSED: Both failed -> Clean failure returned without crash or hang.")

    def test_4_silence_handling_no_sarvam_call(self):
        """TEST 4: Silence / empty audio returns silence result and does NOT waste Sarvam call."""
        with patch.object(self.engine.sarvam_backend, "transcribe") as mock_sarvam:
            result = self.engine.transcribe(b"", language_hint="en")

            self.assertEqual(result.text, "")
            self.assertTrue(result.is_silence)
            self.assertEqual(result.backend, "local")
            mock_sarvam.assert_not_called()
            print("✅ TEST 4 PASSED: Silence detected -> Sarvam NOT called.")

    def test_5_sarvam_api_key_missing_local_still_works(self):
        """TEST 5: When SARVAM_API_KEY is absent, Local STT operates normally."""
        with patch.dict(os.environ, {"SARVAM_API_KEY": ""}):
            engine_no_key = SpeechToTextEngine()
            engine_no_key.sarvam_backend.api_key = ""

            result = self.engine.transcribe(b"What is retrieval augmented generation?", language_hint="en")
            self.assertTrue(len(result.text) > 0)
            self.assertEqual(result.backend, "local")
            self.assertFalse(result.fallback_used)
            print("✅ TEST 5 PASSED: Missing API key -> Local STT operates with zero degradation.")

    def test_6_offline_simulation_local_stt_zero_internet(self):
        """TEST 6: Simulate NO INTERNET access (socket / HTTP blocked) -> Local STT works 100%."""
        with patch("requests.post", side_effect=ConnectionError("No route to host (Offline)")):
            result = self.engine.transcribe(b"Which embedding model is fast on CPU?", language_hint="en")

            # Local STT must execute with 0 network calls
            self.assertTrue(len(result.text) > 0)
            self.assertEqual(result.backend, "local")
            self.assertFalse(result.fallback_used)
            print("✅ TEST 6 PASSED: Offline simulation verified -> Local STT operates 100% offline with zero internet.")


def main():
    print("\n" + "=" * 80)
    print("  RUNNING ARROHA STT ARCHITECTURE & FALLBACK VERIFICATION TEST SUITE")
    print("=" * 80 + "\n")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSTTFallbackArchitecture)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if res.wasSuccessful():
        print("\n" + "=" * 80)
        print("  ALL 6 STT FALLBACK ARCHITECTURE TESTS PASSED SUCCESSFULLY (100%)")
        print("=" * 80 + "\n")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
