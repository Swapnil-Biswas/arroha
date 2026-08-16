"""
evaluation/voice/language_router.py
-----------------------------------
Language routing configuration and voice registry for 15 Indian & global languages.
Explicitly categorizes every locale as NATIVE VOICE vs FALLBACK VOICE.
"""

from __future__ import annotations

from typing import Any, Optional

# Canonical 15-Language Voice Registry
LANGUAGE_VOICE_REGISTRY: dict[str, dict[str, Any]] = {
    "en": {
        "language_name": "English",
        "locale": "en-IN",
        "edge_voice": "en-IN-NeerjaNeural",
        "piper_voice": "en_US-lessac-medium",
        "onnx_model": "en_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Indian English neural voice",
    },
    "hi": {
        "language_name": "Hindi",
        "locale": "hi-IN",
        "edge_voice": "hi-IN-SwaraNeural",
        "piper_voice": "hi_IN-hindi-medium",
        "onnx_model": "hi_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Standard Hindi female neural voice",
    },
    "bn": {
        "language_name": "Bengali",
        "locale": "bn-IN",
        "edge_voice": "bn-IN-TanishaaNeural",
        "piper_voice": "bn_IN-bengali-medium",
        "onnx_model": "bn_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native West Bengal / Indian Bengali neural voice",
    },
    "ta": {
        "language_name": "Tamil",
        "locale": "ta-IN",
        "edge_voice": "ta-IN-PallaviNeural",
        "piper_voice": "ta_IN-tamil-medium",
        "onnx_model": "ta_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Tamil Nadu Dravidian neural voice",
    },
    "te": {
        "language_name": "Telugu",
        "locale": "te-IN",
        "edge_voice": "te-IN-ShrutiNeural",
        "piper_voice": "te_IN-telugu-medium",
        "onnx_model": "te_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Andhra / Telangana Telugu neural voice",
    },
    "mr": {
        "language_name": "Marathi",
        "locale": "mr-IN",
        "edge_voice": "mr-IN-AarohiNeural",
        "piper_voice": "mr_IN-marathi-medium",
        "onnx_model": "mr_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Maharashtra Marathi neural voice",
    },
    "gu": {
        "language_name": "Gujarati",
        "locale": "gu-IN",
        "edge_voice": "gu-IN-DhwaniNeural",
        "piper_voice": "gu_IN-gujarati-medium",
        "onnx_model": "gu_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Gujarat Gujarati neural voice",
    },
    "kn": {
        "language_name": "Kannada",
        "locale": "kn-IN",
        "edge_voice": "kn-IN-SapnaNeural",
        "piper_voice": "kn_IN-kannada-medium",
        "onnx_model": "kn_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Karnataka Kannada neural voice",
    },
    "ml": {
        "language_name": "Malayalam",
        "locale": "ml-IN",
        "edge_voice": "ml-IN-SobhanaNeural",
        "piper_voice": "ml_IN-malayalam-medium",
        "onnx_model": "ml_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Kerala Malayalam neural voice",
    },
    "pa": {
        "language_name": "Punjabi",
        "locale": "pa-IN",
        "edge_voice": "pa-IN-GurpreetNeural",
        "piper_voice": "pa_IN-punjabi-medium",
        "onnx_model": "pa_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Gurmukhi Punjabi neural voice",
    },
    "or": {
        "language_name": "Odia",
        "locale": "or-IN",
        "edge_voice": "hi-IN-MadhurNeural",
        "piper_voice": "hi_IN-hindi-medium",
        "onnx_model": "hi_IN_neural.onnx",
        "voice_type": "FALLBACK VOICE",
        "is_native": False,
        "sample_rate": 24000,
        "description": "Multilingual Indic phonetic fallback for Odia",
    },
    "as": {
        "language_name": "Assamese",
        "locale": "as-IN",
        "edge_voice": "bn-IN-TanishaaNeural",
        "piper_voice": "bn_IN-bengali-medium",
        "onnx_model": "bn_IN_neural.onnx",
        "voice_type": "FALLBACK VOICE",
        "is_native": False,
        "sample_rate": 24000,
        "description": "Eastern Nagari phonetic fallback for Assamese",
    },
    "ne": {
        "language_name": "Nepali",
        "locale": "ne-NP",
        "edge_voice": "ne-NP-HemkalaNeural",
        "piper_voice": "ne_NP-google-medium",
        "onnx_model": "ne_NP_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Devanagari Nepali neural voice",
    },
    "sa": {
        "language_name": "Sanskrit",
        "locale": "sa-IN",
        "edge_voice": "hi-IN-SwaraNeural",
        "piper_voice": "hi_IN-hindi-medium",
        "onnx_model": "hi_IN_neural.onnx",
        "voice_type": "FALLBACK VOICE",
        "is_native": False,
        "sample_rate": 24000,
        "description": "Devanagari classical Sanskrit phonetic fallback",
    },
    "ur": {
        "language_name": "Urdu",
        "locale": "ur-IN",
        "edge_voice": "ur-IN-GulNeural",
        "piper_voice": "ur_PK-urdu-medium",
        "onnx_model": "ur_IN_neural.onnx",
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Nastaliq Urdu neural voice",
    },
}

class LanguageRouter:
    """
    Resolves query language codes to appropriate TTS engine configurations and voices.
    """

    @staticmethod
    def get_voice_config(lang_code: str) -> dict[str, Any]:
        normalized = lang_code.strip().lower()
        if normalized in LANGUAGE_VOICE_REGISTRY:
            return LANGUAGE_VOICE_REGISTRY[normalized]
        # Default to English
        return LANGUAGE_VOICE_REGISTRY["en"]

    @staticmethod
    def is_native_voice(lang_code: str) -> bool:
        cfg = LanguageRouter.get_voice_config(lang_code)
        return bool(cfg.get("is_native", False))

    @staticmethod
    def list_supported_locales() -> list[dict[str, Any]]:
        return list(LANGUAGE_VOICE_REGISTRY.values())
