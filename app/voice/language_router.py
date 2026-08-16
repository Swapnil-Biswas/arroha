"""
app/voice/language_router.py
----------------------------
Production Language Router & Voice Registry for ARROHA.
Supports 15 Indian & global locales, explicitly categorizing Native Voices vs Fallback Voices.
"""

from __future__ import annotations

from typing import Any, Optional

LANGUAGE_VOICE_REGISTRY: dict[str, dict[str, Any]] = {
    "en": {
        "language_name": "English",
        "locale": "en-IN",
        "edge_voice": "en-IN-NeerjaNeural",
        "piper_voice": "en_US-lessac-medium",
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
        "voice_type": "NATIVE VOICE",
        "is_native": True,
        "sample_rate": 24000,
        "description": "Native Nastaliq Urdu neural voice",
    },
}

# Mapping aliases (3-letter ISO 639-2 / 639-3 to 2-letter codes)
LANG_ALIASES = {
    "eng": "en", "hin": "hi", "ben": "bn", "tam": "ta", "tel": "te",
    "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml", "pan": "pa",
    "ori": "or", "asm": "as", "nep": "ne", "san": "sa", "urd": "ur",
}


class LanguageRouter:
    """
    Resolves query language codes to appropriate TTS engine configurations and voices.
    """

    @staticmethod
    def normalize_code(lang_code: Optional[str]) -> str:
        if not lang_code:
            return "en"
        code = lang_code.strip().lower()
        return LANG_ALIASES.get(code, code)

    @staticmethod
    def get_voice_config(lang_code: Optional[str]) -> dict[str, Any]:
        norm = LanguageRouter.normalize_code(lang_code)
        if norm in LANGUAGE_VOICE_REGISTRY:
            return LANGUAGE_VOICE_REGISTRY[norm]
        return LANGUAGE_VOICE_REGISTRY["en"]

    @staticmethod
    def is_native_voice(lang_code: Optional[str]) -> bool:
        cfg = LanguageRouter.get_voice_config(lang_code)
        return bool(cfg.get("is_native", False))

    @staticmethod
    def list_supported_locales() -> list[dict[str, Any]]:
        return list(LANGUAGE_VOICE_REGISTRY.values())
