"""
app/guardrails/grounding.py
---------------------------
Grounding and hallucination detection.
Verifies that generated answers are strictly supported by retrieved context snippets.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from app.config import GROUNDING_SIMILARITY_THRESHOLD, MIN_RETRIEVAL_SCORE
from app.schemas.response import GroundingResult, SourceDocument

# Standardized localized refusal statements across all supported Indian languages
LOCALIZED_REFUSALS: dict[str, str] = {
    "en": "I do not have enough information in the retrieved sources to answer this question.",
    "hi": "प्रदत्त संदर्भ में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
    "bn": "প্রদত্ত সূত্রে এই প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।",
    "ta": "வழங்கப்பட்ட தரவுகளில் இந்தக் கேள்விக்கு பதிலளிக்க போதுமான தகவல் இல்லை.",
    "te": "అందించిన ఆధారాలలో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగినంత సమాచారం లేదు.",
    "mr": "दिलेल्या संदर्भात या प्रश्नाचे उत्तर देण्यासाठी पुरेशी माहिती उपलब्ध नाही.",
    "gu": "આપેલ સંદર્ભમાં આ પ્રશ્નનો જવાબ આપવા માટે પૂરતી માહિતી નથી.",
    "kn": "ಒದಗಿಸಲಾದ ಮಾಹಿತಿಯಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ವಿವರಗಳಿಲ್ಲ.",
    "ml": "നൽകിയിട്ടുള്ള വിവരങ്ങളിൽ ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ ലഭ്യമല്ല.",
    "pa": "ਦਿੱਤੇ ਗਏ ਸਰੋਤਾਂ ਵਿੱਚ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਲਈ ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
    "or": "ପ୍ରଦତ୍ତ ତଥ୍ୟରେ ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବା ପାଇଁ ଯଥେଷ୍ଟ ସୂଚନା ନାହିଁ।",
    "as": "প্ৰদত্ত তথ্যত এই প্ৰশ্নৰ উত্তৰ দিবলৈ পৰ্যাপ্ত সমল নাই।",
    "ne": "प्रदत्त सन्दर्भमा यस प्रश्नको उत्तर दिन पर्याप्त जानकारी छैन।",
    "sa": "प्रदत्तसन्दर्भे अस्य प्रश्नस्य उत्तरं दातुं पर्याप्तसूचना नास्ति।",
    "ur": "فراہم کردہ معلومات میں اس سوال کا جواب دینے کے لیے کافی مواد موجود نہیں ہے۔",
}

REFUSAL_PHRASES = [
    "[insufficient_context]",
    "insufficient_context",
    "do not have enough information",
    "does not contain enough information",
    "not mentioned in the context",
    "not provided in the context",
    "पर्याप्त जानकारी नहीं है", # Hindi
    "পর্যাপ্ত তথ্য নেই",         # Bengali
    "போதுமான தகவல் இல்லை",       # Tamil
    "సరిపోవు సమాచారం లేదు",     # Telugu
    "पुरेशी माहिती नाही",        # Marathi
    "પૂરતી માહિતી નથી",          # Gujarati
    "ಸಾಕಷ್ಟು ವಿವರಗಳಿಲ್ಲ",       # Kannada
    "വിവരങ്ങൾ ലഭ്യമല്ല",         # Malayalam
    "ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ",       # Punjabi
    "যଥେଷ୍ଟ ସୂଚନା ନାହିଁ",         # Odia
    "পৰ্যাপ্ত সমল নাই",           # Assamese
    "पर्याप्त जानकारी छैन",       # Nepali
    "पर्याप्तसूचना नास्ति",        # Sanskrit
    "کافی مواد موجود نہیں",     # Urdu
    "cannot answer",
    "insufficient evidence",
    "no relevant context",
]

PROMPT_LEAK_PATTERNS = [
    re.compile(r"critical\s+rules\s*:", re.IGNORECASE),
    re.compile(r"the\s+critical\s+rules\s+are", re.IGNORECASE),
    re.compile(r"you\s+are\s+a\s+factual", re.IGNORECASE),
    re.compile(r"retrieved\s+context\s*:", re.IGNORECASE),
    re.compile(r"user\s+question\s*:", re.IGNORECASE),
    re.compile(r"language\s+consistency\s*:", re.IGNORECASE),
]


QUESTION_STOPWORDS = {
    # English
    "what", "is", "the", "capital", "of", "how", "does", "work", "in", "and", "why", "who", "when", "where", "which",
    "are", "was", "were", "been", "being", "have", "has", "had", "for", "with", "about", "city", "state", "most", "largest",
    "tell", "me", "give", "explain", "describe", "importance", "history", "details", "average", "temperature", "population",
    "country", "area", "big", "small", "year", "born", "died", "minister", "president", "king", "queen", "prime", "weather",
    "world", "cup", "game", "first", "second", "third", "main", "major", "located", "found", "situated", "known", "called",
    
    # Hindi / Devanagari
    "क्या", "है", "हैं", "की", "का", "के", "और", "इसका", "इसकी", "इसके", "इस", "में", "कहाँ", "कौन", "कैसे", "था", "थी", "थे",
    "पर", "से", "को", "राजधानी", "शहर", "राज्य", "देश", "सबसे", "बड़ा", "बड़ी", "बड़े", "छोटा", "प्रमुख", "इतिहास", "महत्व",
    "वर्ष", "साल", "तापमान", "औसत", "जनसंख्या", "मौसम", "बारे", "बताओ", "दीजिए", "करें", "प्रसिद्ध", "कहा", "जाता",
    
    # Bengali
    "কী", "কি", "কে", "ছিলেন", "এবং", "তিনি", "কোন", "কবে", "কোথায়", "কেন", "হল", "হলো", "হয়", "এর", "তে", "এ", "বা",
    "রাজধানী", "শহর", "রাজ্য", "দেশ", "সবচেয়ে", "বড়", "প্রধান", "ইতিহাস", "গুরুত্ব", "বছর", "তাপমাত্রা", "গড়", "জনসংখ্যা", "বিখ্যাত",
    
    # Tamil
    "எது", "எங்கு", "யார்", "மற்றும்", "அதன்", "என்ன", "எப்படி", "ஒரு", "ஆகும்", "இல்", "உள்ள", "என்பது", "சிறப்பு",
    "தலைநகரம்", "நகரம்", "மாநிலம்", "நாடு", "மிகப்பெரிய", "பெரிய", "முக்கிய", "வரலாறு", "முக்கியத்துவம்", "ஆண்டு", "பிரபலமானது",
    
    # Telugu
    "ఏమిటి", "ఏది", "ఎక్కడ", "ఎవరు", "మరియు", "యొక్క", "ఎప్పుడు", "ఎలా", "ఉంది", "ఉన్న", "అనేది", "ప్రాముఖ్యత",
    "రాజధాని", "నగరం", "రాష్ట్రం", "దేశం", "అతిపెద్ద", "పెద్ద", "ప్రముఖ", "చరిత్ర", "సంవత్సరం", "ఉష్ణోగ్రత", "సగటు", "ప్రసిద్ధి",
    
    # Marathi
    "काय", "आहे", "आहेत", "कोणती", "कोणते", "कोणता", "आणि", "याचा", "याची", "याचे", "कशासाठी", "प्रसिद्ध", "कधी",
    "राजधानी", "शहर", "राज्य", "देश", "सर्वात", "मोठे", "मोठा", "मोठी", "प्रमुख", "इतिहास", "महत्त्व", "वर्ष", "तापमान", "सांगा",
    
    # Gujarati
    "શું", "છે", "કયું", "કયો", "કયા", "અને", "સૌથી", "મોટું", "મોટો", "મોટી", "ક્યાં", "કેવી", "રીતે", "વિશે",
    "રાજધાની", "શહેર", "રાજ્ય", "દેશ", "મુખ્ય", "ઇતિહાસ", "મહત્વ", "વર્ષ", "તાપમાન", "સરેરાશ", "વસ્તી", "પ્રખ્યાત",
}


class GroundingChecker:
    """
    Evaluates factual grounding, query-context relevance, hallucination risk, and prompt leakage.
    Strictly verifies that answers are fully supported by the retrieved dataset passages.
    """

    def __init__(
        self,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
        similarity_threshold: float = GROUNDING_SIMILARITY_THRESHOLD,
    ) -> None:
        self.min_retrieval_score = min_retrieval_score
        self.similarity_threshold = similarity_threshold

    def check_query_context_alignment(
        self,
        query: str,
        sources: list[SourceDocument],
    ) -> tuple[bool, float, str]:
        """
        Verify that key subject entities and content tokens from the query actually exist in the retrieved context.
        Prevents answering spurious passages when queried about non-existent topics (e.g. Mars, Moon, Atlantis).
        """
        if not sources:
            return False, 0.0, "No source documents provided."

        query_lower = query.lower()
        combined_context = " ".join([s.text for s in sources]).lower()

        LANG_TAGS = {"en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ne", "sa", "ur"}
        context_words = set(re.findall(r"[\w\u0900-\u0D7F]+", combined_context))
        context_stems = {w[:4] for w in context_words if len(w) >= 4}

        # Extract substantive tokens (alphanumeric/Indic, length >= 3)
        raw_tokens = re.findall(r"[\w\u0900-\u0D7F]+", query_lower)
        content_tokens = [t for t in raw_tokens if len(t) >= 3 and t not in QUESTION_STOPWORDS and t not in LANG_TAGS]

        if not content_tokens:
            content_tokens = [t for t in raw_tokens if len(t) >= 3 and t not in {"what", "who", "how", "why", "where", "when", "क्या", "কী", "எது", "ఏది", "काय", "શું"}]

        if not content_tokens:
            return True, 1.0, "No content tokens to align."

        # Check exact token in context or stem prefix match for agglutinative Indic words
        matched = []
        for t in content_tokens:
            stem = t[:4] if len(t) >= 4 else t
            if t in context_words or stem in context_stems or (len(t) >= 4 and t in combined_context):
                matched.append(t)

        alignment_score = len(matched) / len(content_tokens)
        max_dense = max((getattr(s, "dense_score", s.score) or 0.0 for s in sources), default=0.0)
        max_score = max((s.score for s in sources), default=0.0)

        # Cross-lingual & Multilingual alignment:
        # 1. Direct lexical/stem match: at least 1 substantive entity (alignment_score >= 0.20) and max_dense >= 0.25
        # 2. Semantic dense match: max_dense >= 0.40 AND max_score >= 0.30
        if alignment_score >= 0.20 and max_dense >= 0.25:
            is_aligned = True
        elif max_dense >= 0.40 and max_score >= 0.30:
            is_aligned = True
        else:
            is_aligned = False

        reason = (
            f"Query subject entities {matched}/{content_tokens} matched context (score {alignment_score:.2f})."
            if is_aligned
            else f"Query subject entities {content_tokens} missing from retrieved context (alignment {alignment_score:.2f}, dense {max_dense:.2f})."
        )
        return is_aligned, alignment_score, reason

    def check(
        self,
        query: str,
        sources: list[SourceDocument],
        generated_answer: str,
    ) -> tuple[GroundingResult, float]:
        """
        Verify grounding of generated answer against sources.
        Returns (GroundingResult, latency_ms).
        """
        t0 = time.perf_counter_ns()

        if not generated_answer or not generated_answer.strip():
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return (
                GroundingResult(
                    is_grounded=False,
                    grounding_score=0.0,
                    refusal_triggered=True,
                    refusal_reason="Generated answer is empty.",
                ),
                latency_ms,
            )

        answer_lower = generated_answer.lower().strip()

        # 1. Check for prompt leakage
        for pattern in PROMPT_LEAK_PATTERNS:
            if pattern.search(answer_lower):
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return (
                    GroundingResult(
                        is_grounded=False,
                        grounding_score=0.0,
                        refusal_triggered=True,
                        refusal_reason="Model leaked prompt instructions.",
                    ),
                    latency_ms,
                )

        # 2. Check for explicit refusal in generated text
        for phrase in REFUSAL_PHRASES:
            if phrase.lower() in answer_lower:
                latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return (
                    GroundingResult(
                        is_grounded=True,
                        grounding_score=1.0,
                        refusal_triggered=True,
                        refusal_reason="Model recognized missing dataset information and triggered refusal.",
                    ),
                    latency_ms,
                )

        # 3. Check Query-Context Subject Entity Alignment
        is_aligned, align_score, align_reason = self.check_query_context_alignment(query, sources)
        if not is_aligned:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return (
                GroundingResult(
                    is_grounded=False,
                    grounding_score=round(align_score, 4),
                    refusal_triggered=True,
                    refusal_reason=align_reason,
                ),
                latency_ms,
            )

        # 4. Check if retrieved context was too weak / missing / below threshold
        max_score = max((s.score for s in sources), default=0.0)
        max_dense = max((getattr(s, "dense_score", s.score) or 0.0 for s in sources), default=0.0)

        if not sources or max_score < self.min_retrieval_score or max_dense < 0.38:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return (
                GroundingResult(
                    is_grounded=False,
                    grounding_score=round(max_score, 4),
                    refusal_triggered=True,
                    refusal_reason=f"Retrieved context relevance ({max_score:.2f}) is below minimum threshold ({self.min_retrieval_score:.2f}).",
                ),
                latency_ms,
            )

        # 5. Content Word and Named Entity Overlap Check
        combined_context = " ".join([s.text for s in sources]).lower()
        answer_words = [w for w in re.findall(r"[\w\u0900-\u0D7F]+", answer_lower) if len(w) > 2]

        if not answer_words:
            overlap_ratio = 1.0
        else:
            matched_words = [w for w in answer_words if w in combined_context]
            overlap_ratio = len(matched_words) / len(answer_words)

        # 6. Strict Number and Year Verification
        # If numbers/years appear in the answer, they MUST appear in the context
        answer_numbers = set(re.findall(r"\b\d{1,6}\b", answer_lower))
        context_numbers = set(re.findall(r"\b\d{1,6}\b", combined_context))
        hallucinated_numbers = answer_numbers - context_numbers
        if hallucinated_numbers:
            latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return (
                GroundingResult(
                    is_grounded=False,
                    grounding_score=0.2,
                    refusal_triggered=True,
                    refusal_reason=f"Hallucinated numeric facts detected: {hallucinated_numbers}",
                ),
                latency_ms,
            )

        is_grounded = overlap_ratio >= self.similarity_threshold

        result = GroundingResult(
            is_grounded=is_grounded,
            grounding_score=round(overlap_ratio, 4),
            refusal_triggered=not is_grounded,
            refusal_reason=None if is_grounded else f"Lexical grounding overlap ({overlap_ratio:.2f}) below threshold ({self.similarity_threshold}).",
        )

        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return result, latency_ms
