"""
ingestion/download.py
---------------------
Multilingual dataset acquisition, streaming, and sample generation.

Features:
  - Streams or downloads specific language shards (e.g. Hindi, Bengali, Tamil, etc.)
  - Parses records via PyArrow without high memory footprint
  - Generates realistic multilingual evaluation/benchmark sets for offline development
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Generator, Iterable, Optional

import pyarrow.parquet as pq

from app.config import RAW_DATA_DIR, SUPPORTED_LANGUAGES
from ingestion.models import DatasetRecord

logger = logging.getLogger(__name__)

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main"

# Language code mapping to shard filenames in ai4bharat/MSMARCO-XI
LANG_SHARD_MAP = {
    "asm": "asmtrain.parquet",
    "ben": "bentrain.parquet",
    "guj": "gujtrain.parquet",
    "hin": "hintrain.parquet",
    "kan": "kantrain.parquet",
    "mal": "maltrain.parquet",
    "mar": "martrain.parquet",
    "nep": "neptrain.parquet",
    "ori": "oritrain.parquet",
    "pan": "pantrain.parquet",
    "san": "santrain.parquet",
    "tam": "tamtrain.parquet",
    "tel": "telval.parquet",   # Telugu has validation split
    "urd": "urdtrain.parquet",
}


def download_language_shard(
    lang: str,
    split: str = "train",
    dest_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
) -> Path:
    """
    Download a single language Parquet shard from Hugging Face if not already cached.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = LANG_SHARD_MAP.get(lang, f"{lang}{split}.parquet")
    dest_file = dest_dir / filename

    if dest_file.exists() and not force_download:
        logger.info("Using cached shard: %s", dest_file)
        return dest_file

    url = f"{HF_DATASET_BASE_URL}/{split}/{filename}"
    logger.info("Downloading shard for language '%s' from %s...", lang, url)

    req = urllib.request.Request(url, headers={"User-Agent": "hhgoa-rag-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response, open(dest_file, "wb") as out_file:
            # Stream download in 1MB chunks
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)
        logger.info("Successfully downloaded: %s (%d bytes)", dest_file, dest_file.stat().st_size)
    except Exception as exc:
        if dest_file.exists():
            dest_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return dest_file


def stream_records_from_parquet(
    parquet_path: Path | str,
    max_records: Optional[int] = None,
    batch_size: int = 256,
) -> Generator[DatasetRecord, None, None]:
    """
    Stream DatasetRecords from a local Parquet file in memory-efficient row batches.
    """
    pf = pq.ParquetFile(str(parquet_path))
    yielded = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        pydict = batch.to_pydict()
        num_rows = len(next(iter(pydict.values())))

        for row_idx in range(num_rows):
            row = {col: pydict[col][row_idx] for col in pydict}

            record = DatasetRecord(
                query_id=row.get("query_id", f"gen_{yielded}"),
                query_type=row.get("query_type"),
                query=str(row.get("query", "")),
                eng_query=str(row.get("Eng_Query", "")),
                answer=str(row.get("Answer", "")),
                eng_answer=str(row.get("Eng_Answer", "")),
                source_lang=str(row.get("source_lang", "en")),
                target_lang=str(row.get("target_lang", "hi")),
                passages=row.get("passages", {}),
                meta=row.get("meta", {}),
            )
            yield record
            yielded += 1

            if max_records and yielded >= max_records:
                return


def create_sample_multilingual_corpus() -> list[DatasetRecord]:
    """
    Generates a curated, high-quality multilingual sample dataset across Indic languages
    for offline testing, benchmarking, and immediate indexing.
    """
    samples = [
        # Hindi Sample
        DatasetRecord(
            query_id=101,
            query="भारत की राजधानी क्या है और इसका इतिहास क्या है?",
            eng_query="What is the capital of India and its history?",
            answer="नई दिल्ली भारत की आधिकारिक राजधानी है।",
            eng_answer="New Delhi is the official capital of India.",
            source_lang="en",
            target_lang="hi",
            passages={
                "English_passages": [
                    "New Delhi is the capital of India and part of the National Capital Territory of Delhi. The foundation stone of the city was laid in 1911 by Emperor George V during the Delhi Durbar.",
                    "Mumbai is the financial capital of India, known for Bollywood and trade, located in the state of Maharashtra along the Arabian Sea coast.",
                    "The Rashtrapati Bhavan, located on Rajpath in New Delhi, is the official residence of the President of India.",
                ],
                "Translated_passages": [
                    "नई दिल्ली भारत की राजधानी है और दिल्ली के राष्ट्रीय राजधानी क्षेत्र का हिस्सा है। इस शहर की आधारशिला 1911 में दिल्ली दरबार के दौरान सम्राट जॉर्ज पंचम द्वारा रखी गई थी।",
                    "मुंबई भारत की वित्तीय राजधानी है, जो बॉलीवुड और व्यापार के लिए जानी जाती है, जो अरब सागर के तट पर महाराष्ट्र राज्य में स्थित है।",
                    "नई दिल्ली में राजपथ पर स्थित राष्ट्रपति भवन, भारत के राष्ट्रपति का आधिकारिक निवास है।",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # Bengali Sample
        DatasetRecord(
            query_id=102,
            query="রবীন্দ্রনাথ ঠাকুর কে ছিলেন এবং তিনি কোন পুরস্কার পেয়েছিলেন?",
            eng_query="Who was Rabindranath Tagore and which award did he receive?",
            answer="রবীন্দ্রনাথ ঠাকুর ছিলেন একজন বিশিষ্ট কবি যিনি সাহিত্যে নোবেল পুরস্কার লাভ করেন।",
            eng_answer="Rabindranath Tagore was a prominent poet who won the Nobel Prize in Literature.",
            source_lang="en",
            target_lang="bn",
            passages={
                "English_passages": [
                    "Rabindranath Tagore was a Bengali polymath who reshaped Bengali literature and music. In 1913, he became the first non-European to win the Nobel Prize in Literature for Gitanjali.",
                    "Satyajit Ray was an Indian filmmaker, screenwriter, and author from Kolkata, known for the Apu Trilogy.",
                    "Kolkata is the capital of the Indian state of West Bengal, situated on the east bank of the Hooghly River.",
                ],
                "Translated_passages": [
                    "রবীন্দ্রনাথ ঠাকুর ছিলেন একজন বাঙালি বহুবিদ্যাবিশারদ যিনি বাংলা সাহিত্য ও সঙ্গীতকে নতুন রূপ দিয়েছিলেন। ১৯১৩ সালে গীতাঞ্জলির জন্য তিনি সাহিত্যে নোবেল পুরস্কার পাওয়া প্রথম অ-ইউরোপীয় ব্যক্তি হন।",
                    "সত্যজিৎ রায় ছিলেন কলকাতার একজন ভারতীয় চলচ্চিত্র নির্মাতা, চিত্রনাট্যকার এবং লেখক, যিনি অপু ট্রিলজির জন্য পরিচিত।",
                    "কলকাতা হলো ভারতের পশ্চিমবঙ্গ রাজ্যের রাজধানী, যা হুগলি নদীর পূর্ব তীরে অবস্থিত।",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # Tamil Sample
        DatasetRecord(
            query_id=103,
            query="தமிழ்நாட்டின் தலைநகரம் எது மற்றும் அதன் சிறப்பு என்ன?",
            eng_query="What is the capital of Tamil Nadu and its significance?",
            answer="சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும்.",
            eng_answer="Chennai is the capital of Tamil Nadu.",
            source_lang="en",
            target_lang="ta",
            passages={
                "English_passages": [
                    "Chennai, formerly known as Madras, is the capital of Tamil Nadu. Located on the Coromandel Coast of the Bay of Bengal, it is a major cultural, economic and educational center.",
                    "Madurai is an energetic, ancient city on the Vaigai River in Tamil Nadu, known for the historic Meenakshi Amman Temple.",
                    "Thanjavur is an important center of South Indian religion, art, and architecture, famous for the Brihadeeswarar Temple.",
                ],
                "Translated_passages": [
                    "சென்னை, முன்னர் மெட்ராஸ் என்று அழைக்கப்பட்டது, தமிழ்நாட்டின் தலைநகரமாகும். வங்காள விரிகுடாவின் கோரமண்டல் கடற்கரையில் அமைந்துள்ள இது ஒரு முக்கிய கலாச்சார, பொருளாதார மற்றும் கல்வி மையமாகும்.",
                    "மதுரை தமிழ்நாட்டில் வைகை ஆற்றின் கரையில் அமைந்துள்ள ஒரு ஆற்றல்மிக்க, பழமையான நகரமாகும், இது வரலாற்று சிறப்புமிக்க மீனாட்சி அம்மன் கோவிலுக்கு பெயர் பெற்றது.",
                    "தஞ்சாவூர் தென்னிந்திய மதம், கலை மற்றும் கட்டிடக்கலையின் ஒரு முக்கிய மையமாகும், இது பிரகதீஸ்வரர் கோவிலுக்கு பிரபலமானது.",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # Marathi Sample
        DatasetRecord(
            query_id=104,
            query="महाराष्ट्राची राजधानी कोणती आहे?",
            eng_query="What is the capital of Maharashtra?",
            answer="मुंबई ही महाराष्ट्राची राजधानी आहे.",
            eng_answer="Mumbai is the capital of Maharashtra.",
            source_lang="en",
            target_lang="mr",
            passages={
                "English_passages": [
                    "Mumbai is the capital city of the Indian state of Maharashtra. It is the financial, commercial, and entertainment capital of India.",
                    "Pune is considered the cultural capital of Maharashtra and a major IT and educational hub.",
                    "Nagpur is the winter capital and the third largest city of the Indian state of Maharashtra.",
                ],
                "Translated_passages": [
                    "मुंबई ही महाराष्ट्र राज्याची राजधानी आहे. ही भारताची आर्थिक, व्यावसायिक आणि मनोरंजनाची राजधानी आहे.",
                    "पुणे हे महाराष्ट्राची सांस्कृतिक राजधानी आणि प्रमुख आयटी आणि शैक्षणिक केंद्र मानले जाते.",
                    "नागपूर ही हिवाळी राजधानी आणि महाराष्ट्र राज्यातील तिसरे सर्वात मोठे शहर आहे.",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # Karnataka Sample
        DatasetRecord(
            query_id=108,
            query="कर्नाटक की राजधानी क्या है?",
            eng_query="What is the capital of Karnataka?",
            answer="बेंगलुरु कर्नाटक की राजधानी है।",
            eng_answer="Bengaluru is the capital of Karnataka.",
            source_lang="en",
            target_lang="kan",
            passages={
                "English_passages": [
                    "Bengaluru (also known as Bangalore) is the capital and largest city of the Indian state of Karnataka. It is widely known as the Silicon Valley of India.",
                    "Mysuru is the second-largest city in the state of Karnataka, famous for the grand Mysuru Palace.",
                ],
                "Translated_passages": [
                    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ এবং ದೊಡ್ಡ ನಗರವಾಗಿದೆ. ಇದು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಪ್ರಸಿದ್ಧವಾಗಿದೆ.",
                    "ಮೈಸೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಎರಡನೇ ದೊಡ್ಡ ನಗರವಾಗಿದೆ, ಇದು ಭವ್ಯವಾದ ಮೈಸೂರು ಅರಮನೆಗೆ ಹೆಸರುವಾಸಿಯಾಗಿದೆ.",
                ],
                "is_selected": [1, 0],
            },
        ),
        # Telugu Sample
        DatasetRecord(
            query_id=105,
            query="హైదరాబాద్ నగరం యొక్క ప్రాముఖ్యత ఏమిటి?",
            eng_query="What is the significance of Hyderabad city?",
            answer="హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని మరియు ప్రధాన ఐటీ కేంద్రం.",
            eng_answer="Hyderabad is the capital of Telangana and a major IT hub.",
            source_lang="en",
            target_lang="te",
            passages={
                "English_passages": [
                    "Hyderabad is the capital and largest city of the Indian state of Telangana. Known as the City of Pearls, it is a prominent technological and pharmaceutical industry center.",
                    "Visakhapatnam is a major port city and commercial hub on the coast of Andhra Pradesh.",
                    "Charminar is a historic monument and mosque located in Hyderabad, constructed in 1591.",
                ],
                "Translated_passages": [
                    "హైదరాబాద్ భారతదేశంలోని తెలంగాణ రాష్ట్ర రాజధాని మరియు అతిపెద్ద నగరం. ముత్యాల నగరంగా పిలువబడే ఇది ప్రముఖ సాంకేతిక మరియు ఔషధ పరిశ్రమ కేంద్రం.",
                    "విశాఖపట్నం ఆంధ్రప్రదేశ్ తీరంలో ఉన్న ఒక ప్రధాన ఓడరేవు నగరం మరియు వాణిజ్య కేంద్రం.",
                    "చార్మినార్ 1591లో నిర్మించబడిన హైదరాబాద్‌లో ఉన్న ఒక చారిత్రక కట్టడం మరియు మసీదు.",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # Gujarati Sample
        DatasetRecord(
            query_id=106,
            query="ગુજરાતનું સૌથી મોટું શહેર કયું છે?",
            eng_query="What is the largest city in Gujarat?",
            answer="અમદાવાદ ગુજરાતનું સૌથી મોટું શહેર છે.",
            eng_answer="Ahmedabad is the largest city in Gujarat.",
            source_lang="en",
            target_lang="gu",
            passages={
                "English_passages": [
                    "Ahmedabad is the most populous city in the Indian state of Gujarat and is situated on the banks of the Sabarmati River. It was designated as India's first UNESCO World Heritage City.",
                    "Gandhinagar is the capital city of Gujarat, named after Mahatma Gandhi.",
                    "Surat is known as the Diamond City of India, famous for diamond cutting and textile industries.",
                ],
                "Translated_passages": [
                    "અમદાવાદ ભારતના ગુજરાત રાજ્યનું સૌથી વધુ વસ્તી ધરાવતું શહેર છે અને તે સાબરમતી નદીના કિનારે આવેલું છે. તેને ભારતના પ્રથમ યુનેસ્કો વર્લ્ડ હેરિટેજ સિટી તરીકે નિયુક્ત કરવામાં આવ્યું હતું.",
                    "ગાંધીનગર ગુજરાતની રાજધાની છે, જેનું નામ મહાત્મા ગાંધીના નામ પરથી રાખવામાં આવ્યું છે.",
                    "સુરત ભારતના ડાયમંડ સિટી તરીકે ઓળખાય છે, જે હીરા કટીંગ અને કાપડ ઉદ્યોગો માટે પ્રખ્યાત છે.",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
        # English General Knowledge Sample
        DatasetRecord(
            query_id=107,
            query="How does Retrieval-Augmented Generation work in voice pipelines?",
            eng_query="How does Retrieval-Augmented Generation work in voice pipelines?",
            answer="Voice RAG converts speech to text, retrieves relevant context from a vector database, and generates grounded answers.",
            eng_answer="Voice RAG converts speech to text, retrieves relevant context from a vector database, and generates grounded answers.",
            source_lang="en",
            target_lang="en",
            passages={
                "English_passages": [
                    "Voice-enabled Retrieval-Augmented Generation (Voice RAG) combines speech recognition with hybrid search over knowledge bases to feed relevant context into large language models for accurate, low-latency spoken responses.",
                    "Automatic Speech Recognition (ASR) converts audio signals into text representations using deep neural networks.",
                    "Vector retrieval uses high-dimensional embeddings to perform semantic similarity matching against precomputed document representations.",
                ],
                "Translated_passages": [
                    "Voice-enabled Retrieval-Augmented Generation (Voice RAG) combines speech recognition with hybrid search over knowledge bases to feed relevant context into large language models for accurate, low-latency spoken responses.",
                    "Automatic Speech Recognition (ASR) converts audio signals into text representations using deep neural networks.",
                    "Vector retrieval uses high-dimensional embeddings to perform semantic similarity matching against precomputed document representations.",
                ],
                "is_selected": [1, 0, 0],
            },
        ),
    ]
    return samples
