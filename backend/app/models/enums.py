"""
Enumerations for the application
"""

from enum import Enum


class QueryType(str, Enum):
    """Types of queries the system can handle"""
    SIMPLE_QA = "SIMPLE_QA"
    COMPARISON = "COMPARISON"
    SUMMARIZATION = "SUMMARIZATION"
    ANALYSIS = "ANALYSIS"
    EXTRACTION = "EXTRACTION"
    MULTI_HOP = "MULTI_HOP"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    MD = "md"


class LanguageCode(str, Enum):
    """Supported language codes"""
    EN = "en"  # English
    HI = "hi"  # Hindi
    BN = "bn"  # Bengali
    TE = "te"  # Telugu
    MR = "mr"  # Marathi
    TA = "ta"  # Tamil
    UR = "ur"  # Urdu
    GU = "gu"  # Gujarati
    KN = "kn"  # Kannada
    ML = "ml"  # Malayalam
    OR = "or"  # Odia
    PA = "pa"  # Punjabi
    AS = "as"  # Assamese
    MAI = "mai"  # Maithili
    SA = "sa"  # Sanskrit
    KS = "ks"  # Kashmiri
    NE = "ne"  # Nepali
    SD = "sd"  # Sindhi
    KOK = "kok"  # Konkani
    DOI = "doi"  # Dogri
    MNI = "mni"  # Manipuri
    SAT = "sat"  # Santali
    BO = "bo"  # Bodo
