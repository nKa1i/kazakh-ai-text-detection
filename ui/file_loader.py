"""Defensive document file loader for Kazakh AI-Text Detection UI.

Supports .txt, .docx, and .pdf formats with:
- 10 MB strict file size ceiling (rejects before reading into memory)
- 25,000 word capping (prevents memory bombs)
- Encoding fallbacks for .txt (UTF-8, UTF-8-SIG, CP1251, Latin-1)
- Graceful fallbacks when optional dependencies (docx, pypdf) are unavailable
"""

import os
import re
from typing import Any, Optional, Tuple

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_WORD_LIMIT = 25000
SIZE_EXCEEDED_MESSAGE = "Файл тым үлкен (максимум 10 MB). / File exceeds 10 MB limit."
TRUNCATION_WARNING = "[Мәтін 25 000 сөз шегінен асты және алғашқы бөлігі ғана өңделді / Document truncated to 25,000 words limit]"


def _resolve_file_path(file_obj_or_path: Any) -> Optional[str]:
    """Extracts a valid file path string from a path or file-like wrapper."""
    if file_obj_or_path is None:
        return None

    if isinstance(file_obj_or_path, (str, os.PathLike)):
        path_str = str(file_obj_or_path)
        return path_str if path_str.strip() else None

    # Handle Gradio File component or NamedTemporaryFile wrappers
    if hasattr(file_obj_or_path, "name") and isinstance(file_obj_or_path.name, str):
        path_str = file_obj_or_path.name
        return path_str if path_str.strip() else None

    # Handle dictionary representation
    if isinstance(file_obj_or_path, dict):
        path_str = file_obj_or_path.get("path") or file_obj_or_path.get("name")
        if path_str and isinstance(path_str, str) and path_str.strip():
            return path_str

    return None


def _load_txt(file_path: str) -> Tuple[str, Optional[str]]:
    """Loads plain text file attempting UTF-8, CP1251, and Latin-1 encodings."""
    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        return "", f"Файлды оқу қатесі: {e} / File read error: {e}"

    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            decoded = raw_bytes.decode(enc)
            # Normalize CRLF and CR to LF for cross-platform consistency
            normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
            return normalized, None
        except UnicodeDecodeError:
            continue

    return "", "Мәтінді кодтау қатесі / Text encoding could not be decoded."


def _load_docx(file_path: str) -> Tuple[str, Optional[str]]:
    """Loads Microsoft Word (.docx) document via python-docx."""
    try:
        import docx
    except (ImportError, Exception):
        return "", "python-docx кітапханасы орнатылмаған. / python-docx library is not installed."

    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs), None
    except Exception as e:
        return "", f"DOCX файлын оқу барысында қате шықты: {e} / Error reading DOCX file: {e}"


def _load_pdf(file_path: str) -> Tuple[str, Optional[str]]:
    """Loads PDF document via pypdf or PyPDF2."""
    pypdf_module = None
    try:
        import pypdf
        pypdf_module = pypdf
    except (ImportError, Exception):
        try:
            import PyPDF2
            pypdf_module = PyPDF2
        except (ImportError, Exception):
            return "", "pypdf немесе PyPDF2 кітапханасы орнатылмаған. / pypdf or PyPDF2 library is not installed."

    try:
        reader = pypdf_module.PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n\n".join(pages_text), None
    except Exception as e:
        return "", f"PDF файлын оқу барысында қате шықты: {e} / Error reading PDF file: {e}"


def load_document_file(
    file_obj_or_path: Any,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[str, Optional[str]]:
    """Loads and extracts text from a file (.txt, .docx, .pdf) defensively.

    Args:
        file_obj_or_path: Path string, Path object, or Gradio file wrapper.
        max_size_bytes: Maximum allowed file size in bytes (default 10 MB).

    Returns:
        Tuple of (extracted_text, error_or_warning_message).
        If extraction is successful and within limits, error message is None.
        If file exceeds 25,000 words, text is truncated and a warning is returned.
    """
    file_path = _resolve_file_path(file_obj_or_path)
    if file_path is None:
        return "", "Файл таңдалмады немесе файл жолы жарамсыз. / No file provided or invalid path."

    if not os.path.exists(file_path):
        return "", f"Файл табылмады: {file_path} / File not found: {file_path}"

    if not os.path.isfile(file_path):
        return "", f"Көрсетілген жол файл емес: {file_path} / Path is not a file: {file_path}"

    # Size check before reading file into memory
    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        return "", f"Файл өлшемін анықтау қатесі: {e} / File size determination error: {e}"

    if file_size > max_size_bytes:
        return "", SIZE_EXCEEDED_MESSAGE

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        text, err = _load_txt(file_path)
    elif ext == ".docx":
        text, err = _load_docx(file_path)
    elif ext == ".pdf":
        text, err = _load_pdf(file_path)
    else:
        return (
            "",
            f"Қолдау көрсетілмейтін файл пішімі ({ext}). Тек .txt, .docx, .pdf қолдау көрсетіледі. / "
            f"Unsupported file format ({ext}). Only .txt, .docx, .pdf are supported.",
        )

    if err is not None:
        return "", err

    # Word limit capping (prevent memory bombs while preserving original formatting)
    match_count = 0
    end_offset = None
    for m in re.finditer(r"\S+", text):
        match_count += 1
        if match_count == MAX_WORD_LIMIT:
            end_offset = m.end()
        elif match_count > MAX_WORD_LIMIT:
            truncated_text = text[:end_offset]
            return truncated_text, TRUNCATION_WARNING

    return text, None
