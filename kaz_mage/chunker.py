"""
Kazakh Document Chunking Module.

Provides sentence-preserving sliding window chunking with linguistic protections
for Kazakh abbreviations (т.б., ж.б., ғ., ғғ., ж., жж., қ., мыс., проф., акад.),
ordinal/cardinal numbers, and dialogue attribution quotes.
"""

import re
from typing import List, Tuple
from .document import DocumentChunk


class SentencePreservingChunker:
    """
    Splits long Kazakh texts into sentence-preserving sliding window chunks
    with exact character span offsets and linguistic boundary protections.
    """

    # Multi-dot abbreviations (e.g., т.б., ж.б., т.с.с., б.з.д., б.з.б., б.з.)
    _ABBR_MULTI = re.compile(
        r'\b(?:т\.\s*б|ж\.\s*б|т\.\s*с\.\s*с|б\.\s*з\.\s*д|б\.\s*з\.\s*б|б\.\s*з)\.',
        re.IGNORECASE | re.UNICODE
    )

    # Single-word abbreviations (e.g., ғ., ғғ., ж., жж., қ., мыс., проф., акад., доц.)
    _ABBR_SINGLE = re.compile(
        r'\b(?:ғ|ғғ|ж|жж|қ|мыс|проф|акад|доц)\.',
        re.IGNORECASE | re.UNICODE
    )

    # Digit followed by period (e.g., 2024 ж., 1. орын, 3.14)
    _DIGIT_DOT = re.compile(
        r'(?<=\d)\.',
        re.UNICODE
    )

    # Quotes with terminal punctuation followed by dialogue attribution or lowercase letter
    _QUOTE_ATTRIB = re.compile(
        r'([.!?…]+)\s*'
        r'(?=[»”"\'’]\s*(?:[—–\-―]\s*)?(?:деді|деп|айтты|сұрады|жауап|бұйырды|мәлімдеді|жазды|үн|қосты|[a-zа-яәіңғүұқөһ]))',
        re.IGNORECASE | re.UNICODE
    )

    # Sentence boundary regex: terminal punctuation optionally followed by closing quotes/brackets,
    # followed by whitespace and an uppercase letter, quote, number, or dialogue dash, or end of string.
    _SENTENCE_SPLIT = re.compile(
        r'([.!?…]+[»”"\'’\)\]]*)(?=\s+(?:[A-ZА-ЯӘІҢҒҮҰҚӨҺ0-9«“"\'\(—–\-―]|\Z)|\Z)',
        re.UNICODE
    )

    def __init__(self, max_words: int = 200, overlap_sentences: int = 1):
        """
        Initialize the sentence-preserving chunker.

        Args:
            max_words: Target maximum word budget per chunk window.
            overlap_sentences: Number of sentences to overlap between successive windows.
        """
        if max_words < 1:
            raise ValueError(f"max_words must be at least 1, got {max_words}")
        self.max_words = max_words
        self.overlap_sentences = max(0, overlap_sentences)

    def split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Segment Kazakh text into sentences with exact character offsets.

        Applies abbreviation and quote protections to prevent incorrect splitting
        at abbreviation periods (т.б., ж., ғғ., etc.) and inside direct speech quotations.

        Returns:
            List of (sentence_text, start_char, end_char) where text[start_char:end_char] == sentence_text.
        """
        if not text or not text.strip():
            return []

        # Create a character-by-character mask of identical length
        # Using a dummy non-terminator character preserves exact character indexing
        masked = list(text)

        # 1. Mask multi-dot abbreviations
        for m in self._ABBR_MULTI.finditer(text):
            for idx in range(m.start(), m.end()):
                if text[idx] == '.':
                    masked[idx] = '\u0000'

        # 2. Mask single-word abbreviations
        for m in self._ABBR_SINGLE.finditer(text):
            for idx in range(m.start(), m.end()):
                if text[idx] == '.':
                    masked[idx] = '\u0000'

        # 3. Mask digit followed by period
        for m in self._DIGIT_DOT.finditer(text):
            masked[m.start()] = '\u0000'

        # 4. Mask terminal punctuation inside quotes followed by dialogue attribution
        for m in self._QUOTE_ATTRIB.finditer(text):
            for idx in range(m.start(1), m.end(1)):
                masked[idx] = '\u0000'

        masked_str = "".join(masked)
        sentences: List[Tuple[str, int, int]] = []
        cursor = 0

        for m in self._SENTENCE_SPLIT.finditer(masked_str):
            end_punct = m.end()

            # Advance cursor over leading whitespace
            while cursor < end_punct and text[cursor].isspace():
                cursor += 1

            if cursor < end_punct:
                sent_text = text[cursor:end_punct]
                sentences.append((sent_text, cursor, end_punct))
                cursor = end_punct

        # Handle any trailing text after the last sentence terminator
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        if cursor < len(text):
            # Trim any trailing whitespace from the document end
            end_char = len(text)
            while end_char > cursor and text[end_char - 1].isspace():
                end_char -= 1
            if end_char > cursor:
                sent_text = text[cursor:end_char]
                sentences.append((sent_text, cursor, end_char))

        return sentences

    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """
        Partition document into overlapping windows of sentences up to max_words budget.

        Returns:
            List of DocumentChunk instances with exact character offsets.
        """
        if not text or not text.strip():
            return []

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        sent_words = [len(s[0].split()) for s in sentences]
        chunks: List[DocumentChunk] = []
        window_start = 0
        chunk_idx = 0
        total_sents = len(sentences)

        while window_start < total_sents:
            current_words = 0
            window_end = window_start

            while window_end < total_sents:
                w = sent_words[window_end]
                # Always include at least one sentence per chunk even if it exceeds max_words
                if window_end > window_start and (current_words + w > self.max_words):
                    break
                current_words += w
                window_end += 1

            start_char = sentences[window_start][1]
            end_char = sentences[window_end - 1][2]
            chunk_text = text[start_char:end_char]
            word_count = len(chunk_text.split())
            sentence_count = window_end - window_start

            chunk = DocumentChunk(
                index=chunk_idx,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                word_count=word_count,
                sentence_count=sentence_count,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if window_end >= total_sents:
                break

            # Slide window start forward
            next_start = max(window_start + 1, window_end - self.overlap_sentences)
            window_start = next_start

        return chunks
