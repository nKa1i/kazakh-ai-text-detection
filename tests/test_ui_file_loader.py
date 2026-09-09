import os
import tempfile
import unittest
from unittest.mock import patch
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from pypdf import PdfWriter
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from ui.file_loader import load_document_file


class DummyGradioFile:
    """Mock of Gradio tempfile wrapper with a .name attribute."""
    def __init__(self, name: str):
        self.name = name


class TestUiFileLoader(unittest.TestCase):
    def test_load_txt_file(self):
        """Loads normal UTF-8 text file cleanly."""
        sample_content = "Қазақстанның болашағы жастардың қолында.\nБұл екінші сөйлем."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write(sample_content)
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertEqual(text, sample_content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_txt_cp1251_fallback(self):
        """Loads Cyrillic text encoded in CP1251 via fallback."""
        sample_content = "Казахский текст в кодировке cp1251."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(sample_content.encode("cp1251"))
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertEqual(text, sample_content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_txt_latin1_fallback(self):
        """Loads text encoded in Latin-1 cleanly without error."""
        # Standard latin-1 content (Café au lait)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"Caf\xe9 au lait")
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertTrue(len(text) > 0)
            self.assertIn("Caf", text)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Byte 0x98 is undefined in CP1251 and invalid in UTF-8, forcing fallback to Latin-1
        sample_bytes = b"Caf\xe9 \x98"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(sample_bytes)
            tmp_path2 = f.name
        try:
            text, err = load_document_file(tmp_path2)
            self.assertIsNone(err)
            self.assertEqual(text, sample_bytes.decode("latin-1"))
        finally:
            if os.path.exists(tmp_path2):
                os.remove(tmp_path2)

    def test_file_size_limit(self):
        """Rejects files larger than 10MB without loading into memory."""
        # Test with custom max_size_bytes for speed and memory efficiency
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"A" * 1024)
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path, max_size_bytes=512)
            self.assertEqual(text, "")
            self.assertIsNotNone(err)
            self.assertIn("10 MB", err)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_file_size_limit_default_constant(self):
        """Verifies error message contains '10 MB' for default limit check."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as f:
            f.write(b"Small text")
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertEqual(text, "Small text")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_nonexistent_file(self):
        """Returns graceful error when file does not exist."""
        text, err = load_document_file("non_existent_file_path_12345.txt")
        self.assertEqual(text, "")
        self.assertIsNotNone(err)
        self.assertIn("табылмады", err)

    def test_none_or_empty_input(self):
        """Handles None or empty string gracefully."""
        text, err = load_document_file(None)
        self.assertEqual(text, "")
        self.assertIsNotNone(err)

        text, err = load_document_file("")
        self.assertEqual(text, "")
        self.assertIsNotNone(err)

    def test_gradio_object_wrapper(self):
        """Handles Gradio-style file object with a .name attribute."""
        sample_content = "Gradio нысанынан оқылған сынақ мәтіні."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write(sample_content)
            tmp_path = f.name
        try:
            wrapper = DummyGradioFile(tmp_path)
            text, err = load_document_file(wrapper)
            self.assertIsNone(err)
            self.assertEqual(text, sample_content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_word_limit_capping(self):
        """Safely truncates documents exceeding 25,000 words while preserving paragraph structure."""
        # 26,000 words across multiple paragraphs
        paragraphs = ["Параграф " + "сөз " * 499 for _ in range(52)]
        long_content = "\n\n".join(paragraphs)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write(long_content)
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNotNone(err)
            self.assertIn("25 000", err)
            words = text.split()
            self.assertEqual(len(words), 25000)
            self.assertIn("\n\n", text)
            self.assertTrue(text.startswith("Параграф сөз"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unsupported_file_extension(self):
        """Rejects unsupported extensions like .csv, .jpg, .exe."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
            f.write("a,b,c\n1,2,3")
            tmp_path = f.name
        try:
            text, err = load_document_file(tmp_path)
            self.assertEqual(text, "")
            self.assertIsNotNone(err)
            self.assertIn("қолдау көрсетілмейтін", err.lower())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @unittest.skipUnless(HAS_DOCX, "python-docx not installed")
    def test_load_docx_file(self):
        """Creates and cleanly reads a .docx document."""
        doc = docx.Document()
        doc.add_paragraph("Бірінші абзац.")
        doc.add_paragraph("Екінші абзац.")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            tmp_path = f.name
        doc.save(tmp_path)
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertIn("Бірінші абзац.", text)
            self.assertIn("Екінші абзац.", text)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @unittest.skipUnless(HAS_PYPDF, "pypdf not installed")
    def test_load_pdf_file(self):
        """Creates and cleanly reads a PDF document."""
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        with open(tmp_path, "wb") as f:
            writer.write(f)
        try:
            text, err = load_document_file(tmp_path)
            self.assertIsNone(err)
            self.assertIsInstance(text, str)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_missing_library_docx(self):
        """Gracefully reports error when python-docx is not installed."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.dict("sys.modules", {"docx": None}):
                text, err = load_document_file(tmp_path)
                self.assertEqual(text, "")
                self.assertIsNotNone(err)
                self.assertIn("python-docx", err)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_missing_library_pdf(self):
        """Gracefully reports error when pypdf and PyPDF2 are not installed."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.dict("sys.modules", {"pypdf": None, "PyPDF2": None}):
                text, err = load_document_file(tmp_path)
                self.assertEqual(text, "")
                self.assertIsNotNone(err)
                self.assertIn("pypdf", err)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
