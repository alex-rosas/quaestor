"""Unit tests for src/quaestor/ingestion/loader.py.

PDF I/O is tested with a real minimal PDF built in-memory (no fixtures on disk).
SEC download is tested by mocking sec_edgar_downloader.Downloader so no
network calls are made.
"""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from quaestor.ingestion.loader import download_sec_filings, load_directory, load_pdf


# ---------------------------------------------------------------------------
# Minimal valid PDF builder
# ---------------------------------------------------------------------------

def _make_minimal_pdf(text: str = "Hello Quaestor") -> bytes:
    """Return the bytes of a single-page PDF containing *text*.

    Constructed entirely from spec so there is no third-party dependency in
    the test helper itself.
    """
    # Compress the page content stream
    stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    compressed = zlib.compress(stream_content)

    objects: list[bytes] = []

    def obj(n: int, content: bytes) -> None:
        objects.append(f"{n} 0 obj\n".encode() + content + b"\nendobj\n")

    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    obj(
        3,
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
    )
    stream_header = f"<< /Length {len(compressed)} /Filter /FlateDecode >>\n".encode()
    obj(4, stream_header + b"stream\n" + compressed + b"\nendstream")
    obj(
        5,
        (
            b"<< /Type /Font /Subtype /Type1 "
            b"/BaseFont /Helvetica >>"
        ),
    )

    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for o in objects:
        offsets.append(len(body))
        body += o

    xref_offset = len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return body + xref + trailer


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------

class TestLoadPdf:
    def test_returns_list_of_documents(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(_make_minimal_pdf("test content"))
        docs = load_pdf(pdf_path)
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_contains_source(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(_make_minimal_pdf())
        docs = load_pdf(pdf_path)
        assert "source" in docs[0].metadata

    def test_metadata_contains_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(_make_minimal_pdf())
        docs = load_pdf(pdf_path)
        assert "page" in docs[0].metadata

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "path_obj.pdf"
        pdf_path.write_bytes(_make_minimal_pdf())
        docs = load_pdf(pdf_path)  # Path, not str
        assert len(docs) >= 1

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "str_path.pdf"
        pdf_path.write_bytes(_make_minimal_pdf())
        docs = load_pdf(str(pdf_path))  # str, not Path
        assert len(docs) >= 1

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "ghost.pdf")

    def test_raises_value_error_for_non_pdf(self, tmp_path: Path) -> None:
        txt_path = tmp_path / "doc.txt"
        txt_path.write_text("not a pdf")
        with pytest.raises(ValueError, match=r"\.pdf"):
            load_pdf(txt_path)


# ---------------------------------------------------------------------------
# load_directory
# ---------------------------------------------------------------------------

class TestLoadDirectory:
    def test_loads_multiple_pdfs(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"doc{i}.pdf").write_bytes(_make_minimal_pdf(f"doc {i}"))
        docs = load_directory(tmp_path)
        assert len(docs) >= 3

    def test_returns_empty_list_when_no_pdfs(self, tmp_path: Path) -> None:
        docs = load_directory(tmp_path)
        assert docs == []

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.pdf").write_bytes(_make_minimal_pdf("nested"))
        docs = load_directory(tmp_path)
        assert len(docs) >= 1

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            load_directory(tmp_path / "does_not_exist")

    def test_raises_for_file_not_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("text")
        with pytest.raises(NotADirectoryError):
            load_directory(f)

    def test_ignores_non_pdf_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("ignore me")
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "report.pdf").write_bytes(_make_minimal_pdf("keep me"))
        docs = load_directory(tmp_path)
        assert len(docs) >= 1
        for doc in docs:
            assert doc.metadata["source"].endswith(".pdf")

    def test_processing_order_is_deterministic(self, tmp_path: Path) -> None:
        for name in ["z.pdf", "a.pdf", "m.pdf"]:
            (tmp_path / name).write_bytes(_make_minimal_pdf(name))
        docs1 = load_directory(tmp_path)
        docs2 = load_directory(tmp_path)
        sources1 = [d.metadata["source"] for d in docs1]
        sources2 = [d.metadata["source"] for d in docs2]
        assert sources1 == sources2


# ---------------------------------------------------------------------------
# download_sec_filings
# ---------------------------------------------------------------------------

class TestDownloadSecFilings:
    def _patched_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject non-empty requester credentials into settings."""
        monkeypatch.setattr(
            "quaestor.ingestion.loader.settings.sec_requester_name", "Test User"
        )
        monkeypatch.setattr(
            "quaestor.ingestion.loader.settings.sec_requester_email", "test@example.com"
        )

    def test_raises_if_name_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("quaestor.ingestion.loader.settings.sec_requester_name", "")
        monkeypatch.setattr(
            "quaestor.ingestion.loader.settings.sec_requester_email", "x@x.com"
        )
        with pytest.raises(ValueError, match="SEC_REQUESTER_NAME"):
            download_sec_filings("AAPL", download_dir=tmp_path)

    def test_raises_if_email_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "quaestor.ingestion.loader.settings.sec_requester_name", "Test"
        )
        monkeypatch.setattr("quaestor.ingestion.loader.settings.sec_requester_email", "")
        with pytest.raises(ValueError, match="SEC_REQUESTER_EMAIL"):
            download_sec_filings("AAPL", download_dir=tmp_path)

    def test_calls_downloader_with_correct_args(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patched_settings(monkeypatch)
        mock_dl_instance = MagicMock()
        mock_dl_cls = MagicMock(return_value=mock_dl_instance)

        with patch("quaestor.ingestion.loader.Downloader", mock_dl_cls):
            download_sec_filings("AAPL", form="10-K", limit=2, download_dir=tmp_path)

        mock_dl_cls.assert_called_once_with("Test User", "test@example.com", tmp_path)
        mock_dl_instance.get.assert_called_once_with("10-K", "AAPL", limit=2)

    def test_returns_expected_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patched_settings(monkeypatch)
        with patch("quaestor.ingestion.loader.Downloader", MagicMock()):
            result = download_sec_filings("JPM", form="10-K", limit=1, download_dir=tmp_path)
        assert result == tmp_path / "sec-edgar-filings" / "JPM" / "10-K"

    def test_creates_download_dir_if_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patched_settings(monkeypatch)
        new_dir = tmp_path / "deep" / "nested"
        with patch("quaestor.ingestion.loader.Downloader", MagicMock()):
            download_sec_filings("AAPL", download_dir=new_dir)
        assert new_dir.is_dir()

    def test_default_download_dir_uses_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patched_settings(monkeypatch)
        monkeypatch.setattr(
            "quaestor.ingestion.loader.settings.data_raw_dir", tmp_path
        )
        mock_dl = MagicMock()
        with patch("quaestor.ingestion.loader.Downloader", MagicMock(return_value=mock_dl)):
            result = download_sec_filings("AAPL")
        expected_dir = tmp_path / "sec_filings"
        assert result == expected_dir / "sec-edgar-filings" / "AAPL" / "10-K"
