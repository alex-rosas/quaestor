"""Document loading for Quaestor.

Provides four entry points:
- load_pdf              — load a single PDF from disk into LangChain Documents
- load_directory        — batch-load every PDF in a directory tree
- download_sec_filings  — pull SEC EDGAR filings via sec-edgar-downloader
- load_edgar_submission — extract & load the main document from a
                          full-submission.txt (EDGAR SGML container)
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from quaestor.config import settings

logger = logging.getLogger(__name__)

try:
    from sec_edgar_downloader import Downloader  # type: ignore[import]
except ImportError:  # pragma: no cover
    Downloader = None  # type: ignore[assignment,misc]


def load_pdf(path: Path) -> list[Document]:
    """Load a single PDF file and return one Document per page.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of Documents, one per page, each with metadata fields
        ``source`` (str path) and ``page`` (0-based int).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *path* is not a ``.pdf`` file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info("Loading PDF: %s", path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    logger.info("Loaded %d pages from %s", len(docs), path.name)
    return docs


def load_directory(directory: Path) -> list[Document]:
    """Recursively load all PDF files found under *directory*.

    Args:
        directory: Root directory to search.

    Returns:
        Concatenated list of Documents from every PDF found.
        Files are processed in sorted order for determinism.

    Raises:
        NotADirectoryError: If *directory* does not exist or is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pdf_paths = sorted(directory.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDF files found under %s", directory)
        return []

    logger.info("Found %d PDF file(s) in %s", len(pdf_paths), directory)
    all_docs: list[Document] = []
    for pdf_path in pdf_paths:
        all_docs.extend(load_pdf(pdf_path))
    logger.info("Loaded %d total pages from %s", len(all_docs), directory)
    return all_docs


def download_sec_filings(
    ticker: str,
    form: str = "10-K",
    limit: int = 1,
    download_dir: Path | None = None,
) -> Path:
    """Download SEC EDGAR filings for *ticker* into *download_dir*.

    Uses the User-Agent identity from ``settings.sec_requester_name`` and
    ``settings.sec_requester_email`` — both must be non-empty.

    Args:
        ticker:       Stock ticker symbol, e.g. ``"AAPL"``.
        form:         SEC form type, e.g. ``"10-K"`` or ``"10-Q"``.
        limit:        Maximum number of filings to download.
        download_dir: Root directory for downloaded filings.
                      Defaults to ``settings.data_raw_dir / "sec_filings"``.

    Returns:
        Path to the directory where filings were saved.

    Raises:
        ValueError: If requester name or email are not configured.
        ImportError: If ``sec_edgar_downloader`` is not installed.
    """
    if not settings.sec_requester_name or not settings.sec_requester_email:
        raise ValueError(
            "SEC_REQUESTER_NAME and SEC_REQUESTER_EMAIL must be set in .env "
            "before downloading EDGAR filings."
        )

    if Downloader is None:  # pragma: no cover
        raise ImportError(
            "sec-edgar-downloader is not installed. Run: uv add sec-edgar-downloader"
        )

    if download_dir is None:
        download_dir = settings.data_raw_dir / "sec_filings"

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %d %s filing(s) for %s → %s",
        limit, form, ticker, download_dir,
    )
    dl = Downloader(
        settings.sec_requester_name,
        settings.sec_requester_email,
        download_dir,
    )
    dl.get(form, ticker, limit=limit)

    # sec-edgar-downloader v5 uses hyphens in the directory name
    ticker_dir = download_dir / "sec-edgar-filings" / ticker / form
    logger.info("Filings saved to %s", ticker_dir)
    return ticker_dir


def load_edgar_submission(submission_dir: Path) -> list[Document]:
    """Load documents from an SEC EDGAR filing directory.

    SEC EDGAR delivers filings as a ``full-submission.txt`` SGML container
    that embeds one or more documents (10-K HTML, exhibits, XBRL data).
    This function finds the primary filing document (the first ``<TYPE>10-K``
    or ``<TYPE>10-Q`` block), extracts its HTML content, saves it alongside
    the submission, and loads it into LangChain Documents.

    Falls back to loading any ``.htm`` / ``.html`` files already present on
    disk (useful when the submission was already extracted).

    Args:
        submission_dir: Directory containing ``full-submission.txt`` or
                        already-extracted ``.htm`` files.

    Returns:
        List of Documents with ``source``, ``page``, and ``filing_type``
        metadata fields.

    Raises:
        FileNotFoundError: If *submission_dir* does not exist.
        ValueError:        If no loadable content can be found.
    """
    submission_dir = Path(submission_dir)
    if not submission_dir.is_dir():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    # --- Try already-extracted HTML files first ---
    existing_htm = sorted(submission_dir.rglob("*.htm")) + sorted(
        submission_dir.rglob("*.html")
    )
    if existing_htm:
        return _load_htm_files(existing_htm)

    # --- Extract from full-submission.txt ---
    full_submission = submission_dir / "full-submission.txt"
    if not full_submission.exists():
        raise ValueError(
            f"No .htm files or full-submission.txt found in {submission_dir}"
        )

    logger.info("Extracting documents from %s", full_submission)
    extracted = _extract_from_sgml(full_submission, submission_dir)

    if not extracted:
        raise ValueError(
            f"Could not extract any documents from {full_submission}"
        )

    return _load_htm_files(extracted)


# ---------------------------------------------------------------------------
# Private helpers for load_edgar_submission
# ---------------------------------------------------------------------------

def _extract_from_sgml(sgml_path: Path, output_dir: Path) -> list[Path]:
    """Parse EDGAR SGML container and write embedded HTML docs to *output_dir*.

    Returns the list of paths written.
    """
    primary_types = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    extracted: list[Path] = []

    text = sgml_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "<DOCUMENT>":
            # Scan header fields until <TEXT>
            doc_type = ""
            filename = ""
            j = i + 1
            while j < len(lines) and lines[j].strip() != "<TEXT>":
                stripped = lines[j].strip()
                if stripped.startswith("<TYPE>"):
                    doc_type = stripped[6:].strip()
                elif stripped.startswith("<FILENAME>"):
                    filename = stripped[10:].strip()
                j += 1

            if doc_type in primary_types and j < len(lines):
                # Collect content between <TEXT> and </TEXT>
                content_lines: list[str] = []
                k = j + 1
                while k < len(lines) and lines[k].strip() != "</TEXT>":
                    content_lines.append(lines[k])
                    k += 1

                if content_lines:
                    out_name = filename or f"{doc_type.replace('/', '_')}.htm"
                    out_path = output_dir / out_name
                    out_path.write_text("".join(content_lines), encoding="utf-8")
                    logger.info("Extracted %s → %s (%d lines)", doc_type, out_name, len(content_lines))
                    extracted.append(out_path)
                    i = k
                    continue
        i += 1

    return extracted


def _load_htm_files(paths: list[Path]) -> list[Document]:
    """Load HTML/HTM files using BeautifulSoup to strip tags."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]
        _use_bs4 = True
    except ImportError:
        _use_bs4 = False

    docs: list[Document] = []
    for path in paths:
        raw = path.read_text(encoding="utf-8", errors="replace")

        if _use_bs4:
            soup = BeautifulSoup(raw, "html.parser")
            # Remove script/style noise
            for tag in soup(["script", "style", "ix:header", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            # Minimal tag stripper — good enough for XBRL-inline HTML
            import re
            text = re.sub(r"<[^>]+>", " ", raw)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)

        # Split into ~page-sized chunks for metadata consistency
        chunk_chars = 4000
        for idx, start in enumerate(range(0, len(text), chunk_chars)):
            snippet = text[start : start + chunk_chars].strip()
            if snippet:
                docs.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "source": str(path),
                            "page": idx,
                            "filing_type": path.suffix,
                        },
                    )
                )

    logger.info("Loaded %d document segment(s) from %d file(s)", len(docs), len(paths))
    return docs
