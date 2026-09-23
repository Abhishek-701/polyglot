"""HTML/PDF text extraction with heading-preserving sectioning. See SPEC.md
Section 10.1.

Kept deliberately simple: strip obvious boilerplate tags, then split on
h1-h6 into (heading, section_text) pairs. PDFs don't have reliable heading
markup without layout analysis, so PDF sections are per-page.
"""

import io

from bs4 import BeautifulSoup
from pypdf import PdfReader

_BOILERPLATE_TAGS = ["nav", "header", "footer", "script", "style", "aside", "form", "noscript"]
_HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


def extract_sections(content: bytes, content_type: str, url: str) -> list[tuple[str, str]]:
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return _extract_pdf_sections(content)
    return _extract_html_sections(content)


def _extract_html_sections(content: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(content, "lxml")
    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = soup.body or soup
    sections: list[tuple[str, list[str]]] = []
    current_heading = "document"
    current_paragraphs: list[str] = []

    for element in body.find_all([*_HEADING_TAGS, "p", "li"]):
        if element.name in _HEADING_TAGS:
            if current_paragraphs:
                sections.append((current_heading, current_paragraphs))
            current_heading = element.get_text(strip=True) or current_heading
            current_paragraphs = []
        else:
            text = element.get_text(strip=True)
            if text:
                current_paragraphs.append(text)

    if current_paragraphs:
        sections.append((current_heading, current_paragraphs))

    return [(heading, " ".join(paragraphs)) for heading, paragraphs in sections if paragraphs]


def _extract_pdf_sections(content: bytes) -> list[tuple[str, str]]:
    reader = PdfReader(io.BytesIO(content))
    sections = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append((f"page {i + 1}", text))
    return sections
