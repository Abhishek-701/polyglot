"""HTML/PDF section extraction, tested with in-memory fixtures (no network).
Real-world PDF text extraction quality is checked once config/sources.yaml
has real documents to ingest (blocked on the human filling it in).
"""

import io

from pypdf import PdfWriter

from polyglot.retrieval.extract import extract_sections


def test_html_sections_split_on_headings() -> None:
    html = b"""
    <html><body>
    <nav>Home | About</nav>
    <article>
    <h1>Refunds</h1>
    <p>You may be entitled to a refund if your flight is cancelled.</p>
    <h2>How to request</h2>
    <p>Contact the airline within 30 days.</p>
    </article>
    <footer>Copyright 2024</footer>
    </body></html>
    """
    sections = extract_sections(html, "text/html", "https://example.com")
    assert sections == [
        ("Refunds", "You may be entitled to a refund if your flight is cancelled."),
        ("How to request", "Contact the airline within 30 days."),
    ]


def test_html_boilerplate_tags_are_stripped() -> None:
    html = (
        b"<html><body><nav>NAVLINK</nav><p>Real content</p>"
        b"<footer>FOOTERLINK</footer></body></html>"
    )
    sections = extract_sections(html, "text/html", "https://example.com")
    text = " ".join(section_text for _heading, section_text in sections)
    assert "NAVLINK" not in text
    assert "FOOTERLINK" not in text
    assert "Real content" in text


def test_html_with_no_headings_falls_back_to_document_section() -> None:
    html = b"<html><body><p>Just one paragraph, no headings.</p></body></html>"
    sections = extract_sections(html, "text/html", "https://example.com")
    assert sections == [("document", "Just one paragraph, no headings.")]


def test_pdf_extraction_does_not_crash_and_returns_a_list() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    sections = extract_sections(buf.getvalue(), "application/pdf", "https://example.com/doc.pdf")
    assert isinstance(sections, list)
