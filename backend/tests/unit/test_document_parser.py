from unittest.mock import patch

from app.rag.document_parser import parse_document


def test_markdown_is_read_as_utf8_text():
    assert parse_document("readme.md", "# 标题\n正文".encode()) == "# 标题\n正文"


def test_pdf_pages_are_joined_as_searchable_text():
    class Page:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self):
            return self.text

    with patch("app.rag.document_parser.PdfReader") as reader:
        reader.return_value.pages = [Page("第一页政策"), Page("第二页条款")]
        text = parse_document("policy.pdf", b"valid-pdf-placeholder")

    assert text == "第一页政策\n\n第二页条款"


def test_scanned_pdf_pages_fall_back_to_ocr():
    class EmptyPage:
        def extract_text(self):
            return ""

    with (
        patch("app.rag.document_parser.PdfReader") as reader,
        patch(
            "app.rag.document_parser._ocr_pdf_pages",
            return_value={0: "OCR 识别出的中文批复内容"},
        ) as ocr,
    ):
        reader.return_value.pages = [EmptyPage()]
        text = parse_document("scan.pdf", b"scanned-pdf")

    assert text == "OCR 识别出的中文批复内容"
    ocr.assert_called_once_with(b"scanned-pdf", [0])
