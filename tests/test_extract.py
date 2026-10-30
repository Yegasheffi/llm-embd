from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from embd.config import load_config
from embd.extract import UnsupportedFile, extract_text, is_supported


def _zip(members: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class ExtractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config = load_config(home=self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_html_strips_scripts_and_styles(self) -> None:
        data = b"<html><style>x{}</style><body><h1>Title</h1><script>bad()</script><p>Hello &amp; world</p></body></html>"
        result = extract_text("page.html", data, self.config)
        self.assertIn("Title", result.text)
        self.assertIn("Hello & world", result.text)
        self.assertNotIn("bad()", result.text)
        self.assertNotIn("x{}", result.text)

    def test_docx(self) -> None:
        xml = """<?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>第一段</w:t></w:r></w:p>
            <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>
          </w:body>
        </w:document>"""
        data = _zip({"word/document.xml": xml})
        result = extract_text("note.docx", data, self.config)
        self.assertIn("第一段", result.text)
        self.assertIn("Second paragraph", result.text)
        self.assertEqual(result.kind, "docx")

    def test_pptx(self) -> None:
        xml = """<?xml version="1.0"?>
        <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
               xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
        </p:sld>"""
        data = _zip({"ppt/slides/slide1.xml": xml})
        result = extract_text("deck.pptx", data, self.config)
        self.assertIn("Slide text", result.text)

    def test_xlsx(self) -> None:
        shared = """<?xml version="1.0"?>
        <sst xmlns:s="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <s:si><s:t>Name</s:t></s:si><s:si><s:t>Alice</s:t></s:si>
        </sst>"""
        sheet = """<?xml version="1.0"?>
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c></row>
            <row r="2"><c r="A2" t="s"><v>1</v></c></row>
          </sheetData>
        </worksheet>"""
        data = _zip({
            "xl/sharedStrings.xml": shared,
            "xl/worksheets/sheet1.xml": sheet,
        })
        result = extract_text("table.xlsx", data, self.config)
        self.assertIn("Name", result.text)
        self.assertIn("Alice", result.text)

    def test_ipynb(self) -> None:
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "body"]},
                {"cell_type": "code", "source": ["print('hi')"]},
            ]
        }
        result = extract_text("nb.ipynb", json.dumps(notebook).encode(), self.config)
        self.assertIn("# Title", result.text)
        self.assertIn("print('hi')", result.text)

    def test_binary_is_rejected(self) -> None:
        with self.assertRaises(UnsupportedFile):
            extract_text("blob.txt", b"\x00\x01\x02" * 100, self.config)

    def test_supported_extensions(self) -> None:
        self.assertTrue(is_supported("a.md", self.config))
        self.assertTrue(is_supported("Makefile", self.config))
        self.assertFalse(is_supported("a.exe", self.config))
        self.assertFalse(is_supported("a.png", self.config))


if __name__ == "__main__":
    unittest.main()
