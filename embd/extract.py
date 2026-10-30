"""Text extraction for the file types a local RAG index actually meets.

Plain text and source code are decoded directly. Office formats are ZIP+XML, so
they can be handled with the standard library (no python-docx / python-pptx).
PDF needs an optional parser; if ``pypdf`` is installed it is used, otherwise
the file is reported as skipped with a clear message instead of silently
producing garbage.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .config import Config


class ExtractError(RuntimeError):
    pass


class UnsupportedFile(ExtractError):
    pass


@dataclass(frozen=True, slots=True)
class Extracted:
    text: str
    kind: str


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def is_supported(path: str | Path, config: Config) -> bool:
    p = Path(path)
    name = p.name
    lower = name.lower()
    if lower in {x.lower() for x in config.index["text_basenames"]}:
        return True
    suffix = p.suffix.lower()
    if suffix in {x.lower() for x in config.index["text_extensions"]}:
        return True
    # Extension-less source files such as Makefile / LICENSE.
    return not suffix and bool(re.match(r"^[A-Z][A-Z0-9_.-]{1,32}$", name))


def extract_text(path: str | Path, data: bytes, config: Config) -> Extracted:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        return Extracted(_extract_docx(data), "docx")
    if suffix == ".pptx":
        return Extracted(_extract_pptx(data), "pptx")
    if suffix == ".xlsx":
        return Extracted(_extract_xlsx(data), "xlsx")
    if suffix == ".ipynb":
        return Extracted(_extract_notebook(data), "ipynb")
    if suffix in {".html", ".htm"}:
        return Extracted(_extract_html(data), "html")
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        return Extracted(_extract_pdf(data), "pdf")
    return Extracted(_decode_text(data), suffix.lstrip(".") or "text")


def _decode_text(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise UnsupportedFile("looks like binary data (NUL byte)")
    text: str
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Last resort: latin-1 always decodes, but reject if it looks binary.
            text = data.decode("latin-1")

    if not text:
        return ""
    control = sum(1 for ch in text[:8192] if ord(ch) < 9 or 13 < ord(ch) < 32)
    if control > max(8, len(text[:8192]) * 0.02):
        raise UnsupportedFile("looks like binary data (too many control characters)")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _zip_xml(data: bytes, member: str) -> ET.Element:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            raw = archive.read(member)
    except KeyError as exc:
        raise UnsupportedFile(f"missing {member} in Office archive") from exc
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile("invalid Office archive") from exc
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise UnsupportedFile("invalid XML inside Office document") from exc


def _extract_docx(data: bytes) -> str:
    root = _zip_xml(data, "word/document.xml")
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{_W}p"):
        pieces: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t" and node.text:
                pieces.append(node.text)
            elif node.tag == f"{_W}tab":
                pieces.append("\t")
            elif node.tag in {f"{_W}br", f"{_W}cr"}:
                pieces.append("\n")
        line = "".join(pieces).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _extract_pptx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = sorted(
                (n for n in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                key=lambda n: int(re.search(r"(\d+)", n).group(1)),  # type: ignore[union-attr]
            )
            slides: list[str] = []
            for index, name in enumerate(names, 1):
                raw = archive.read(name)
                try:
                    root = ET.fromstring(raw)
                except ET.ParseError:
                    continue
                texts = [node.text for node in root.iter(f"{_A}t") if node.text]
                if texts:
                    slides.append(f"# Slide {index}\n" + "\n".join(texts))
            return "\n\n".join(slides)
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile("invalid PPTX archive") from exc


def _extract_xlsx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.iter(f"{_S}si"):
                    shared.append("".join(t.text or "" for t in item.iter(f"{_S}t")))
            sheet_names = sorted(
                (n for n in archive.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                key=lambda n: int(re.search(r"(\d+)", n).group(1)),  # type: ignore[union-attr]
            )
            chunks: list[str] = []
            for sheet_index, name in enumerate(sheet_names, 1):
                root = ET.fromstring(archive.read(name))
                rows: list[str] = []
                for row in root.iter(f"{_S}row"):
                    values: list[str] = []
                    for cell in row.iter(f"{_S}c"):
                        kind = cell.get("t")
                        if kind == "inlineStr":
                            value = "".join(t.text or "" for t in cell.iter(f"{_S}t"))
                        else:
                            raw = cell.find(f"{_S}v")
                            value = raw.text if raw is not None and raw.text else ""
                            if kind == "s":
                                try:
                                    value = shared[int(value)]
                                except (ValueError, IndexError):
                                    pass
                        if value:
                            values.append(value.strip())
                    if values:
                        rows.append("\t".join(values))
                if rows:
                    chunks.append(f"# Sheet {sheet_index}\n" + "\n".join(rows))
            return "\n\n".join(chunks)
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile("invalid XLSX archive") from exc


class _HTMLText(HTMLParser):
    _BLOCK = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "pre", "blockquote", "table",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts).replace("\r", "")
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r" *\n *", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _extract_html(data: bytes) -> str:
    parser = _HTMLText()
    parser.feed(_decode_text(data))
    parser.close()
    return parser.text()


def _extract_notebook(data: bytes) -> str:
    try:
        notebook = json.loads(_decode_text(data))
    except json.JSONDecodeError as exc:
        raise UnsupportedFile("invalid .ipynb JSON") from exc
    cells = notebook.get("cells", [])
    blocks: list[str] = []
    for cell in cells:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            kind = cell.get("cell_type", "cell")
            blocks.append(f"## [{kind}]\n{source.strip()}")
    return "\n\n".join(blocks)


def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
        except ImportError as exc:
            raise UnsupportedFile(
                "PDF 需要可选依赖：pip install pypdf（其余格式零依赖）"
            ) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as exc:  # pragma: no cover - depends on malformed PDFs
        raise UnsupportedFile(f"PDF 解析失败：{exc}") from exc
