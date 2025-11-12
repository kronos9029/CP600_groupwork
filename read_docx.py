import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def extract_text(docx_path: Path) -> str:
    """Return plaintext from the main document body of a DOCX file."""
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        texts = [node.text for node in paragraph.iter(f"{namespace}t") if node.text]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python read_docx.py <path-to-docx>")
    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    print(extract_text(path))


if __name__ == "__main__":
    main()
