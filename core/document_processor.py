from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_uploaded_file(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(content)
        if suffix in {".txt", ".md", ".markdown"}:
            return content.decode("utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {suffix}")

    def split_text(self, text: str, filename: str, document_id: str) -> list[TextChunk]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        idx = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_chunk_{idx:04d}",
                        document_id=document_id,
                        filename=filename,
                        text=chunk_text,
                        chunk_index=idx,
                    )
                )
                idx += 1
            if end >= len(normalized):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def _parse_pdf(self, content: bytes) -> str:
        from io import BytesIO

        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
