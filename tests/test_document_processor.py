from core.document_processor import DocumentProcessor


def test_split_text_uses_stable_ids_and_overlap():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
    chunks = processor.split_text("abcdefghijklmnopqrstuvwxyz", "demo.md", "demo123")

    assert [chunk.chunk_id for chunk in chunks] == [
        "demo123_chunk_0000",
        "demo123_chunk_0001",
        "demo123_chunk_0002",
    ]
    assert chunks[0].text == "abcdefghij"
    assert chunks[1].text.startswith("ij")
    assert all(chunk.filename == "demo.md" for chunk in chunks)


def test_parse_markdown_bytes():
    processor = DocumentProcessor()
    assert processor.parse_uploaded_file("note.md", "# 标题".encode("utf-8")) == "# 标题"
