
import hashlib
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = 350


def load_pdf(file_path):
    """Extract text from each PDF page."""
    records = []
    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            records.append({
                "text": text.strip(),
                "page": page_number,
            })

    return records


def load_docx(file_path):
    """Extract text from a Word document."""
    document = Document(str(file_path))

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs)

    if not text:
        return []

    return [{
        "text": text,
        "page": None,
    }]


def load_documents(raw_dir=RAW_DATA_DIR):
    """Load all supported documents from department folders."""
    documents = []
    supported_extensions = {".pdf", ".docx"}

    for file_path in sorted(Path(raw_dir).rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        relative_path = file_path.relative_to(raw_dir)
        department = relative_path.parts[0]

        if file_path.suffix.lower() == ".pdf":
            pages = load_pdf(file_path)
        else:
            pages = load_docx(file_path)

        file_hash = hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()

        for page_record in pages:
            documents.append({
                "text": page_record["text"],
                "filename": file_path.name,
                "file_path": str(relative_path),
                "department": department,
                "file_type": file_path.suffix.lower().replace(".", ""),
                "page": page_record["page"],
                "file_hash": file_hash,
            })

    return documents


def split_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks while keeping sentences together."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<!\d\.)(?<=[.!?])\s+",
        text
    )

    chunks = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_length = 0

            words = sentence.split()
            part = ""

            for word in words:
                if part and len(part) + len(word) + 1 > chunk_size:
                    chunks.append(part)
                    part = word
                else:
                    part = f"{part} {word}".strip()

            if part:
                chunks.append(part)

            continue

        added_length = len(sentence) + (
            1 if current_sentences else 0
        )

        if current_length + added_length > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_length = len(sentence)
        else:
            current_sentences.append(sentence)
            current_length += added_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def create_chunks(documents):
    """Create chunks with metadata and deterministic IDs."""
    all_chunks = []

    for document in documents:
        text_chunks = split_text(document["text"])

        for index, chunk_text in enumerate(text_chunks):
            unique_value = (
                f'{document["file_path"]}|'
                f'{document["page"]}|'
                f'{index}|'
                f'{chunk_text}'
            )

            chunk_id = hashlib.sha256(
                unique_value.encode("utf-8")
            ).hexdigest()

            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "filename": document["filename"],
                "file_path": document["file_path"],
                "department": document["department"],
                "file_type": document["file_type"],
                "page": document["page"],
                "file_hash": document["file_hash"],
                "chunk_index": index,
            }

            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    print("Loading policy documents...")

    documents = load_documents()
    print(f"Documents/pages loaded: {len(documents)}")

    print("Creating sentence-aware chunks...")

    chunks = create_chunks(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("\nFirst chunk preview:")

    if chunks:
        print("Filename:", chunks[0]["filename"])
        print("Department:", chunks[0]["department"])
        print("Page:", chunks[0]["page"])
        print("Chunk ID:", chunks[0]["id"])
        print("Text:", chunks[0]["text"][:500])
    else:
        print("No text chunks found.")
