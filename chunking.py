"""Text chunking shared by build_index.py and the tests."""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping word chunks.

    Each chunk has up to ``chunk_size`` words and repeats the last ``overlap``
    words of the previous chunk, so a passage cut at a boundary is still
    retrievable from one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks
