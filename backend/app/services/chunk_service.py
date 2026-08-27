from typing import List


def chunk_text(text: str, max_words: int = 220, min_words: int = 40) -> List[str]:
    """
    Paragraph-wise chunking for research papers.
    - Splits by paragraphs
    - Merges short paragraphs
    - Splits very long paragraphs
    """
    if not text or not text.strip():
        return []

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Further split single newlines if paragraph is still huge
    paragraphs = []
    for p in raw_paragraphs:
        if len(p.split()) > max_words * 2:
            parts = [x.strip() for x in p.split("\n") if x.strip()]
            paragraphs.extend(parts if parts else [p])
        else:
            paragraphs.append(p)

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    def flush():
        nonlocal current, current_words
        if current:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
        current = []
        current_words = 0

    for para in paragraphs:
        words = para.split()
        word_count = len(words)

        # Very long paragraph → split by sentences/words
        if word_count > max_words:
            flush()
            start = 0
            while start < word_count:
                end = min(start + max_words, word_count)
                piece = " ".join(words[start:end]).strip()
                if piece:
                    chunks.append(piece)
                # small overlap
                start = end - 30 if end < word_count else end
            continue

        # If adding this paragraph exceeds max → flush first
        if current_words + word_count > max_words and current_words >= min_words:
            flush()

        current.append(para)
        current_words += word_count

    flush()

    # Remove tiny noisy chunks
    cleaned = [c for c in chunks if len(c.split()) >= 20]
    return cleaned if cleaned else chunks