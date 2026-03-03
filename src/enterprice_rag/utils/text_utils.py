import re


def split_into_sentences(text: str) -> list[str]:
    """lightweight sentence splitter (works well for most English text)"""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
