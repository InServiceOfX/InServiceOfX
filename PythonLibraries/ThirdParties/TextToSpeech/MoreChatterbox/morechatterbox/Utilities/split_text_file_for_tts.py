"""
Split a text file into segment files for TTS (e.g. Chatterbox) that works
better with shorter inputs. Segments are split at sentence boundaries;
single sentences longer than max_chars become their own segment.
"""

from pathlib import Path
from typing import List
import re


def split_text_file_for_tts(
    text_file_path: Path,
    max_chars: int = 380,
    output_stem_suffix: str = "Input",
) -> List[Path]:
    """
    Read a text file, split into segments of at most max_chars, avoiding
    splitting mid-sentence. Write each segment as a new .txt in the same
    directory with the same base name plus an index (e.g. baseInput1.txt,
    baseInput2.txt). Single sentences longer than max_chars are kept as
    one segment.

    Args:
        text_file_path: Path to the source text file.
        max_chars: Maximum characters per segment (default 380, empirical
            for Chatterbox). Configurable.
        output_stem_suffix: Suffix before the index (default "Input"), so
            "foo.txt" -> "fooInput1.txt", "fooInput2.txt".

    Returns:
        List of paths to the written segment files (1-based index in name).
    """
    path = Path(text_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    # Split at sentence boundaries (. ! ? followed by space)
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]

    if not sentences:
        # No sentence-ending punctuation: treat whole text as one segment
        segments = [text] if text else []
    else:
        segments = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            # +1 for space between sentences when current is non-empty
            sent_len = len(sentence) + (1 if current else 0)

            if current_len + sentence_len <= max_chars:
                current.append(sentence)
                current_len += sentence_len
            else:
                if current:
                    segments.append(" ".join(current))
                if len(sentence) > max_chars:
                    segments.append(sentence)
                    current = []
                    current_len = 0
                else:
                    current = [sentence]
                    current_len = len(sentence)

        if current:
            segments.append(" ".join(current))

    if not segments:
        return []

    parent = path.parent
    stem = path.stem
    written: List[Path] = []

    for i, segment in enumerate(segments, start=1):
        out_name = f"{stem}{output_stem_suffix}{i}.txt"
        out_path = parent / out_name
        out_path.write_text(segment, encoding="utf-8")
        written.append(out_path)

    return written