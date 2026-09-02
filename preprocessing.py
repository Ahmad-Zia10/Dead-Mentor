"""Clean raw source texts and tag them with structural citations.

Two jobs, both aimed at retrieval quality:

1. Project Gutenberg files wrap the real work in a license header and a long
   legal footer. Left in place they get embedded and retrieved, so a question
   about "rights" or "restrictions" pulls the license instead of the author.
2. A citation of "meditations.txt" is useless to a reader and invites the model
   to invent a passage number. Where the text has structure (books, chapters,
   letters) we capture it so the model can cite what it was actually given.
"""

import re

# Gutenberg wraps the work in these markers.
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF TH(?:E|IS) PROJECT GUTENBERG.*?\*\*\*", re.I)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF TH(?:E|IS) PROJECT GUTENBERG.*?\*\*\*", re.I)

# Marcus: "THE FIRST BOOK" .. "THE TWELFTH BOOK" head the actual meditations.
MARCUS_BOOK = re.compile(
    r"^THE (FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH"
    r"|ELEVENTH|TWELFTH) BOOK\s*$",
    re.M,
)
# An appendix, translator's notes and a glossary trail the twelfth book. None
# are Marcus's words -- the appendix in particular is an essay *about* him,
# written in the third person, which would badly pollute a first-person persona.
MARCUS_BACK_MATTER = re.compile(r"^(APPENDIX|NOTES|GLOSSARY)\s*$", re.M)

# Darwin: "CHAPTER I." or "CHAPTER 4. NATURAL SELECTION."
DARWIN_CHAPTER = re.compile(r"^CHAPTER\s+([IVXL]+|\d+)\.?\s*(.*?)\s*$", re.M)

# Feynman: "=== CHAPTER 1: ATOMS IN MOTION ===", "==== LETTER TO ARLINE, 1945 ====".
# The number of '=' varies between markers, so match a run of them.
FEYNMAN_MARKER = re.compile(r"^={3,}\s*(.+?)\s*={3,}\s*$", re.M)

# Darwin's autobiography has no CHAPTER headings; its divisions are the
# all-caps date-range lines that also make up its contents list.
AUTOBIO_SECTION = re.compile(
    r"^(CAMBRIDGE 1828-1831\.|.*VOYAGE OF THE .BEAGLE.*|FROM MY RETURN TO ENGLAND.*"
    r"|FROM MY MARRIAGE.*|RESIDENCE AT DOWN.*|MY SEVERAL PUBLICATIONS\."
    r"|WRITTEN MAY 1ST, 1881\.)\s*$",
    re.M,
)

# Human-readable work titles for citations, keyed by source filename.
WORK_TITLES = {
    "meditations.txt": "Meditations",
    "origin_of_species.txt": "On the Origin of Species",
    "autobiography.txt": "Autobiography",
    "feynman_lectures.txt": "The Feynman Lectures",
    "feynman_letters.txt": "Feynman Letters",
}

ORDINAL_TO_ROMAN = {
    "FIRST": "I", "SECOND": "II", "THIRD": "III", "FOURTH": "IV",
    "FIFTH": "V", "SIXTH": "VI", "SEVENTH": "VII", "EIGHTH": "VIII",
    "NINTH": "IX", "TENTH": "X", "ELEVENTH": "XI", "TWELFTH": "XII",
}


MAX_LABEL_CHARS = 60

# Roman numerals must survive title-casing ("IV" must not become "Iv").
ROMAN_NUMERAL = re.compile(r"[IVXLCDM]+[.:,]?")


def tidy_label(raw_label: str) -> str:
    """Normalise a heading into a short, readable citation label."""
    label = re.sub(r"\s+", " ", raw_label).strip()
    label = label.replace("_", "").strip()
    label = label.strip("“”‘’\"'").strip()
    label = label.rstrip(".,—- ").strip()
    if len(label) > MAX_LABEL_CHARS:
        label = label[:MAX_LABEL_CHARS].rsplit(" ", 1)[0] + "..."
    # Headings are typically all-caps in these sources; title-case them unless
    # the heading already carries its own mixed casing. A prefix we added
    # ourselves ("Chapter IV: ...") is ignored when judging that, so the
    # shouted half still gets tidied.
    head, sep, tail = label.partition(": ")
    if sep:
        # Title-case each half on its own, so a shouted title is tidied even
        # when the prefix ("Chapter IV") is already mixed case -- and vice versa.
        label = f"{titlecase_if_shouted(head)}{sep}{titlecase_if_shouted(tail)}"
    else:
        label = titlecase_if_shouted(label)
    return label


def titlecase_if_shouted(text: str) -> str:
    """Title-case text only if it is entirely uppercase, preserving numerals."""
    letters = [c for c in text if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return text
    # Roman numerals and short all-caps tokens should stay as they are.
    return " ".join(
        word if ROMAN_NUMERAL.fullmatch(word) else word.title()
        for word in text.split(" ")
    )


def work_title(filename: str) -> str:
    """Human-readable title for a source file."""
    return WORK_TITLES.get(filename, filename)


def strip_gutenberg(text: str) -> str:
    """Return only the text between the Gutenberg start/end markers."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]

    end = GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]

    return text.strip()


def strip_front_matter(text: str, filename: str) -> str:
    """Drop tables of contents, editor prefaces and translator notes."""
    if filename == "meditations.txt":
        # Content starts at "THE FIRST BOOK"; before it is the contents list
        # and the translator's introduction.
        first_book = MARCUS_BOOK.search(text)
        if first_book:
            text = text[first_book.start():]
        back = MARCUS_BACK_MATTER.search(text)
        if back:
            text = text[: back.start()]

    elif filename == "origin_of_species.txt":
        # Early CHAPTER matches are the contents list; the body restarts with a
        # bare "CHAPTER I." on its own line. Use the last one as the start.
        bare = list(re.finditer(r"^CHAPTER\s+I\.\s*$", text, re.M))
        if bare:
            text = text[bare[-1].start():]
        # The alphabetical index at the end is page-number noise, not prose.
        index = re.search(r"^INDEX\.\s*$", text, re.M)
        if index:
            text = text[: index.start()]

    elif filename == "autobiography.txt":
        # Skip the title block and contents list; the narrative starts at the
        # editor's bracketed note.
        body = re.search(r"^\[My father", text, re.M)
        if body:
            text = text[body.start():]

    return text.strip()


def find_sections(text: str, filename: str) -> list[tuple[int, str]]:
    """Locate structural divisions as (offset, label) pairs, in order.

    An empty list means the file has no usable structure, in which case chunks
    are cited by work title alone.
    """
    sections: list[tuple[int, str]] = []

    if filename == "meditations.txt":
        for match in MARCUS_BOOK.finditer(text):
            roman = ORDINAL_TO_ROMAN[match.group(1)]
            sections.append((match.start(), f"Book {roman}"))

    elif filename == "autobiography.txt":
        for match in AUTOBIO_SECTION.finditer(text):
            sections.append((match.start(), tidy_label(match.group(1))))

    elif filename == "origin_of_species.txt":
        for match in DARWIN_CHAPTER.finditer(text):
            number, title = match.group(1), match.group(2)
            label = f"Chapter {number}"
            if title:
                label = tidy_label(f"{label}: {title}")
            sections.append((match.start(), label))

    elif filename in ("feynman_lectures.txt", "feynman_letters.txt"):
        for match in FEYNMAN_MARKER.finditer(text):
            sections.append((match.start(), tidy_label(match.group(1))))

    return sections


def citation_at(sections: list[tuple[int, str]], offset: int, title: str) -> str:
    """Build the citation for a chunk starting at `offset`.

    Picks the last section heading at or before the chunk. Falls back to the
    work title when the chunk sits before any heading.
    """
    label = None
    for start, section_label in sections:
        if start <= offset:
            label = section_label
        else:
            break

    return f"{title}, {label}" if label else title


def clean_text(text: str, filename: str) -> str:
    """Full cleaning pass for one source file."""
    text = strip_gutenberg(text)
    text = strip_front_matter(text, filename)
    # Collapse runs of blank lines so the splitter's paragraph separator is
    # a reliable boundary.
    return re.sub(r"\n{3,}", "\n\n", text)
