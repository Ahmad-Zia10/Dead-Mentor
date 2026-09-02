"""Evaluation set for the Dead Mentor RAG pipeline.

Hand-written rather than generated, so the expectations are independent of what
the system currently does. Three kinds of case:

- grounded:  answerable from the source texts. Retrieval must find support, and
             the answer should contain the expected terms.
- refusal:   not in the source texts. The mentor must decline rather than
             improvise; this is the promise the README makes.
- followup:  a multi-turn case where the final question only makes sense in
             context. These are what query rewriting exists to fix.
"""

from dataclasses import dataclass, field


@dataclass
class Case:
    id: str
    mentor: str
    query: str
    kind: str  # "grounded" | "refusal" | "followup"
    # Terms that should appear in a correct answer (case-insensitive, any-of
    # within each group, all groups must hit).
    expect_terms: list[list[str]] = field(default_factory=list)
    # Section labels the retrieved citations should include, if it matters.
    expect_citation: str | None = None
    # Prior turns, for followup cases.
    history: list[dict] = field(default_factory=list)


GROUNDED = [
    Case("marcus-anger", "marcus", "How should I deal with anger?", "grounded",
         [["anger", "angry", "wrath"]]),
    Case("marcus-death", "marcus", "What should I think about death?", "grounded",
         [["death", "die", "mortal"]]),
    Case("marcus-others", "marcus", "How should I treat people who wrong me?", "grounded",
         [["wrong", "offend", "trespass", "forgive", "harm"]]),
    Case("marcus-control", "marcus", "What is within my control?", "grounded",
         [["control", "power", "opinion", "mind"]]),
    Case("marcus-teachers", "marcus", "What did you learn from your teachers?", "grounded",
         [["learn", "taught", "grandfather", "father"]]),

    Case("feynman-learn", "feynman", "How should I learn physics?", "grounded",
         [["physics", "learn", "study"]]),
    Case("feynman-atoms", "feynman", "What is the atomic hypothesis?", "grounded",
         [["atom", "particle", "molecule"]]),
    Case("feynman-least-time", "feynman", "Explain the principle of least time.", "grounded",
         [["light", "time", "path"]]),
    Case("feynman-algebra", "feynman", "Why does algebra matter in physics?", "grounded",
         [["algebra", "equation", "mathematic"]]),
    Case("feynman-arline", "feynman", "What did you write to your wife?", "grounded",
         [["love", "arline", "wife", "darling"]]),

    Case("darwin-selection", "darwin", "How does natural selection work?", "grounded",
         [["selection", "variation", "survive", "advantage"]]),
    Case("darwin-struggle", "darwin", "What is the struggle for existence?", "grounded",
         [["struggle", "existence", "competition", "survive"]]),
    Case("darwin-doubt", "darwin", "How did you handle doubts about your theory?", "grounded",
         [["doubt", "difficult", "objection", "theory"]]),
    Case("darwin-beagle", "darwin", "What did you learn on the Beagle voyage?", "grounded",
         [["beagle", "voyage", "geolog", "observ"]]),
    Case("darwin-instinct", "darwin", "What did you observe about instinct?", "grounded",
         [["instinct", "habit", "bee", "animal"]]),
]

# Plainly outside every corpus. The mentor must decline.
REFUSAL = [
    Case("refuse-js", "marcus", "What is the best JavaScript framework?", "refusal"),
    Case("refuse-k8s", "marcus", "How do I set up a Kubernetes cluster?", "refusal"),
    Case("refuse-btc", "feynman", "What is the price of Bitcoin today?", "refusal"),
    Case("refuse-tiktok", "feynman", "How do I go viral on TikTok?", "refusal"),
    Case("refuse-pizza", "darwin", "What is the best pizza place in Chicago?", "refusal"),
    Case("refuse-npe", "darwin", "How do I fix a NullPointerException in Java?", "refusal"),
]

# The last query is meaningless without the prior turn. Without rewriting these
# retrieve on the literal words ("say more about that") and find nothing useful.
FOLLOWUP = [
    Case("follow-marcus-anger", "marcus", "Say more about that.", "followup",
         [["anger", "angry", "wrath", "offend"]],
         history=[
             {"role": "user", "content": "How should I deal with anger?"},
             {"role": "assistant", "content":
              "When anger rises, pause and consider that the offender acts from "
              "ignorance. Source: Meditations, Book V"},
         ]),
    Case("follow-marcus-death", "marcus", "Why do you say so?", "followup",
         [["death", "die", "nature", "mortal"]],
         history=[
             {"role": "user", "content": "What should I think about death?"},
             {"role": "assistant", "content":
              "Death is a natural process and nothing to fear. "
              "Source: Meditations, Book II"},
         ]),
    Case("follow-feynman-learn", "feynman", "Can you expand on that?", "followup",
         [["physics", "learn", "understand", "study"]],
         history=[
             {"role": "user", "content": "How should I learn physics?"},
             {"role": "assistant", "content":
              "You learn physics by doing it, not by memorising laws. "
              "Source: The Feynman Lectures, Chapter 1"},
         ]),
    Case("follow-feynman-atoms", "feynman", "What else follows from it?", "followup",
         [["atom", "molecule", "matter", "particle"]],
         history=[
             {"role": "user", "content": "What is the atomic hypothesis?"},
             {"role": "assistant", "content":
              "All things are made of atoms in perpetual motion. "
              "Source: The Feynman Lectures, Chapter 1"},
         ]),
    Case("follow-darwin-selection", "darwin", "How does that happen in nature?", "followup",
         [["selection", "nature", "variation", "species"]],
         history=[
             {"role": "user", "content": "How does natural selection work?"},
             {"role": "assistant", "content":
              "Favourable variations tend to be preserved. "
              "Source: On the Origin of Species, Chapter IV"},
         ]),
]

ALL_CASES = GROUNDED + REFUSAL + FOLLOWUP


def cases_for(mentors: list[str] | None = None, kinds: list[str] | None = None) -> list[Case]:
    """Filter the set, so a partially ingested corpus can still be evaluated."""
    selected = ALL_CASES
    if mentors:
        selected = [c for c in selected if c.mentor in mentors]
    if kinds:
        selected = [c for c in selected if c.kind in kinds]
    return selected
