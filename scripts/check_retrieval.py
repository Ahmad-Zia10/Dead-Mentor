"""Inspect raw retrieval scores, to tune RETRIEVAL_SCORE_THRESHOLD.

The threshold decides when a mentor says "that is not in my writings" instead of
answering from weak matches. Run this with on-topic and deliberately off-topic
questions and pick a value that separates them.

    python scripts/check_retrieval.py
    python scripts/check_retrieval.py marcus "how do I deal with anger?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RETRIEVAL_SCORE_THRESHOLD  # noqa: E402
from rag import retrieve_with_scores  # noqa: E402

# On-topic questions should score well above the threshold; the off-topic ones
# below it. If they overlap, the threshold cannot separate them.
PROBES = [
    ("marcus", "How should I deal with anger?", True),
    ("marcus", "What is virtue?", True),
    ("marcus", "What is the best JavaScript framework?", False),
    ("marcus", "How do I set up a Kubernetes cluster?", False),
    ("feynman", "How should I learn physics?", True),
    ("feynman", "What is the principle of least time?", True),
    ("feynman", "What are your thoughts on TikTok marketing?", False),
    ("darwin", "How does natural selection work?", True),
    ("darwin", "How did you handle doubt about your work?", True),
    ("darwin", "What is the price of Bitcoin?", False),
]


def report(mentor: str, query: str, on_topic: bool | None = None):
    scored = retrieve_with_scores(mentor, query)
    scores = [score for _, score in scored]
    top = max(scores) if scores else 0.0
    kept = sum(1 for s in scores if s >= RETRIEVAL_SCORE_THRESHOLD)

    flag = ""
    if on_topic is True and kept == 0:
        flag = "  <-- MISS: on-topic question rejected"
    elif on_topic is False and kept > 0:
        flag = "  <-- LEAK: off-topic question answered"

    label = {True: "on ", False: "off", None: "   "}[on_topic]
    print(f"[{label}] {mentor:<8} top={top:.3f} kept={kept}/{len(scores)}  {query}{flag}")
    return scored


def main():
    if len(sys.argv) >= 3:
        mentor, query = sys.argv[1], " ".join(sys.argv[2:])
        scored = report(mentor, query)
        for doc, score in scored:
            citation = doc.metadata.get("citation") or doc.metadata.get("source_file")
            print(f"    {score:.3f}  [{citation}]  {doc.page_content[:90]!r}")
        return

    print(f"threshold = {RETRIEVAL_SCORE_THRESHOLD}\n")
    for mentor, query, on_topic in PROBES:
        report(mentor, query, on_topic)


if __name__ == "__main__":
    main()
