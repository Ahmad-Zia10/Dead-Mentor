"""Run the evaluation set and report retrieval and answer quality.

Without this there is no way to tell whether a chunking, threshold or prompt
change helped or hurt. Metrics reported:

  retrieval@k   did retrieval return anything above the threshold
  groundedness  is the answer's wording actually present in the retrieved
                chunks (a rough check for invented content)
  citation      did the answer cite only citations it was handed
  keyword       does the answer contain the terms a correct answer needs
  refusal       did an unanswerable question get declined

Usage:
    python evals/run_eval.py                        # every case
    python evals/run_eval.py --mentors marcus feynman
    python evals/run_eval.py --kinds followup       # just the multi-turn cases
    python evals/run_eval.py --retrieval-only       # no LLM calls
    python evals/run_eval.py --no-rewrite           # A/B the rewrite step
    python evals/run_eval.py --json out.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from agent import Message, ask_mentor  # noqa: E402
from evals.dataset import cases_for  # noqa: E402
from rag import retrieve_with_scores  # noqa: E402
from config import RETRIEVAL_SCORE_THRESHOLD  # noqa: E402

REFUSAL_MARKERS = ("nothing in my writings", "not in the source", "cannot find")

# The free embedding tier rate-limits short bursts, and the eval fires one call
# per case back to back. Retry those rather than reporting them as failures --
# a rate limit says nothing about retrieval quality.
RATE_LIMIT_MARKERS = (
    "resource_exhausted", "429", "rate limit", "quota",
    # The provider occasionally returns nothing under load, which the agent
    # surfaces as its own error. Transient in the same way, so retry it too.
    "empty response",
)
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 20


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def with_retry(fn, *args, **kwargs):
    """Call fn, backing off on rate limits. Other errors propagate at once."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_rate_limit(e) or attempt == MAX_ATTEMPTS:
                raise
            wait = BACKOFF_SECONDS * attempt
            print(f"    rate limited, retrying in {wait}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS - 1})")
            time.sleep(wait)


STOPWORDS = frozenset({
    "the", "and", "that", "this", "with", "your", "you", "for", "are", "but",
    "not", "from", "have", "has", "was", "were", "what", "which", "who", "how",
    "why", "when", "his", "her", "their", "them", "they", "there", "then",
    "would", "could", "should", "about", "into", "than", "such", "these",
    "those", "will", "can", "its", "it's", "our", "out", "one", "all", "any",
    "may", "more", "most", "some", "own", "same", "too", "very", "just",
    "source", "meditations", "chapter", "book", "letter", "lectures",
})


def content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{4,}", text.lower())
        if w not in STOPWORDS
    }


def groundedness(answer: str, chunks: list[str]) -> float:
    """Share of the answer's content words that appear in the retrieved text.

    Crude but useful: a confidently invented passage scores low because its
    distinctive words are absent from the sources. Paraphrase and the mentor's
    own framing drag it down too, so read it as a relative signal across runs,
    not an absolute truth score.
    """
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    source_words = content_words(" ".join(chunks))
    return len(answer_words & source_words) / len(answer_words)


def cited_sources(answer: str) -> list[str]:
    """Pull the citations an answer claims, from [brackets] or a Source: line."""
    cites = re.findall(r"\[([^\]]{3,80})\]", answer)
    for line in re.findall(r"Source:\s*(.+)", answer):
        cites.append(line.strip().strip("[]").strip())
    return [c.strip() for c in cites if c.strip()]


def normalise_citation(text: str) -> str:
    """Fold a citation to a comparable form.

    Models reproduce a citation faithfully but reformat its whitespace -- the
    Feynman chapter headings come back with a narrow no-break space (U+202F)
    where the source had a plain one. Comparing raw strings reports those as
    fabricated citations, so collapse all whitespace and punctuation spacing
    before matching.
    """
    folded = re.sub(r"\s+", " ", text.replace(" ", " ").replace("\xa0", " "))
    # The Feynman headings carry a space before the colon ("Chapter 2 : Basics")
    # which models tidy up when quoting. Spacing around punctuation is never
    # the difference between a real citation and an invented one.
    folded = re.sub(r"\s*([:,])\s*", r"\1", folded)
    return folded.lower().strip(" .,:;[]")


def citation_ok(answer: str, sources: list[str]) -> bool | None:
    """True if every claimed citation matches something retrieved.

    None means the answer made no citation claim, so there is nothing to check.
    """
    claimed = cited_sources(answer)
    if not claimed:
        return None

    supplied = [normalise_citation(s) for s in sources]
    for claim in claimed:
        c = normalise_citation(claim)
        # A citation counts as supported if it is a substring of a supplied one
        # or vice versa -- models often shorten "Meditations, Book V" to "Book V".
        if not any(c in s or s in c for s in supplied):
            return False
    return True


def keywords_hit(answer: str, expect_terms: list[list[str]]) -> bool:
    lowered = answer.lower()
    return all(
        any(term.lower() in lowered for term in group)
        for group in expect_terms
    )


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def evaluate(case, retrieval_only: bool) -> dict:
    history = [Message(**m) for m in case.history]
    result = {"id": case.id, "mentor": case.mentor, "kind": case.kind}
    started = time.time()

    # Mirror the agent's own path so the eval measures what production does,
    # including query rewriting for follow-ups.
    state = agent.MentorState(
        mentor_name=case.mentor,
        user_query=case.query,
        conversation_history=agent.trim_history(history),
    )
    state = agent.rewrite_node(state)
    result["search_query"] = state.search_query
    result["rewritten"] = state.search_query != case.query
    if state.rewrite_error:
        # A rewrite that never ran is not evidence that rewriting does not help.
        result["rewrite_error"] = state.rewrite_error

    scored = with_retry(retrieve_with_scores, case.mentor, state.search_query)
    kept = [(d, s) for d, s in scored if s >= RETRIEVAL_SCORE_THRESHOLD]
    result["top_score"] = round(max((s for _, s in scored), default=0.0), 3)
    result["kept"] = len(kept)
    result["retrieved"] = bool(kept)

    if retrieval_only:
        result["elapsed"] = round(time.time() - started, 2)
        # Retrieval alone decides the refusal cases and whether a followup found
        # anything; keyword and groundedness need the generated answer.
        result["pass"] = (not kept) if case.kind == "refusal" else bool(kept)
        return result

    answer = with_retry(ask_mentor, case.mentor, case.query, history=history)
    text = answer["response"]
    result["elapsed"] = round(time.time() - started, 2)
    result["sources"] = sorted(set(answer["sources"]))
    result["refused"] = is_refusal(text)
    # Keep the whole answer: a truncated one cannot be re-scored offline, which
    # is exactly what you want when a citation check fails and you need to see
    # what was actually claimed.
    result["answer"] = text

    if case.kind == "refusal":
        result["pass"] = result["refused"]
        return result

    chunks = [d.page_content for d, _ in kept]
    result["groundedness"] = round(groundedness(text, chunks), 3)
    result["citation_ok"] = citation_ok(text, answer["sources"])
    result["keywords"] = keywords_hit(text, case.expect_terms) if case.expect_terms else None

    result["pass"] = bool(
        result["retrieved"]
        and not result["refused"]
        and result["citation_ok"] is not False
        and result["keywords"] is not False
    )
    return result


def summarise(results: list[dict]):
    errored = [r for r in results if r.get("error")]
    results = [r for r in results if not r.get("error")]

    if errored:
        print(f"\n{len(errored)} case(s) could not run:")
        for r in errored:
            print(f"  {r['id']:<26} {r['error'][:100]}")
        print("  These are excluded from the scores below.")

    if not results:
        print("\nNo cases completed -- nothing to score.")
        return

    by_kind: dict[str, list[dict]] = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r)

    print("\n" + "=" * 72)
    print(f"{'kind':<12}{'pass':>10}{'retrieved':>12}{'grounded':>11}{'cite ok':>10}")
    print("-" * 72)
    for kind, group in sorted(by_kind.items()):
        passed = sum(1 for r in group if r["pass"])
        retrieved = sum(1 for r in group if r.get("retrieved"))
        grounded = [r["groundedness"] for r in group if "groundedness" in r]
        cites = [r["citation_ok"] for r in group if r.get("citation_ok") is not None]
        print(
            f"{kind:<12}{passed:>4}/{len(group):<5}{retrieved:>8}/{len(group):<3}"
            f"{(sum(grounded)/len(grounded) if grounded else 0):>11.2f}"
            f"{(sum(cites)):>7}/{len(cites) if cites else 0:<3}"
        )
    print("-" * 72)
    total = sum(1 for r in results if r["pass"])
    print(f"{'TOTAL':<12}{total:>4}/{len(results)}")

    failures = [r for r in results if not r["pass"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            reason = []
            if not r.get("retrieved"):
                reason.append(f"no chunks above threshold (top={r.get('top_score')})")
            if r.get("refused") and r["kind"] != "refusal":
                reason.append("refused an answerable question")
            if r["kind"] == "refusal" and not r.get("refused"):
                reason.append("answered an unanswerable question")
            if r.get("citation_ok") is False:
                reason.append("cited a source it was not given")
            if r.get("keywords") is False:
                reason.append("missing expected terms")
            print(f"  {r['id']:<26} {'; '.join(reason) or 'unknown'}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--mentors", nargs="*", help="Limit to these mentors")
    parser.add_argument("--kinds", nargs="*", help="Limit to grounded/refusal/followup")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip generation; no LLM calls")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="Disable query rewriting, to A/B its effect")
    parser.add_argument("--json", help="Write full results to this file")
    parser.add_argument("--delay", type=float, default=6.0,
                        help="Seconds to pause between cases, to stay under the "
                             "embedding tier's per-minute limit (default: 6)")
    args = parser.parse_args()

    if args.no_rewrite:
        # Neutralise the rewrite step so the same cases run against raw queries.
        agent.is_followup = lambda query: False

    cases = cases_for(args.mentors, args.kinds)
    if not cases:
        parser.error("No cases matched those filters.")

    print(f"Running {len(cases)} cases "
          f"(threshold={RETRIEVAL_SCORE_THRESHOLD}, "
          f"rewrite={'off' if args.no_rewrite else 'on'})\n")

    results = []
    for index, case in enumerate(cases):
        if index and args.delay:
            time.sleep(args.delay)
        try:
            r = evaluate(case, args.retrieval_only)
        except Exception as e:
            # An infrastructure failure (quota, network) is not an eval result.
            # Marking it as a plain failure would hide it among real regressions
            # -- and would let a refusal case "pass" simply because nothing was
            # retrieved.
            r = {"id": case.id, "mentor": case.mentor, "kind": case.kind,
                 "pass": False, "error": str(e)[:200]}
        results.append(r)

        mark = "ERR " if r.get("error") else ("PASS" if r["pass"] else "FAIL")
        extra = ""
        if r.get("rewritten"):
            extra = f"  rewritten -> {r['search_query'][:60]!r}"
        elif "error" in r:
            extra = f"  ERROR: {r['error'][:80]}"
        print(f"[{mark}] {r['id']:<26} top={r.get('top_score', 0):<6} "
              f"kept={r.get('kept', 0)}{extra}")

    summarise(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    if any(r.get("error") for r in results):
        return 2  # infrastructure problem, results are incomplete
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
