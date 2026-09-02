import json
import re
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from config import GROQ_API_KEY, GROQ_MODEL, PERSONAS_DIR
from rag import retrieve_context


# --- Limits ---
# History arrives from the client, so it is capped server side. One turn is a
# user message plus the assistant reply, hence the *2.
MAX_HISTORY_TURNS = 6
MAX_MESSAGE_CHARS = 4000

# Query rewriting: how much prior conversation to show the rewriter, and the
# ceiling on what it may return (a runaway answer must not become the query).
REWRITE_HISTORY_TURNS = 2
REWRITE_CONTEXT_CHARS = 400
MAX_REWRITTEN_QUERY_CHARS = 300

# Words and shapes that signal a query leaning on earlier context. Matching
# here only decides whether to spend a rewrite call, so a false positive costs
# latency and a false negative just leaves the old behaviour in place.
FOLLOWUP_MARKERS = frozenset({
    "it", "its", "that", "this", "those", "these", "they", "them", "their",
    "he", "him", "his", "she", "her", "hers", "then", "such",
    "else", "again", "instead", "elaborate", "expand",
})
FOLLOWUP_PHRASES = (
    "say more", "tell me more", "go on", "what about", "how about",
    "what do you mean", "explain that", "expand on", "like what",
    "for example", "such as", "and then", "keep going",
    # "why"/"how" are too common to treat as markers on their own -- they open
    # plenty of standalone questions -- but these framings only make sense as a
    # reply to something already said.
    "why do you say", "why is that", "why so", "how so", "says who",
    "what makes you say", "on what grounds",
)
# A very short question is almost always leaning on context ("Why?", "Such as?").
SHORT_QUERY_WORDS = 4


class MentorError(Exception):
    """Raised when a mentor query cannot be completed."""


# --- Message Definition ---
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# --- State Definition ---
class MentorState(BaseModel):
    mentor_name: str
    user_query: str
    # The query actually sent to the retriever. Equal to user_query for a
    # standalone question; rewritten to include context for a follow-up.
    search_query: str = ""
    # Set when the rewrite step failed and retrieval fell back to the raw query.
    rewrite_error: Optional[str] = None
    retrieved_chunks: list = []
    sources: list = []
    response: Optional[str] = None
    conversation_history: list[Message] = []


# --- Load persona config ---
def load_persona(mentor_name: str) -> dict:
    path = f"{PERSONAS_DIR}/{mentor_name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_followup(query: str) -> bool:
    """Guess whether a query depends on earlier turns to make sense.

    Deliberately a cheap heuristic rather than an LLM call: its only job is to
    decide whether the rewrite is worth paying for.
    """
    lowered = query.lower().strip()

    if any(phrase in lowered for phrase in FOLLOWUP_PHRASES):
        return True

    words = re.findall(r"[a-z']+", lowered)
    if not words:
        return False

    if len(words) <= SHORT_QUERY_WORDS:
        return True

    return bool(FOLLOWUP_MARKERS.intersection(words))


def trim_history(history: list[Message]) -> list[Message]:
    """Keep only the most recent turns, and clamp any oversized message."""
    recent = history[-(MAX_HISTORY_TURNS * 2):]
    return [
        Message(role=msg.role, content=msg.content[:MAX_MESSAGE_CHARS])
        for msg in recent
    ]


# --- Node 1: Resolve follow-ups into standalone search queries ---
def rewrite_node(state: MentorState) -> MentorState:
    """Fold conversation context into the query before retrieval.

    "Say more about that" embeds as a context-free string and retrieves noise,
    so a follow-up is rewritten into a standalone question first. Standalone
    questions skip the rewrite rather than paying for an extra LLM call.
    """
    state.search_query = state.user_query

    if not state.conversation_history or not is_followup(state.user_query):
        return state

    transcript = "\n".join(
        f"{'User' if m.role == 'user' else 'Mentor'}: {m.content[:REWRITE_CONTEXT_CHARS]}"
        for m in state.conversation_history[-REWRITE_HISTORY_TURNS * 2:]
    )

    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
        result = llm.invoke([
            SystemMessage(content=(
                "Rewrite the user's latest message as a standalone search query "
                "for a library of one author's writings.\n\n"
                "Resolve pronouns and references using the conversation, and keep "
                "the topic words that make the subject searchable.\n"
                "Write only the subject being asked about. Do not add the author's "
                "name, book titles, chapter or book numbers -- the search is "
                "already limited to that author's works, so those words only "
                "dilute the query.\n"
                "Reply with the query alone: no quotes, no preamble, no explanation."
            )),
            HumanMessage(content=(
                f"Conversation:\n{transcript}\n\n"
                f"Latest message: {state.user_query}\n\nStandalone query:"
            )),
        ])
        rewritten = result.content.strip().strip('"').strip()
    except Exception as e:
        # Rewriting is an optimisation, not a requirement. If it fails, fall
        # back to the raw query rather than failing the whole request. Record
        # the failure so an evaluation run cannot mistake a silently skipped
        # rewrite for a rewrite that genuinely changed nothing.
        state.rewrite_error = str(e)[:200]
        return state

    if rewritten and len(rewritten) <= MAX_REWRITTEN_QUERY_CHARS:
        state.search_query = rewritten

    return state


# --- Node 2: Retrieve relevant chunks ---
def retrieve_node(state: MentorState) -> MentorState:
    try:
        docs = retrieve_context(state.mentor_name, state.search_query or state.user_query)
    except Exception as e:
        raise MentorError(f"Retrieval failed: {e}") from e

    state.retrieved_chunks = [doc.page_content for doc in docs]
    # Prefer the structural citation ("Meditations, Book IV") captured at
    # ingestion; fall back to the filename for vectors ingested before that.
    state.sources = [
        doc.metadata.get("citation") or doc.metadata.get("source_file", "Unknown")
        for doc in docs
    ]
    return state


# --- Node 3: Generate grounded response ---
def generate_node(state: MentorState) -> MentorState:
    try:
        persona = load_persona(state.mentor_name)
    except (OSError, json.JSONDecodeError) as e:
        raise MentorError(f"Could not load persona '{state.mentor_name}': {e}") from e

    # Nothing cleared the relevance threshold. Answering anyway is how a
    # grounded system starts inventing, so decline in the mentor's voice
    # without spending an LLM call.
    if not state.retrieved_chunks:
        state.response = (
            f"I find nothing in my writings that speaks to this. "
            f"I would rather tell you so plainly than invent an answer."
        )
        state.conversation_history = state.conversation_history + [
            Message(role="user", content=state.user_query),
            Message(role="assistant", content=state.response),
        ]
        return state

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    # Label each passage with its citation so the model can only cite what it
    # was actually handed, instead of inventing a passage number to satisfy the
    # persona's "always cite" instruction.
    context = "\n\n".join(
        f"[{citation}]\n{chunk}"
        for citation, chunk in zip(state.sources, state.retrieved_chunks)
    )

    # Build conversation history messages. Past assistant turns are replayed as
    # AIMessage so they cannot act as extra system instructions.
    history_messages = [
        HumanMessage(content=msg.content) if msg.role == "user"
        else AIMessage(content=msg.content)
        for msg in state.conversation_history
    ]

    messages = [
        SystemMessage(content=f"""
{persona['system_prompt']}

Use ONLY the following source material to answer.
If the answer is not in the sources, say so honestly.

Each passage below is preceded by its citation in square brackets. When you
cite, use one of those citations exactly as written. Do not invent a book,
chapter, letter or passage number that does not appear there.
End your response with: Source: [citation]

SOURCE MATERIAL:
{context}
        """),
        *history_messages,
        HumanMessage(content=state.user_query)
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        raise MentorError(f"Language model call failed: {e}") from e

    state.response = response.content

    # Update conversation history
    state.conversation_history = state.conversation_history + [
        Message(role="user", content=state.user_query),
        Message(role="assistant", content=state.response),
    ]

    return state


# --- Build the Graph ---
def build_agent():
    graph = StateGraph(MentorState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


agent = build_agent()


# --- Main function to call from API ---
def ask_mentor(
    mentor_name: str,
    query: str,
    history: Optional[list[Message]] = None,
) -> dict:
    state = MentorState(
        mentor_name=mentor_name,
        user_query=query,
        conversation_history=trim_history(history or [])
    )
    result = agent.invoke(state)

    if not result["response"]:
        raise MentorError("The mentor returned an empty response.")

    return {
        "response": result["response"],
        "sources": result["sources"],
        "history": result["conversation_history"],
    }
