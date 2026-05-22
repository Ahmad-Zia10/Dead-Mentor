import json
from typing import Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from config import GROQ_API_KEY, GROQ_MODEL, PERSONAS_DIR
from rag import retrieve_context


# --- State Definition ---
class MentorState(BaseModel):
    mentor_name: str
    user_query: str
    retrieved_chunks: list = []
    sources: list = []
    response: Optional[str] = None
    conversation_history: list = []


# --- Load persona config ---
def load_persona(mentor_name: str) -> dict:
    path = f"{PERSONAS_DIR}/{mentor_name}.json"
    with open(path) as f:
        return json.load(f)


# --- Node 1: Retrieve relevant chunks ---
def retrieve_node(state: MentorState) -> MentorState:
    docs = retrieve_context(state.mentor_name, state.user_query)
    state.retrieved_chunks = [doc.page_content for doc in docs]
    state.sources = [doc.metadata.get("source_file", "Unknown") for doc in docs]
    return state


# --- Node 2: Generate grounded response ---
def generate_node(state: MentorState) -> MentorState:
    persona = load_persona(state.mentor_name)
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

    context = "\n\n".join(state.retrieved_chunks)

    # Build conversation history messages
    history_messages = []
    for msg in state.conversation_history:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(SystemMessage(content=msg["content"]))

    messages = [
        SystemMessage(content=f"""
{persona['system_prompt']}

Use ONLY the following source material to answer.
If the answer is not in the sources, say so honestly.
Always end your response with: Source: [book or letter name]

SOURCE MATERIAL:
{context}
        """),
        *history_messages,
        HumanMessage(content=state.user_query)
    ]

    response = llm.invoke(messages)
    state.response = response.content

    # Update conversation history
    state.conversation_history.append({
        "role": "user",
        "content": state.user_query
    })
    state.conversation_history.append({
        "role": "assistant",
        "content": state.response
    })

    return state


# --- Build the Graph ---
def build_agent():
    graph = StateGraph(MentorState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


agent = build_agent()


# --- Main function to call from API ---
def ask_mentor(mentor_name: str, query: str, history: list = []) -> dict:
    state = MentorState(
        mentor_name=mentor_name,
        user_query=query,
        conversation_history=history
    )
    result = agent.invoke(state)
    return {
        "response": result["response"],
        "sources": result["sources"],
        "history": result["conversation_history"]
    }



