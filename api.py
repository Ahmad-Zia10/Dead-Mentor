import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import Message, MentorError, ask_mentor
from config import ALLOWED_ORIGINS, MENTORS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dead Mentor API",
    description="Converse with history's greatest minds",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    mentor: str
    query: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []


class QueryResponse(BaseModel):
    response: str
    sources: list
    history: list[Message]


# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "Dead Mentor API is running"}


@app.get("/mentors")
def list_mentors():
    return {"mentors": MENTORS}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    if request.mentor not in MENTORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mentor '{request.mentor}'. Choose from {MENTORS}"
        )
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        result = ask_mentor(
            mentor_name=request.mentor,
            query=request.query,
            history=request.history
        )
    except MentorError as e:
        # Upstream provider (Pinecone / Gemini / Groq) failed. Log the detail,
        # return a generic message so internals are not leaked to the client.
        logger.exception("Mentor query failed for '%s'", request.mentor)
        raise HTTPException(
            status_code=503,
            detail="The mentor is unavailable right now. Please try again."
        ) from e

    return result


@app.get("/health")
def health():
    return {"status": "healthy", "mentors_available": len(MENTORS)}
