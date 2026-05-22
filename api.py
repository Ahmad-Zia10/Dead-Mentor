from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import ask_mentor
from config import MENTORS

app = FastAPI(
    title="Dead Mentor API",
    description="Converse with history's greatest minds",
    version="1.0.0"
)


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    mentor: str
    query: str
    history: list = []


class QueryResponse(BaseModel):
    response: str
    sources: list
    history: list


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

    result = ask_mentor(
        mentor_name=request.mentor,
        query=request.query,
        history=request.history
    )
    return result


@app.get("/health")
def health():
    return {"status": "healthy", "mentors_available": len(MENTORS)}