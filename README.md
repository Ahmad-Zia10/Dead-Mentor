# 🏛️ Dead Mentor

> *Converse with history's greatest minds — grounded in their actual words.*

Dead Mentor is an AI-powered conversational platform that lets you have real, meaningful dialogues with history's greatest thinkers. Ask Marcus Aurelius how to handle rejection, have Feynman break down a concept you can't grasp, or ask Darwin how he thought about uncertainty and iteration.

Every response is strictly grounded in RAG (Retrieval Augmented Generation) over each mentor's actual source texts — their books, letters, and documented writings. The AI retrieves relevant passages first, then constructs a response using only what that person actually wrote or said. Every answer comes with a cited source. If the source texts don't contain an answer, the mentor says so honestly rather than fabricating one.

---

## 🧠 Mentors

| Mentor | Era | Sources |
|---|---|---|
| Marcus Aurelius | 121–180 AD | Meditations |
| Richard Feynman | 1918–1988 | The Feynman Lectures, Letters |
| Charles Darwin | 1809–1882 | On the Origin of Species, Autobiography |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | Google Gemini (gemini-embedding-001) |
| Vector Database | Pinecone |
| Agent Framework | LangGraph |
| RAG Pipeline | LangChain |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Railway |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI
    │
    ▼
FastAPI Backend (/ask endpoint)
    │
    ▼
LangGraph Agent
    ├── Retrieve Node → Pinecone Vector Store (namespace per mentor)
    │       └── Gemini Embeddings (semantic search)
    └── Generate Node → Groq LLM
            ├── Persona Config (speaking style, constraints)
            ├── Retrieved Chunks (source material)
            └── Conversation History (multi-turn memory)
    │
    ▼
Cited Response
```

---

## 📁 Project Structure

```
dead-mentor/
├── data/
│   └── raw/
│       ├── marcus/          # Meditations
│       ├── feynman/         # Lectures and Letters
│       └── darwin/          # Origin of Species, Autobiography
├── personas/
│   ├── marcus.json          # Persona config and system prompt
│   ├── feynman.json
│   └── darwin.json
├── logs/
├── ingestion.py             # Document loading, chunking, embedding pipeline
├── rag.py                   # Pinecone retrieval layer
├── agent.py                 # LangGraph agent with retrieve and generate nodes
├── api.py                   # FastAPI backend
├── ui.py                    # Streamlit frontend
├── config.py                # Centralized configuration and env validation
├── healthcheck.py           # API key validation utility
├── requirements.txt
├── .env                     # API keys (never committed)
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Ahmad-Zia10/dead-mentor.git
cd dead-mentor
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=dead-mentor
```

Get your API keys from:
- **Groq** → [console.groq.com](https://console.groq.com)
- **Gemini** → [aistudio.google.com](https://aistudio.google.com)
- **Pinecone** → [app.pinecone.io](https://app.pinecone.io)

### 5. Create Pinecone Index

On [app.pinecone.io](https://app.pinecone.io):
- Create a new index named `dead-mentor`
- Dimensions: `3072`
- Metric: `cosine`

### 6. Run health check

```bash
python healthcheck.py
```

All three services should show ✅ before proceeding.

### 7. Run ingestion pipeline

```bash
python ingestion.py
```

> ⚠️ **Note:** Gemini free tier allows 1000 embedding requests per day. With ~4800 total chunks across all mentors, ingestion may need to be run across multiple days or accounts. The pipeline supports resuming from any batch — see `start_batch` parameter in `ingestion.py`.

### 8. Run the application

In terminal 1 — start the FastAPI backend:
```bash
uvicorn api:app --reload
```

In terminal 2 — start the Streamlit frontend:
```bash
streamlit run ui.py
```

Open **http://localhost:8501** in your browser.

---

## 🔌 API Reference

### `GET /`
Health check — confirms API is running.

### `GET /mentors`
Returns list of available mentors.

**Response:**
```json
{
  "mentors": ["marcus", "feynman", "darwin"]
}
```

### `POST /ask`
Ask a mentor a question.

**Request:**
```json
{
  "mentor": "marcus",
  "query": "How do I deal with rejection?",
  "history": []
}
```

**Response:**
```json
{
  "response": "...",
  "sources": ["meditations.txt"],
  "history": [...]
}
```

### `GET /health`
Returns health status and mentor count.

---

## 🧪 Running Health Check

```bash
python healthcheck.py
```

Expected output:
```
Running health checks...

✅ Groq working — response: Hello
✅ Gemini embeddings working — vector length: 3072
✅ Pinecone working — index: dead-mentor, vectors: 4867

Health check complete.
```

---

## 💡 How It Works

### 1. Ingestion (One-time)
Source texts are loaded, split into 500-character chunks, embedded using Gemini's `gemini-embedding-001` model, and stored in Pinecone with a separate namespace per mentor.

### 2. Retrieval
When a user asks a question, it is embedded using the same Gemini model. Pinecone performs a cosine similarity search within the mentor's namespace and returns the top 5 most semantically relevant chunks.

### 3. Generation
The retrieved chunks are passed as context to the Groq LLM along with the mentor's persona config (speaking style, constraints, system prompt) and full conversation history. The LLM generates a response grounded strictly in the retrieved source material.

### 4. Citation
Every response includes a source citation. The mentor is instructed never to fabricate opinions or quotes not found in the source texts.

---

## ➕ Adding a New Mentor

1. Create a folder: `data/raw/<mentor_name>/`
2. Add source texts as `.txt` or `.pdf` files
3. Create `personas/<mentor_name>.json` with name, era, sources, speaking style and system prompt
4. Add mentor name to `MENTORS` list in `config.py`
5. Run `ingestion.py` for the new mentor only:
```python
ingest_mentor("<mentor_name>", start_batch=0)
```

---

## 🚀 Deployment

This project is deployed on Railway. See deployment instructions in the deployment guide.

Environment variables required on Railway:
```
GROQ_API_KEY
GEMINI_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

---

## 🔮 Roadmap

- [ ] Add more mentors (Nietzsche, Tesla, Einstein, Cleopatra)
- [ ] React frontend with museum-themed UI
- [ ] Voice mode — speak to your mentor
- [ ] Cross-mentor debates — ask two mentors the same question
- [ ] Mobile app

---

## 👤 Author

**Ahmad Zia**
- GitHub: [@Ahmad-Zia10](https://github.com/Ahmad-Zia10)
- LinkedIn: [linkedin.com/in/ahmad-zia](https://linkedin.com/in/ahmad-zia)

---

## 📄 License

MIT License — feel free to fork, extend and build on this project.

---

*"You have power over your mind, not outside events. Realize this, and you will find strength." — Marcus Aurelius*