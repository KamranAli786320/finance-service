# 💰 AI Personal Finance Assistant

An AI-powered Personal Finance Assistant built with **FastAPI**, **LangGraph**, **RAG pipeline**, and **Groq LLaMA 3.3**. Users can interact with their financial data conversationally using natural language.

---

## 🏗️ System Architecture

```
User (Chat UI)
      │
      ▼
FastAPI Backend (app/main.py)
      │
      ├──▶ POST /chat ──▶ LangGraph Agent
      │                        │
      │              ┌─────────┴──────────┐
      │              ▼                    ▼
      │       Intent: transactions    Intent: rag
      │              │                    │
      │       Mock Banking API      RAG Pipeline
      │       (transactions.json)   (TF-IDF + Cosine)
      │              │                    │
      │              └─────────┬──────────┘
      │                        ▼
      │               Groq LLaMA 3.3 (Response Generation)
      │
      ├──▶ GET /api/transactions  (Mock Banking API)
      ├──▶ GET /api/insights      (Spending Analysis)
      └──▶ GET /health            (Health Check)

n8n Workflow (Orchestration Layer)
      │
      └──▶ Webhook → Intent Router → FastAPI → Response
```

---

## ✨ Features

- 💬 **Natural Language Chat** — Ask questions in plain English
- 🤖 **LangGraph Agent** — Intelligently routes between transactions and RAG (no hardcoded logic)
- 📊 **Mock Banking API** — 25 simulated transactions with categories, amounts, timestamps
- 📈 **Financial Insights** — Weekly/monthly summaries, top categories, week-over-week comparison
- 📚 **RAG Pipeline** — TF-IDF embeddings + cosine similarity retrieval from financial documents
- 🧠 **Multi-turn Memory** — Session-based conversation history
- 🎨 **Clean Chat UI** — Dark theme with sidebar quick queries and live stats
- 🔄 **n8n Workflow** — Exported workflow for orchestration automation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| LLM | Groq LLaMA 3.3 70B Versatile |
| Agent | LangGraph |
| RAG | TF-IDF + Cosine Similarity (local, no external vector DB) |
| Memory | In-memory session store |
| Workflow | n8n |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |

---

## 📁 Project Structure

```
finance-assistant/
├── run.py                          # Entry point → uvicorn
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md
├── app/
│   ├── main.py                     # FastAPI app + CORS + static serving
│   ├── agent.py                    # LangGraph graph (intent_router + executor nodes)
│   ├── rag.py                      # RAG pipeline (chunking, TF-IDF, retrieval)
│   ├── memory.py                   # In-memory multi-turn session store
│   └── routers/
│       ├── chat.py                 # POST /chat endpoint
│       └── banking.py              # GET /api/transactions + /api/insights
├── data/
│   ├── transactions.json           # 25 mock transactions
│   └── docs/
│       ├── budgeting_strategies.md
│       ├── saving_techniques.md
│       └── financial_literacy.md
├── app/static/
│   └── index.html                  # Chat UI
└── n8n/
    └── finance-assistant-workflow.json
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Groq API Key → [console.groq.com](https://console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/KamranAli786320/finance-service.git
cd finance-service
```

### 2. Create virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```
Open `.env` and set your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
APP_HOST=0.0.0.0
APP_PORT=8000
```

### 5. Run the server
```bash
python run.py
```

### 6. Open the app
- **Chat UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 🤖 How the Agent Works

The LangGraph graph has **2 nodes**:

1. **`intent_router`** — LLM reads the user message and returns `transactions` or `rag` (no hardcoded if/else rules)
2. **`executor`** — Based on intent:
   - `transactions` → fetches mock banking data, calculates insights, generates response
   - `rag` → retrieves relevant document chunks via cosine similarity, generates grounded response

```
User Message
     │
     ▼
intent_router (LLM decides)
     │
     ├── "transactions" ──▶ Fetch API Data ──▶ LLM Response
     └── "rag"          ──▶ Retrieve Docs  ──▶ LLM Response
```

---

## 📚 RAG Pipeline

The RAG system uses **local TF-IDF embeddings** — no external vector database or OpenAI embeddings required.

**Process:**
1. **Document Ingestion** — 3 markdown files loaded from `data/docs/`
2. **Text Chunking** — Documents split into overlapping chunks (500 chars, 100 overlap)
3. **Embeddings** — TF-IDF token vectors generated for each chunk
4. **Retrieval** — Cosine similarity used to find top-3 relevant chunks
5. **Generation** — Retrieved context passed to Groq LLaMA for grounded response

**Knowledge Base Documents:**
- `budgeting_strategies.md` — 50/30/20 rule, zero-based budgeting, envelope method
- `saving_techniques.md` — Emergency funds, automated savings, expense reduction
- `financial_literacy.md` — Compound interest, net worth, debt management, inflation

---

## 🔄 n8n Workflow

The n8n workflow (`n8n/finance-assistant-workflow.json`) orchestrates:

1. **Webhook Trigger** — Receives user queries via HTTP POST
2. **Intent Router** — Classifies query as financial data or knowledge request
3. **Backend Call** — Forwards to FastAPI `/chat` endpoint
4. **Response Handler** — Returns structured response to caller

Import the JSON file into your n8n instance to activate the workflow.

---

## 💬 Example Queries

### Transaction Queries
```
"How much did I spend last week?"
"What category did I spend the most on this week?"
"Compare this week to last week"
"Show me my recent transactions"
"What was my biggest expense this month?"
"How much did I spend on food?"
```

### Financial Advice (RAG)
```
"What is a good budgeting strategy for beginners?"
"How can I save more money?"
"What is the 50/30/20 rule?"
"How do I build an emergency fund?"
"What is zero-based budgeting?"
"How does inflation affect my savings?"
```

### Multi-turn Memory
```
Turn 1: "How much did I spend on shopping?"
Turn 2: "Is that too much?"
Turn 3: "What should I do to cut it down?"
```
The assistant remembers context across all turns within the same session.

---

## 📊 Mock Transactions Dataset

25 simulated transactions covering categories:
- 🍔 Food & Dining
- 🛒 Groceries
- 🏠 Rent & Utilities
- 🚗 Transport
- 🛍️ Shopping
- 💊 Health & Fitness
- 🎬 Entertainment

Each transaction includes: `id`, `user_id`, `amount`, `category`, `description`, `timestamp`

---

## 🎁 Bonus Features Implemented

| Feature | Status |
|---------|--------|
| Conversation Memory | ✅ In-memory session store |
| Clean Chat UI | ✅ Dark theme with sidebar |
| Intent Badge | ✅ Shows `transactions` or `rag` per response |
| Response Latency | ✅ Live ms counter in UI |
| API Documentation | ✅ Swagger UI at /docs |
| Observability | ✅ Console logging for all requests |

---

## ⚠️ Assumptions & Limitations

- **Mock Data Only** — No real banking API is used. All transactions are simulated.
- **In-Memory Storage** — Session memory resets when server restarts. Redis can be added for persistence.
- **Local RAG** — TF-IDF used instead of OpenAI embeddings. Suitable for demo; production would use a proper vector DB (Pinecone, ChromaDB, etc.)
- **Single User** — System designed for demo with `user_001`. Multi-user auth can be added.
- **n8n** — Workflow requires a running n8n instance to activate. JSON export is provided.

---

## 📝 Design Decisions

1. **Groq + LLaMA 3.3** — Chosen for fast inference (2-4 second responses) and free tier availability
2. **LangGraph over LangChain** — More explicit state management, easier to debug agent routing
3. **TF-IDF RAG** — No external dependencies or API costs; sufficient for financial document retrieval
4. **FastAPI** — Async support, automatic OpenAPI docs, lightweight
5. **No Redis** — Kept simple with in-memory store; Redis integration is straightforward to add

---

## 👨‍💻 Author

Built as part of AI Engineer Assessment.
