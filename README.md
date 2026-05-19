# Helpdesk 2.0 with RAG

Intelligent helpdesk system based on LangGraph with vector search (RAG) using ChromaDB.

## Features

- 🎫 **Ticket Management** - Create and track support tickets
- 🔍 **RAG Vector Search** - Semantic search in knowledge base
- 🤖 **Automatic Classification** - Auto-categorizes tickets as automatic or escalated
- 👨‍💼 **Human-in-the-Loop** - Escalation system for complex issues
- 💾 **State Persistence** - SQLite checkpointing for conversation state
- 📊 **Confidence Scoring** - RAG confidence-based routing
- 🔄 **Streaming Updates** - Real-time progress tracking

## Description

Helpdesk 2.0 is an intelligent technical support system that uses:
- **LangGraph** for orchestrating ticket flow
- **ChromaDB** as vector database for semantic search
- **MultiQueryRetriever** for improved search accuracy
- **Human-in-the-Loop** for escalation to human agents
- **Checkpointing** with SQLite for state persistence

## Architecture

```
User → Classification → RAG (ChromaDB) → Confidence Evaluation
                                                  ↓
                              High Confidence → Automatic Response
                              Low Confidence → Human Escalation
```

## Techniques Used

### 1. Retrieval Augmented Generation (RAG)
- Vector search in knowledge base
- MultiQueryRetriever for broader coverage
- Confidence calculation based on relevance

### 2. LangGraph State Management
- Typed state with `TypedDict`
- SQLite checkpointing
- Interrupts for human intervention waiting
- Event streaming

### 3. Human-in-the-Loop
- Graph pause for human response waiting
- State update with human input
- Flow continuation post-intervention

## Project Structure

```
helpdesk_system/
├── app.py              # Streamlit application
├── graph.py            # LangGraph workflow
├── rag_system.py       # RAG system with ChromaDB
├── setup_rag.py        # Vectorstore configuration
├── config.py           # Configuration
├── docs/               # Knowledge base documentation
│   ├── faq.md
│   ├── manual_usuario.md
│   └── guia_resolucion_problemas.md
├── chroma_db/          # Vectorstore (git ignored)
└── .env.example        # Environment template
```

## Requirements

- Python 3.10+
- OpenAI API Key

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

## Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Configure RAG:**
   - On first run, click "Configure RAG" in the sidebar
   - This will load documents from `docs/` into the vectorstore

3. **Create tickets:**
   - Enter the problem description
   - The system will automatically classify
   - Automatic response or human escalation

4. **Handle escalated tickets:**
   - Escalated tickets show RAG context
   - Write a response or use the RAG response as base

## System Flow

1. 📝 User submits query
2. 🤖 Automatic classification (automatic/escalated)
3. 🔍 RAG vector search
4. 📊 Confidence evaluation (>0.6 = automatic)
5. 👨‍💼 Escalation if confidence is low
6. ✅ Final response