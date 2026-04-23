# RAG Customer Support Assistant
### LangGraph + ChromaDB + Human-in-the-Loop (HITL)

---

## Project Structure

```
rag_support_assistant/
├── main.py                # Entry point — CLI + interactive REPL
├── rag_pipeline.py        # PDF loading, embedding, retrieval, LLM answering
├── graph_workflow.py      # LangGraph nodes, edges, routing logic
├── hitl.py                # Human-in-the-Loop escalation handler
├── utils.py               # Logger, confidence heuristic, text cleaner
├── create_sample_pdf.py   # One-time helper to create a demo PDF
└── requirements.txt
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install fpdf2          # only needed for the sample PDF generator

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Generate the sample knowledge-base PDF  (skip if you have your own)
python create_sample_pdf.py

# 4. Index the PDF into ChromaDB
python main.py --build

# 5. Ask questions interactively
python main.py

# OR run a single query
python main.py --query "How do I return a product?"
```

---

## LangGraph Workflow

```
[START]
   │
   ▼
processing_node   ← retrieval + LLM answer generation
   │
   ▼
decision_node     ← checks confidence & escalation_flag
   │
   ├── "escalate" ──► escalation_node  ← HITL handler
   │                        │
   └── "output"  ──► output_node  ◄────┘
                        │
                      [END]
```

### Escalation triggers
| Condition | What happens |
|-----------|--------------|
| No documents found | `escalation_flag = True` immediately |
| Best similarity score < 0.30 | `escalation_flag = True` |
| LLM confidence < 0.50 | `escalation_flag = True` after decision node |
| All checks pass | Answer returned directly |

---

## State Object

```python
{
    "query"            : str,    # user's question
    "retrieved_docs"   : list,   # LangChain Document chunks
    "answer"           : str,    # LLM or human answer
    "confidence"       : float,  # 0.0 → 1.0 heuristic score
    "escalation_flag"  : bool,   # True if escalated
    "escalation_reason": str,    # why it was escalated
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI key |
| `PDF_PATH` | `knowledge_base.pdf` | Path to your PDF |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage dir |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `TOP_K` | `4` | Chunks to retrieve per query |
| `SIM_THRESHOLD` | `0.30` | Minimum similarity score |

---

## Using a Free/Local LLM (no OpenAI cost)

Swap the OpenAI classes in `rag_pipeline.py` for Ollama:

```bash
pip install langchain-ollama
ollama pull mistral
```

```python
# rag_pipeline.py — replace OpenAI imports with:
from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm        = ChatOllama(model="mistral", temperature=0)
```
