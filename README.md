# Sales Call AI Copilot

A command-line Retrieval-Augmented Generation (RAG) chatbot designed to help sales teams query, understand, and summarize past sales calls.

## Architecture & Storage Design

* **LLM**: Google Gemini for fast, conversational text generation.
* **Embeddings**: Google Generative AI Embeddings.
* **Vector Store**: Local FAISS index (`faiss-cpu`). FAISS was chosen because it is highly efficient for in-memory similarity search and allows the application to remain self-contained without requiring external database instances like Postgres/pgvector or Docker containerization.
* **Framework**: LangChain, strictly utilizing modern LangChain Expression Language (LCEL) pipelines.

---

## Setup Instructions

### 1. Create and Activate a Virtual Environment

Ensure you are using Python 3.9+, then create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the pinned requirements:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Copy `.env.example` to `.env` and add your Google API Key.

```bash
cp .env.example .env
```

*(Open `.env` and insert your `GOOGLE_API_KEY`)*

### 4. Application Configuration

Edit the existing `config.json` file in the root directory to manage your model names and chunking strategy.

```json
{
    "data_dir": "data",
    "index_dir": "faiss_index",
    "embedding_model": "models/gemini-embedding-001",
    "llm_model": "models/gemini-2.5-flash",
    "chunk_size": 1000,
    "chunk_overlap": 200
}
```

### 5. Add Data

Place your call transcript `.txt` files into a `data/` directory in the project root.

---

## Usage

### 1. Running the Interactive CLI

Launch the main application to interact with the Copilot:

```bash
python cli.py
```

**Supported Commands**:

* `list my call ids`: Lists all unique transcripts currently in the vector store.
* `ingest a new call transcript from <path>`: Embeds and stores a new document dynamically.
* `summarise the last call`: Generates a summary of the most recent call transcript.
* `Give me all negative comments when pricing was mentioned in the calls`: Retrieves pricing-related negative feedback across calls.
* `quit` or `exit`: Closes the application.
* *Natural Language*: Any other input is treated as a RAG question. The bot will search the transcripts and cite its sources.

### 2. Running the Evaluation Suite

To test the bot's factual recall and accuracy against the provided datasets, run evaluation mode from the unified test script:

```bash
python test_system.py --mode evaluation
```

This script will execute predefined factual questions, print the expected vs. actual responses to the console, and export a CSV (`evaluation_results.csv`) for manual citation review.

Use:

```bash
python test_system.py --mode evaluation
python test_system.py --mode all
```

### 3. Resetting the System
If you want to clear the Copilot's memory and start from a clean slate, you can delete the local vector database. The application will automatically generate a new one the next time it runs.

**On macOS/Linux:**
```bash
rm -rf faiss_index
```

**On Windows:**

```bash
rmdir /s /q faiss_index
```

---

## Troubleshooting & Known API Issues

**Google API `404 NOT_FOUND` Errors**
The original project requirements specified using `models/embedding-001` and `gemini-1.5-flash`. However, Google frequently deprecates older aliases on their `v1beta` endpoints. Depending on the region or the specific Google API key provided, using these exact strings might result in a `404 NOT_FOUND` error.

If you encounter a 404 error when running `cli.py`, it means your API key is tied to newer model aliases.

**To resolve this:**

1. Run the included utility script to check which models your specific key is authorized to use:
```bash
python check_models.py
```


2. Open `config.json` and update `embedding_model` and `llm_model` to match the exact strings printed in your terminal (e.g., `"models/text-embedding-004"` and `"models/gemini-1.5-flash-latest"`).

---

## Assumptions

* Call transcripts are standard text files (`.txt`).
* The local FAISS index is saved to a folder named `faiss_index/` to persist data between sessions.
* The chunking strategy (`size=1000`, `overlap=200`) is optimized for maintaining conversational context in standard B2B sales transcripts.