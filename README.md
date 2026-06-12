# ContextFlow 🔍

**ContextFlow** is a Retrieval-Augmented Generation (RAG) application that lets users chat with PDFs and web pages using local AI models. Upload a document or provide URLs, and ContextFlow retrieves relevant information before generating grounded answers with source attribution.

---

## Features

* 📄 Upload PDFs and query their contents
* 🔍 OCR support for scanned documents
* 🔗 Index and chat with web pages
* 💬 Conversational interface with chat history
* 📚 Source-backed responses
* ⚡ Local inference using Ollama
* 🎨 Clean custom Streamlit UI

---

## Tech Stack

| Component    | Technology          |
| ------------ | ------------------- |
| Frontend     | Streamlit           |
| LLM          | Llama 3.2 (Ollama)  |
| Embeddings   | nomic-embed-text    |
| Framework    | LangChain           |
| Vector Store | InMemoryVectorStore |
| OCR          | Tesseract OCR       |
| Web Scraping | Selenium            |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/contextflow.git
cd contextflow
```

### Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Required Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Run the Application

```bash
streamlit run app.py
```

---

## Usage

1. Upload a PDF or provide one or more URLs.
2. Wait for indexing to complete.
3. Ask questions about the content.
4. Review retrieved sources for transparency and verification.

---

## Project Structure

```text
contextflow/
├── app.py
├── style.css
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Future Improvements

* Support multiple LLM options
* Persistent vector database
* Document collections
* Export chat history
* Multi-file querying

---

## License

MIT License
