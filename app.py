import streamlit as st
import tempfile
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain_core.documents import Document

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

import streamlit as st

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
template = """You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided content."

Context:
{context}

Conversation so far:
{history}

Current question:
{question}

Answer:"""

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def ocr_pdf(file_path):
    pages = convert_from_path(file_path, dpi=200)
    documents = []
    for i, page_img in enumerate(pages):
        text = pytesseract.image_to_string(page_img)
        documents.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
    return documents

def load_page(url):
    import time
    loader = SeleniumURLLoader(urls=[url], headless=True, arguments=["--no-sandbox", "--disable-dev-shm-usage"])
    documents = loader.load()
    time.sleep(3)
    return documents

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        if sum(len(d.page_content.strip()) for d in documents) < 20:
            documents = ocr_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    return documents

def split_text(documents):
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True).split_documents(documents)

def index_docs(documents):
    st.session_state.vector_store.add_documents(documents)

def retrieve_docs(query):
    return st.session_state.vector_store.similarity_search(query, k=8)

def stream_answer(question, context, history="No previous conversation."):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | st.session_state.model
    yield from chain.stream({"question": question, "context": context, "history": history})

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scraped_urls" not in st.session_state:
    st.session_state.scraped_urls = []
if "loaded_pdf" not in st.session_state:
    st.session_state.loaded_pdf = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(embeddings)
if "model" not in st.session_state:
    st.session_state.model = OllamaLLM(model="llama3.2", streaming=True)

# ─────────────────────────────────────────────
# PAGE CONFIG — no sidebar
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ContextFlow",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# COMPUTE STATE
# ─────────────────────────────────────────────
has_source = bool(st.session_state.scraped_urls or st.session_state.loaded_pdf)

if has_source:
    active_source = st.session_state.loaded_pdf or f"{len(st.session_state.scraped_urls)} page(s) indexed"
else:
    active_source = None

# ─────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([3, 2, 1.2])

with nav_col1:
    st.markdown(
        '<div class="cf-nav-brand">Context<span style="color:#A78BFA">Flow</span></div>',
        unsafe_allow_html=True
    )

with nav_col2:
    if has_source:
        st.markdown(
            '<div class="cf-status ready" style="margin-top:2px">'
            '<div class="cf-dot ready"></div>'
            f'{active_source}'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="cf-status idle" style="margin-top:2px">'
            '<div class="cf-dot idle"></div>'
            'awaiting source'
            '</div>',
            unsafe_allow_html=True
        )

with nav_col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.scraped_urls = []
        st.session_state.loaded_pdf = ""
        st.session_state.vector_store = InMemoryVectorStore(embeddings)
        st.rerun()

st.markdown('<hr style="margin-top:0.4rem;margin-bottom:0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="cf-hero">
    <p class="cf-hero-title">Context<span>Flow</span></p>
    <p class="cf-hero-sub">ground your questions in real documents, not hallucinations.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KNOWLEDGE SOURCES (collapsible)
# ─────────────────────────────────────────────
with st.expander("Knowledge Sources", expanded=True):
    col_url, col_pdf = st.columns([1.15, 0.85], gap="medium")

    with col_url:
        url_input = st.text_area(
            "Web Pages",
            placeholder="https://example.com\nhttps://another.com",
            height=108,
            label_visibility="visible"
        )

    with col_pdf:
        uploaded_pdf = st.file_uploader(
            "PDF Document",
            type=["pdf"],
            accept_multiple_files=False
        )

    # Model selector — small, tucked below sources
    mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
    with mcol1:
        st.markdown('<div class="cf-label">Model</div>', unsafe_allow_html=True)
        selected_model = st.selectbox("Model", ["llama3.2"], index=0, label_visibility="collapsed")
        st.session_state.model = OllamaLLM(model=selected_model, streaming=True)

    st.divider()


# ─────────────────────────────────────────────
# PDF LOADING
# ─────────────────────────────────────────────
if uploaded_pdf and uploaded_pdf.name != st.session_state.loaded_pdf:
    with st.spinner(f"Reading and embedding {uploaded_pdf.name}…"):
        pdf_docs = load_pdf(uploaded_pdf)
        chunked = split_text(pdf_docs)
        index_docs(chunked)
        st.session_state.loaded_pdf = uploaded_pdf.name
        st.session_state.scraped_urls = []
        st.session_state.chat_history = []
    st.success(f"**{uploaded_pdf.name}** indexed — ask anything about it.")

# ─────────────────────────────────────────────
# URL SCRAPING
# ─────────────────────────────────────────────
urls = [u.strip() for u in url_input.splitlines() if u.strip()]

if urls and urls != st.session_state.scraped_urls:
    with st.spinner(f"Scraping and embedding {len(urls)} page(s)…"):
        all_documents = []
        for url in urls:
            st.toast(f"Indexing {url}", icon="🔗")
            all_documents.extend(load_page(url))
        chunked_docs = split_text(all_documents)
        index_docs(chunked_docs)
        st.session_state.scraped_urls = urls
        st.session_state.loaded_pdf = ""
        st.session_state.chat_history = []
    st.success(f"**{len(urls)} page(s)** indexed — ask anything about them.")

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View sources"):
                for i, source_doc in enumerate(message["sources"], start=1):
                    origin = source_doc.metadata.get("source", "Unknown source")
                    icon = "📄" if str(origin).endswith(".pdf") else "🔗"
                    excerpt = source_doc.page_content[:280]
                    st.markdown(
                        f'<div class="cf-src">'
                        f'<span class="cf-src-icon">{icon}</span>'
                        f'<div><div class="cf-src-origin">src {i} &middot; <code style="color:#7C5CDB;font-size:0.68rem">{origin}</code></div>'
                        f'<div class="cf-src-excerpt">{excerpt}…</div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
if not st.session_state.chat_history:
    if not has_source:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem 2rem;">
            <div style="font-size:1.8rem;margin-bottom:0.8rem;opacity:0.12;">🔍</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.74rem;
                        color:#252535;letter-spacing:0.04em;line-height:2;">
                paste a URL or upload a PDF above<br>then ask anything about it
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem 1.5rem;">
            <div style="font-size:1.8rem;margin-bottom:0.8rem;opacity:0.15;">💬</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.74rem;
                        color:#252535;letter-spacing:0.04em;">
                source ready — type your first question below
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
question = st.chat_input("Ask anything about the loaded content…")

if question:
    if not has_source:
        st.warning("Load a URL or PDF first, then ask your question.")
    else:
        with st.chat_message("user"):
            st.write(question)

        source_documents = retrieve_docs(question)
        context = "\n\n".join([doc.page_content for doc in source_documents])

        history_lines = []
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_str = "\n".join(history_lines) if history_lines else "No previous conversation."

        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(question, context, history_str))
            if source_documents:
                with st.expander("📄 View sources"):
                    for i, source_doc in enumerate(source_documents, start=1):
                        origin = source_doc.metadata.get("source", "Unknown source")
                        icon = "📄" if str(origin).endswith(".pdf") else "🔗"
                        excerpt = source_doc.page_content[:280]
                        st.markdown(
                            f'<div class="cf-src">'
                            f'<span class="cf-src-icon">{icon}</span>'
                            f'<div><div class="cf-src-origin">src {i} &middot; <code style="color:#7C5CDB;font-size:0.68rem">{origin}</code></div>'
                            f'<div class="cf-src-excerpt">{excerpt}…</div></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_documents
        })
