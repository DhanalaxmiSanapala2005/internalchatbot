"""
app.py
======
ChatGPT-style Streamlit UI for the Internal Company Assistant.
All backend logic (RAG, embeddings, vector store, LLM, SQLite) lives in core.py.
"""

import uuid
import streamlit as st
import core

st.set_page_config(
    page_title="Internal Company Assistant",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------------
# Design: an "internal archive" feel rather than a generic chat-toy look —
# warm paper background, ink-navy text, a single deep-green accent used
# sparingly, and a serif/sans pairing (Newsreader for headings, Inter for
# UI chrome) since this tool's whole job is trustworthy documents, not
# casual conversation.
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600;6..72,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --ink: #1B2430;
        --paper: #FAF8F4;
        --paper-raised: #FFFFFF;
        --slate: #5B6472;
        --signal: #2F6F4E;
        --signal-soft: #E7EFE9;
        --line: #E4E0D8;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp { background-color: var(--paper); }

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--paper);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { color: var(--ink); }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 2px 18px 2px;
    }
    .sidebar-brand .mark {
        width: 30px; height: 30px; min-width: 30px;
        border-radius: 8px;
        background-color: var(--signal);
        color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Newsreader', serif;
        font-weight: 600; font-size: 1rem;
    }
    .sidebar-brand .name {
        font-family: 'Newsreader', serif;
        font-weight: 600;
        font-size: 1.08rem;
        letter-spacing: -0.01em;
    }

    .conv-group-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--slate);
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin: 18px 4px 6px 4px;
    }

    hr { border-color: var(--line) !important; margin: 1.1rem 0 !important; }

    /* ---------------- Buttons ---------------- */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--line);
        background-color: var(--paper-raised);
        color: var(--ink);
        font-weight: 500;
        font-size: 0.88rem;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--signal);
        color: var(--signal);
        background-color: var(--signal-soft);
    }
    .new-chat-btn button {
        background-color: var(--ink) !important;
        color: var(--paper) !important;
        border: 1px solid var(--ink) !important;
        font-weight: 600 !important;
    }
    .new-chat-btn button:hover {
        background-color: #2B3648 !important;
        color: var(--paper) !important;
    }

    /* ---------------- Expanders ---------------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        background-color: var(--paper-raised);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* ---------------- File uploader ---------------- */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: var(--paper);
        border: 1px dashed var(--line);
        border-radius: 10px;
    }

    /* ---------------- Sliders ---------------- */
    input[type="range"] { accent-color: var(--signal); }
    div[data-baseweb="slider"] div[role="slider"] { background-color: var(--signal) !important; }

    /* ---------------- Main chat header ---------------- */
    .chat-title {
        font-family: 'Newsreader', serif;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
        color: var(--ink);
    }
    .chat-subtitle {
        color: var(--slate);
        margin-bottom: 1.6rem;
        font-size: 0.96rem;
    }

    /* ---------------- Empty state ---------------- */
    .empty-state {
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 28px 24px;
        color: var(--slate);
        font-size: 0.92rem;
        background-color: var(--paper-raised);
        margin-bottom: 1rem;
    }
    .empty-state b { color: var(--ink); }

    /* ---------------- Chat bubbles ---------------- */
    .user-bubble {
        background-color: var(--ink);
        color: var(--paper);
        padding: 12px 16px;
        border-radius: 14px 14px 3px 14px;
        max-width: 72%;
        margin-left: auto;
        margin-bottom: 16px;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .assistant-row {
        display: flex;
        gap: 10px;
        margin-bottom: 6px;
        align-items: flex-start;
    }
    .assistant-avatar {
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 7px;
        background-color: var(--signal);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--paper);
        font-family: 'Newsreader', serif;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .assistant-bubble {
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        color: var(--ink);
        padding: 12px 16px;
        border-radius: 3px 14px 14px 14px;
        max-width: 76%;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* ---------------- Source citations ---------------- */
    .source-card {
        display: flex;
        gap: 10px;
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        border-left: 3px solid var(--signal);
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 6px;
        margin-left: 38px;
        font-size: 0.82rem;
        color: var(--slate);
        align-items: baseline;
    }
    .source-card b { color: var(--ink); font-weight: 600; }
    .source-card .page-badge {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.72rem;
        color: var(--signal);
        background-color: var(--signal-soft);
        border-radius: 4px;
        padding: 1px 6px;
    }

    /* ---------------- Knowledge base / security cards ---------------- */
    .kb-card {
        border: 1px solid var(--line);
        border-left: 3px solid var(--signal);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: var(--paper-raised);
    }

    /* ---------------- Chat input ---------------- */
    div[data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid var(--line) !important;
    }

    /* ---------------- Scrollbar ---------------- */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background-color: var(--line); border-radius: 4px; }
    ::-webkit-scrollbar-track { background-color: transparent; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# SESSION STATE INIT  (SQLite remains the source of truth)
# ------------------------------------------------------------------
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "rag_settings" not in st.session_state:
    st.session_state.rag_settings = {
        "chunk_size": core.Config.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": core.Config.DEFAULT_CHUNK_OVERLAP,
        "top_k": core.Config.DEFAULT_TOP_K,
        "similarity_threshold": core.Config.DEFAULT_SIMILARITY_THRESHOLD,
    }

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "session_id" not in st.session_state:
    # Stable per-browser-tab identifier used for rate limiting and
    # security audit logging in core.py — not a user account.
    st.session_state.session_id = str(uuid.uuid4())


def new_chat():
    st.session_state.active_conversation_id = None


def load_conversation(conv_id: str):
    st.session_state.active_conversation_id = conv_id


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="mark">IC</div>'
        '<div class="name">Internal Company Assistant</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("+ New Chat", use_container_width=True):
        new_chat()
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.search_query = st.text_input(
        "Search", placeholder="Search conversations...",
        label_visibility="collapsed", value=st.session_state.search_query,
    )

    conversations = core.search_conversations(st.session_state.search_query)
    grouped = core.group_conversations_by_date(conversations)

    for label in ["TODAY", "YESTERDAY", "PREVIOUS 7 DAYS", "OLDER"]:
        items = grouped[label]
        if not items:
            continue
        st.markdown(f'<div class="conv-group-label">{label}</div>', unsafe_allow_html=True)
        for conv in items:
            col1, col2 = st.columns([5, 1])
            with col1:
                is_active = conv["id"] == st.session_state.active_conversation_id
                label = ("› " if is_active else "") + conv["title"]
                if st.button(
                    label,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                ):
                    load_conversation(conv["id"])
                    st.rerun()
            with col2:
                if st.button("×", key=f"del_{conv['id']}"):
                    core.delete_conversation(conv["id"])
                    if st.session_state.active_conversation_id == conv["id"]:
                        st.session_state.active_conversation_id = None
                    st.rerun()

    st.markdown("---")

    # ---------------- Knowledge Base ----------------
    with st.expander("Knowledge Base", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload Documents", type=["pdf", "docx", "txt"],
            accept_multiple_files=True, label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("Process Uploaded Files", use_container_width=True):
                for uf in uploaded_files:
                    file_bytes = uf.read()
                    with st.spinner(f"Processing {uf.name}..."):
                        result = core.ingest_document(
                            uf.name, file_bytes,
                            chunk_size=st.session_state.rag_settings["chunk_size"],
                            chunk_overlap=st.session_state.rag_settings["chunk_overlap"],
                        )
                    if result.error:
                        st.error(f"✗ {uf.name}: {result.error}")
                    elif result.skipped:
                        st.info(f"⏭ {uf.name} already in knowledge base (skipped).")
                    else:
                        st.success(f"✓ {uf.name} processed\n\n✓ {result.chunks_created} chunks created\n\n✓ Added to knowledge base")

        st.markdown("**Documents in Knowledge Base**")
        docs = core.list_knowledge_base()
        if not docs:
            st.caption("No documents uploaded yet.")
        else:
            for d in docs:
                st.markdown(
                    f'<div class="kb-card"><b>{d["source"]}</b><br>'
                    f'<span style="color:#777;font-size:0.8rem;">'
                    f'{d["document_type"].upper()} • {d["chunks"]} chunks • '
                    f'{d["uploaded_at"][:10] if d["uploaded_at"] else "-"}</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Delete {d['source']}", key=f"delkb_{d['source']}"):
                    core.delete_document(d["source"])
                    st.rerun()

        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding index from stored documents..."):
                count = core.get_vector_store().rebuild_index_from_documents_dir(
                    st.session_state.rag_settings["chunk_size"],
                    st.session_state.rag_settings["chunk_overlap"],
                )
            st.success(f"Rebuilt index from {count} document(s).")

    # ---------------- RAG Settings ----------------
    with st.expander("RAG Settings", expanded=False):
        st.session_state.rag_settings["chunk_size"] = st.slider(
            "Chunk size", 200, 2000, st.session_state.rag_settings["chunk_size"], step=50
        )
        st.session_state.rag_settings["chunk_overlap"] = st.slider(
            "Chunk overlap", 0, 500, st.session_state.rag_settings["chunk_overlap"], step=10
        )
        st.session_state.rag_settings["top_k"] = st.slider(
            "Top K", 1, 15, st.session_state.rag_settings["top_k"]
        )
        st.session_state.rag_settings["similarity_threshold"] = st.slider(
            "Similarity threshold", 0.0, 1.0,
            st.session_state.rag_settings["similarity_threshold"], step=0.05
        )

    # ---------------- Security ----------------
    with st.expander("Security", expanded=False):
        st.caption(
            f"Prompt-injection filter: **{'ON' if core.Config.ENABLE_PROMPT_INJECTION_FILTER else 'OFF'}**\n\n"
            f"Content moderation: **{'ON' if core.Config.ENABLE_CONTENT_MODERATION else 'OFF'}**\n\n"
            f"PII redaction on answers: **{'ON' if core.Config.ENABLE_PII_REDACTION else 'OFF'}**\n\n"
            f"Max input length: **{core.Config.MAX_INPUT_CHARS} chars**\n\n"
            f"Max upload size: **{core.Config.MAX_UPLOAD_MB} MB**\n\n"
            f"Rate limit: **{core.Config.RATE_LIMIT_MAX_REQUESTS} msgs / "
            f"{core.Config.RATE_LIMIT_WINDOW_SECONDS}s per session**\n\n"
            "All settings are configured via `.env`."
        )
        st.markdown("**Recent security events**")
        events = core.get_recent_security_events(limit=15)
        if not events:
            st.caption("No security events logged yet.")
        else:
            for e in events:
                st.markdown(
                    f'<div class="kb-card"><b>{e["event_type"]}</b><br>'
                    f'<span style="color:#777;font-size:0.78rem;">{e["timestamp"][:19]} '
                    f'&mdash; {e["detail"]}</span></div>',
                    unsafe_allow_html=True,
                )

    # ---------------- About ----------------
    with st.expander("About", expanded=False):
        st.caption(
            f"LLM Provider: **{core.Config.LLM_PROVIDER}**\n\n"
            f"Embedding model: **{core.Config.EMBEDDING_MODEL}**\n\n"
            "Retrieval-augmented assistant for internal company knowledge: "
            "meeting notes, project docs, PDFs, DOCX, TXT files."
        )


# ------------------------------------------------------------------
# MAIN CHAT AREA
# ------------------------------------------------------------------
if core.COLLECTION_WAS_RESET:
    st.warning(core.COLLECTION_WAS_RESET)

st.markdown('<div class="chat-title">Internal Company Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-subtitle">Ask questions about company meetings, projects and documents.</div>',
    unsafe_allow_html=True,
)

# Load existing messages (or empty state for a brand-new chat)
messages = []
if st.session_state.active_conversation_id:
    messages = core.get_messages(st.session_state.active_conversation_id)

chat_container = st.container()
with chat_container:
    if not messages:
        st.markdown(
            '<div class="empty-state">Ask a question to get started — for example, '
            '<b>"What were the key decisions in last week\'s project meeting?"</b> '
            'Answers are drawn only from documents in your Knowledge Base, with sources cited below each answer.</div>',
            unsafe_allow_html=True,
        )
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="assistant-row">', unsafe_allow_html=True)
            st.markdown('<div class="assistant-avatar">AI</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if msg.get("sources"):
                with st.container():
                    for s in msg["sources"]:
                        page_badge = (
                            f'<span class="page-badge">p. {s["page"]}</span>'
                            if s.get("page") else ""
                        )
                        st.markdown(
                            f'<div class="source-card"><b>{s["source"]}</b>{page_badge} '
                            f'<span>score {s.get("score", "")}</span></div>',
                            unsafe_allow_html=True,
                        )

# ------------------------------------------------------------------
# CHAT INPUT
# ------------------------------------------------------------------
user_question = st.chat_input("Ask a question about your company documents...")

if user_question:
    # Create a conversation if this is a brand-new chat
    if not st.session_state.active_conversation_id:
        title = user_question.strip()[:50] + ("..." if len(user_question.strip()) > 50 else "")
        conv_id = core.create_conversation(title)
        st.session_state.active_conversation_id = conv_id
    else:
        conv_id = st.session_state.active_conversation_id

    core.add_message(conv_id, "user", user_question)

    with st.spinner("Searching company documents and generating an answer..."):
        settings = st.session_state.rag_settings
        response = core.answer_question(
            user_question,
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
            top_k=settings["top_k"],
            similarity_threshold=settings["similarity_threshold"],
            session_id=st.session_state.session_id,
        )

    core.add_message(conv_id, "assistant", response.answer, response.sources)
    st.rerun()
