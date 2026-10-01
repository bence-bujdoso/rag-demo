"""Streamlit UI - Agentic RAG Chatbot (Professional Light Theme)"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import time
import hashlib
import base64
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent import run_agent
from tests.evaluation import EVALUATION_QUESTIONS

def _get_data_folder_hash(data_dir: str) -> str:
    """Számít sha256 hash-t a data mappa összes .txt fájljából (név + méret + mtime)."""
    h = hashlib.sha256()
    if not os.path.isdir(data_dir):
        return ""
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            stat = os.stat(filepath)
            h.update(filename.encode())
            h.update(str(stat.st_size).encode())
            h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()

def _init_rag_index():
    """Indítja el a RAG indexelt és tárolja session_state-ba. Csak indexel újra ha a data mappa változott."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    current_hash = _get_data_folder_hash(data_dir)
    
    if st.session_state.get("_rag_hash") != current_hash or "rag" not in st.session_state:
        from rag import HybridRAG, HybridConfig
        try:
            _rag = HybridRAG(HybridConfig())
            _rag.initialize()
            st.session_state.rag = _rag
            st.session_state._rag_hash = current_hash
        except Exception:
            st.session_state.rag = None
    return st.session_state.get("rag")

_ = _init_rag_index()  # Initialize RAG index (checks for changes)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = ""
if "node_timings" not in st.session_state:
    st.session_state.node_timings = []
if "bottleneck" not in st.session_state:
    st.session_state.bottleneck = ""
if "start_time" not in st.session_state:
    st.session_state.start_time = 0.0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0
if "nodes" not in st.session_state:
    st.session_state.nodes = [
        {"id": 0, "name": "Vezérlés", "status": "idle", "desc": "Kérdés fogadása, szövegfeldolgozás", "elapsed": 0.0},
        {"id": 1, "name": "Döntés", "status": "idle", "desc": "Intent értékelése, útvonal döntés", "elapsed": 0.0},
        {"id": 2, "name": "TF-IDF Keresés", "status": "idle", "desc": "Szöveg TF-IDF vektorizálása és hasonlóság-számítás", "elapsed": 0.0},
        {"id": 3, "name": "Vektoros Keresés", "status": "idle", "desc": "Betöltött kérdés vektorizálása, FAISS keresés", "elapsed": 0.0},
        {"id": 4, "name": "Fúzió", "status": "idle", "desc": "RRF vagy súlyozott kombinálás (alpha)", "elapsed": 0.0},
        {"id": 5, "name": "Kontextus Összeállítás", "status": "idle", "desc": "Top dokumentumok szövegének összefűzése", "elapsed": 0.0},
        {"id": 6, "name": "Eszköz", "status": "idle", "desc": "Számítás, dátum, egyszerű logikai műveletek", "elapsed": 0.0},
        {"id": 7, "name": "Generálás", "status": "idle", "desc": "GLM4 válasz generálás, stream", "elapsed": 0.0},
    ]
if "test_mode" not in st.session_state:
    st.session_state.test_mode = None
if "test_running" not in st.session_state:
    st.session_state.test_running = False
if "test_results" not in st.session_state:
    st.session_state.test_results = None
if "test_progress" not in st.session_state:
    st.session_state.test_progress = ""
if "test_logs" not in st.session_state:
    st.session_state.test_logs = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "current_node" not in st.session_state:
    st.session_state.current_node = -1
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []
if "start_time" not in st.session_state:
    st.session_state.start_time = {}
if "actual_node_timings" not in st.session_state:
    st.session_state.actual_node_timings = {}

# Helper functions

def get_node_index_by_name(name):
    for i, n in enumerate(st.session_state.nodes):
        if n["name"] == name:
            return i
    return -1

# Simple CSS template
TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#ffffff; color:#171717; font-family:'Inter',system-ui,sans-serif; }
.card { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:0.9rem; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; }
.cn { font-weight:600; font-size:0.85rem; color:#111827; }
.cb { font-size:0.62rem; padding:0.12rem 0.35rem; border-radius:8px; font-weight:500; }
.bi { background:rgba(107,114,128,0.08); color:#6b7280; }
.br { background:rgba(59,130,246,0.12); color:#3b82f6; }
.bd { background:rgba(99,102,241,0.12); color:#6366f1; }

.chat-container { 
    display: flex; 
    flex-direction: column; 
    height: 520px; 
    background: #f9fafb; 
    border: 1px solid #e5e7eb; 
    border-radius: 14px; 
    overflow: hidden; 
}
.chat-messages { 
    flex: 1; 
    overflow-y: auto; 
    padding: 1rem; 
    display: flex; 
    flex-direction: column; 
    gap: 0.75rem; 
}
.msg-row { display: flex; gap: 0.5rem; max-width: 85%; }
.msg-row.user { align-self: flex-end; flex-direction: row-reverse; }
.msg-row.bot { align-self: flex-start; }
.msg-bubble { 
    padding: 0.6rem 0.9rem; 
    border-radius: 12px; 
    font-size: 0.875rem; 
    line-height: 1.5; 
    white-space: pre-wrap; 
    word-wrap: break-word; 
}
.msg-user { 
    background: #eff6ff; 
    border: 1px solid #dbeafe; 
    color: #1e3a5f;
    border-bottom-right-radius: 4px;
}
.msg-bot { 
    background: #ffffff; 
    border: 1px solid #e5e7eb; 
    color: #171717;
    border-bottom-left-radius: 4px;
}
.msg-avatar { 
    width: 28px; 
    height: 28px; 
    border-radius: 50%; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-size: 0.75rem; 
    flex-shrink: 0; 
}
.avatar-user { background: #dbeafe; color: #3b82f6; }
.avatar-bot { background: #f3f4f6; color: #6b7280; }

.chat-input-area { 
    border-top: 1px solid #e5e7eb; 
    padding: 0.75rem 1rem; 
    background: #ffffff; 
}
.chat-input-row { 
    display: flex; 
    gap: 0.5rem; 
    align-items: center; 
}
.chat-input { 
    flex: 1; 
    background: #ffffff; 
    border: 1px solid #d1d5db; 
    border-radius: 8px; 
    padding: 0.5rem 0.75rem; 
    color: #171717; 
    font-size: 0.875rem; 
    width: 100%;
    box-sizing: border-box;
}
.stTextInput input, div[data-testid="stTextInputField"] input {
    width: 100% !important;
    box-sizing: border-box;
}
[data-testid="stTextInput"] {
    width: 100% !important;
}
.chat-input:focus { outline:none; border-color:#6366f1; }
.send-btn { 
    background: #6366f1; 
    color: #ffffff; 
    border: none; 
    border-radius: 8px; 
    padding: 0.5rem 1rem; 
    font-weight: 600; 
    cursor: pointer; 
    white-space: nowrap; 
}
.send-btn:hover { background: #4f46e5; }
.send-btn:disabled { background: #a5b4fc; cursor: not-allowed; }

.quick-bar { 
    display: flex; 
    gap: 0.5rem; 
    padding: 0.5rem 0; 
    flex-wrap: wrap; 
}
.quick-btn { 
    background: #f3f4f6; 
    border: 1px solid #e5e7eb; 
    color: #374151; 
    border-radius: 8px; 
    padding: 0.4rem 0.85rem; 
    font-size: 0.75rem; 
    font-weight: 500; 
    cursor: pointer; 
    transition: all 0.15s; 
}
.quick-btn:hover { background: #e5e7eb; border-color: #d1d5db; }

.timing-bar { background:#e5e7eb; border-radius:4px; height:8px; margin:2px 0; }
.timing-fill { background:#6366f1; border-radius:4px; height:8px; }
.bottleneck { background:#fef3c7; border:1px solid #f59e0b; border-radius:8px; padding:0.5rem 0.7rem; margin-top:0.5rem; font-size:0.8rem; }

.pipeline-nodes { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.node-card { flex: 1; min-width: 140px; background:#ffffff; border:1px solid #e5e7eb; border-radius:10px; padding:0.75rem; }
.node-status { display: inline-flex; align-items: center; gap: 0.25rem; font-size:0.6rem; padding:0.1rem 0.3rem; border-radius:6px; font-weight:500; }
.status-idle { background:rgba(107,114,128,0.08); color:#6b7280; }
.status-running { background:rgba(59,130,246,0.12); color:#3b82f6; }
.status-done { background: rgba(76, 175, 80, 0.12);  /* green */ color:#6366f1; }
.node-name { font-weight:600; font-size:0.75rem; color:#111827; margin-bottom:0.25rem; }
.node-desc { font-size:0.65rem; color:#6b7280; }

.retrieved-doc { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:0.5rem; margin-bottom:0.4rem; font-size:0.7rem; }
.retrieved-doc-head { display:flex; justify-content:space-between; margin-bottom:0.25rem; font-weight:500; }
.retrieved-doc-content { color:#6b7280; white-space:pre-wrap; font-family:inherit; }

.main-content { display: flex; flex-direction: column; gap: 1rem; }
.section-title { font-weight:600; font-size:0.85rem; color:#111827; margin-bottom:0.5rem; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%; }

/* Small operation buttons */
.op-bar .stButton>button {
    padding: 0.2rem 0.5rem !important;
    font-size: 0.8rem !important;
    height: auto !important;
}
    .node-elapsed {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 2px;
    }
"""


# Inject CSS
st.html(f"<style>{TEMPLATE}</style>")

# Remove empty Streamlit widget labels from DOM
components.html("""<script>
(function(){var d=window.parent.document;function rm(){var l=d.querySelector('label[data-testid="stWidgetLabel"]');if(l&&!l.textContent.trim())l.remove();else setTimeout(rm,300)}rm();
</script>""", height=0, scrolling=False)

# Header
st.html("<h1 style='font-size:1.5rem; font-weight:700; margin-bottom:0.25rem;'>⚡ <span style='color:#6366f1'>Agentic RAG</span> Pipeline</h1>")

# Quick questions

# Test buttons

# Main content
left_col, right_col = st.columns([3, 2], gap="large")

# Left: Chat
with left_col:
    st.html('<div class="section-title">💬 Chat</div>')
    
    chat_html = '<div class="chat-container"><div class="chat-messages" id="chat-messages">'
    for m in st.session_state.messages[-20:]:
        role = m["role"]
        icon = "👤" if role == "user" else "🤖"
        avatar_cls = "avatar-user" if role == "user" else "avatar-bot"
        row_cls = "user" if role == "user" else "bot"
        bubble_cls = "msg-user" if role == "user" else "msg-bot"
        safe_text = m['text'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
        chat_html += f'''
        <div class="msg-row {row_cls}">
            <div class="msg-avatar {avatar_cls}">{icon}</div>
            <div class="msg-bubble {bubble_cls}">{safe_text}</div>
        </div>'''
    chat_html += '</div></div>'
    chat_html += '<script>var chat = window.parent.document.querySelector(\".chat-messages\"); if (chat) { chat.scrollTop = chat.scrollHeight; }</script>'
    st.html(chat_html)

    with st.container():
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                " ",
                placeholder="Írjon egy kérdést...",
                key="chat_input",
                disabled=st.session_state.running,
            )
        with col_send:
            send_clicked = st.button(
                "Küldés",
                key="send_btn",
                use_container_width=True,
                disabled=st.session_state.running,
            )

    if send_clicked and user_query and user_query.strip():
        st.session_state.pending_query = user_query.strip()
        st.rerun()

# Right: Pipeline nodes + timing
with right_col:
    st.html('<div class="section-title">🔧 Műveletek</div>')
    st.html('<div class="op-bar">')
    qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
    quick_questions = [
        ("🤔 Próbaidő?", "Mennyi a munkaszerződés próbaideje?"),
        ("💰 Nettó fizetés?", "Mi a nettó fizetés bruttóból?"),
        ("📅 Határidő?", "Mikor van munkaügyi bírósági határidő?"),
        ("🏠 Lakbér?", "Mennyi a lakbér támogatás 2026-ban?"),
        ("📊 Minimálbér?", "Mennyi a minimálbér 2026 szeptemberében?"),
    ]
    for i, (label, query) in enumerate(quick_questions):
        with [qcol1, qcol2, qcol3, qcol4, qcol5][i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_query = query
                st.rerun()

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        if st.button("Funkcionális Értékelés", key="func_test", use_container_width=True):
            st.session_state.test_mode = 'functional'
            st.session_state.test_running = True
            st.session_state.test_progress = "Starting functional test..."
            st.session_state.test_logs = []
            st.rerun()
    with tcol2:
        if st.button("Teljesítményteszt", key="load_test", use_container_width=True):
            st.session_state.test_mode = 'load'
            st.session_state.test_running = True
            st.session_state.test_progress = "Starting load test..."
            st.session_state.test_logs = []
            st.rerun()

    st.html('</div>')

    # Test Progress
    if st.session_state.test_running:
        st.html('<div class="section-title">🧪 Teszt Folyamatban</div>')
        st.html(f'<p>{st.session_state.test_progress}</p>')
        if st.session_state.test_logs:
            st.html('<p>Legutolsó lépések:</p>')
            st.html('<ul>')
            for log in st.session_state.test_logs[-5:]:
                st.html(f'<li>{log}</li>')
            st.html('</ul>')
    st.html('<div class="section-title">🔄 Pipeline Nódok</div>')
    nodes_html = '<div class="pipeline-nodes">'
    for i, n in enumerate(st.session_state.nodes):
        status = n["status"]
        status_cls = f"status-{status}" if status in ("idle", "running", "completed") else "status-idle"
        status_text = "● FUT" if status == "running" else ("✓ KÉSZ" if status == "completed" else "○ VÁR")
        
        # Get elapsed time from nodes
        elapsed = n.get("elapsed", 0.0)
        
        # Also check actual_node_timings (which stores start timestamps)
        if i in st.session_state.actual_node_timings:
            ts = st.session_state.actual_node_timings[i]
            if isinstance(ts, (int, float)):
                elapsed = max(elapsed, time.time() - ts)
            elif isinstance(ts, dict):
                elapsed = max(elapsed, ts.get("elapsed", 0.0))
        
        ops_html = ""
        if st.session_state.node_timings and i < len(st.session_state.node_timings):
            ops = st.session_state.node_timings[i].get("operations", [])
            if ops:
                ops_list = "".join([f"<div style='font-size:0.6rem; color:#6b7280; margin-left:0.5rem;'>• {op}</div>" for op in ops[:10]])
                ops_html = f"<div style='margin-top:0.25rem;'>{ops_list}</div>"
        
        # Add elapsed time to display
        elapsed_display = f" ({elapsed:.2f}s)" if elapsed > 0.0 else ""
        
        nodes_html += f'''
        <div class="node-card">
            <div class="node-name">{i+1}. {n["name"]}{elapsed_display}</div>
            <div class="node-desc">{n["desc"]}</div>\n                    <div class="node-elapsed">{elapsed:.2f}s</div>
            {ops_html}
            <div class="node-status {status_cls}">{status_text}</div>
        </div>'''
    nodes_html += '</div>'
    st.html(nodes_html)
    # Prompt (GLM4-hez küldve)
    if st.session_state.get("prompt"):
        st.html('<div class=\"section-title\">📤 Prompt (GLM4-hez)</div>')
        st.html(f"<p><strong>Időtartam:</strong> {st.session_state.total_time:.2f} másodperc</p>")
        with st.expander("Mutasd a teljes promptot", expanded=False):
            st.code(st.session_state.prompt, language="text", line_numbers=True)


    # Timing display
    if st.session_state.node_timings:
        st.html('<div class="section-title">⏱️ Nód Végrehajtási Idők</div>')
        max_time = max((t.get("duration_seconds", 0) for t in st.session_state.node_timings), default=1)
        timing_html = ""
        for t in st.session_state.node_timings:
            name = t.get("node_name", "?")
            dur = t.get("duration_seconds", 0)
            pct = (dur / max_time) * 100 if max_time > 0 else 0
            timing_html += f"""
            <div style="margin-bottom:6px;">
            <span style="font-size:0.75rem; color:#6b7280;">{name}</span>
            <span style="font-size:0.75rem; color:#111827; margin-left:0.5rem;">{dur:.2f}s</span>
            <div class="timing-bar" style="width:100%;"><div class="timing-fill" style="width:{pct}%;"></div></div>
            </div>
            """
        st.html(timing_html)

    # Bottleneck
    if st.session_state.bottleneck:
        st.html(f"""
        <div class="bottleneck">
        <strong>⚠️ Bottleneck:</strong> {st.session_state.bottleneck} (leghosszabb futás)
        </div>
        """)


    # Test Results
    if st.session_state.test_results:
        st.html('<div class="section-title">🧪 Teszt Eredmények</div>')
        res = st.session_state.test_results
        st.html(f"<p><strong>Típus:</strong> {res['mode']}</p>")
        st.html(f"<p><strong>Összes lekérdezés:</strong> {res['total_queries']}</p>")
        st.html(f"<p><strong>Átlagos latency:</strong> {res['avg_latency']:.2f} másodperc</p>")
        st.html(f"<p><strong>Minimum latency:</strong> {res['min_latency']:.2f} másodperc</p>")
        st.html(f"<p><strong>Maximum latency:</strong> {res['max_latency']:.2f} másodperc</p>")
        st.html(f"<p><strong>Fő szűk keresztmetszet:</strong> {res['bottleneck_node']} ({res['bottleneck_time']:.2f} másodperc)</p>")
        st.html('<p><strong>Átlagos nœd-idők:</strong></p>')
        st.html('<ul>')
        for name, timing in res['avg_node_timings'].items():
            st.html(f'<li>{name}: {timing:.2f} másodperc</li>')
        st.html('</ul>')
# Retrieved Documents
    # Show document count early if available from RAG search
    if st.session_state.get("retrieved_doc_count") is not None:
        st.html(f"""<div class='section-title' style='color:#111827; border-bottom:1px solid #e5e7eb; padding-bottom:0.25rem; margin-bottom:0.5rem;'>📄 Retrievált Dokumentumok</div>""")
        st.html(f"""<span style='font-size:0.85rem; color:#6b7280;'>📊 {st.session_state.retrieved_doc_count} dokumentum a válaszhoz</span>""")
        
        # Also show full details if we have completed the RAG processing (nodes 2-5)
        completed_rag_nodes = sum(1 for i in [2, 3, 4, 5] if i < len(st.session_state.nodes) and st.session_state.nodes[i].get("status") == "completed")
        if completed_rag_nodes >= 4:  # TF-IDF, Vector, Fusion, Context nodes completed
            docs = st.session_state.retrieved_docs
            if docs:
                docs_html = ""
                for i, doc in enumerate(docs):
                    if isinstance(doc, dict):
                        source = doc.get("source", "ismeretlen")
                        content = doc.get("content", doc.get("page_content", ""))
                        score = doc.get("score", None)
                        score_str = f" | Szim.: {score:.3f}" if score is not None else ""
                    else:
                        source = "ismeretlen"
                        content = doc
                        score_str = ""
                    doc_preview = content[:250] + "..." if len(content) > 250 else content
                    docs_html += f'''
<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:0.6rem; margin-bottom:0.5rem; font-size:0.75rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; font-weight:500;">
<span style="color:#111827;">Dokumentum {i+1}</span>
<span style="color:#6b7280; font-size:0.65rem;">{source}{score_str}</span>
</div>
<div style="color:#6b7280; white-space:pre-wrap; font-family:inherit; line-height:1.4; margin-top:0.1rem;">{doc_preview}</div>
</div>'''
                st.html(docs_html)
            else:
                st.html("""<span style='font-size:0.75rem; color:#6b7280;'>Nincs retrieved dokumentum.</span>""")
    else:
        # Fallback to original logic if no early count available
        completed_nodes = sum(1 for n in st.session_state.nodes if n.get("status") == "completed")
        if completed_nodes >= 7:
            st.html(f"""<div class='section-title' style='color:#111827; border-bottom:1px solid #e5e7eb; padding-bottom:0.25rem; margin-bottom:0.5rem;'>📄 Retrievált Dokumentumok</div>""")
            docs = st.session_state.retrieved_docs
            st.html(f"""<span style='font-size:0.75rem; color:#6b7280;'>{len(docs)} dokumentumot használtunk fel a válaszhoz.</span>""")
            if docs:
                docs_html = ""
                for i, doc in enumerate(docs):
                    if isinstance(doc, dict):
                        source = doc.get("source", "ismeretlen")
                        content = doc.get("content", doc.get("page_content", ""))
                        score = doc.get("score", None)
                        score_str = f" | Szim.: {score:.3f}" if score is not None else ""
                    else:
                        source = "ismeretlen"
                        content = doc
                        score_str = ""
                    doc_preview = content[:250] + "..." if len(content) > 250 else content
                    docs_html += f'''
<div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; padding:0.6rem; margin-bottom:0.5rem; font-size:0.75rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem; font-weight:500;">
<span style="color:#111827;">Dokumentum {i+1}</span>
<span style="color:#6b7280; font-size:0.65rem;">{source}{score_str}</span>
</div>
<div style="color:#6b7280; white-space:pre-wrap; font-family:inherit; line-height:1.4; margin-top:0.1rem;">{doc_preview}</div>
</div>'''
                st.html(docs_html)
            else:
                st.html("""<span style='font-size:0.75rem; color:#6b7280;'>Nincs retrieved dokumentum.</span>""")

# 2D RAG VISUALIZATION DASHBOARD
st.html("<div class='section-title'>📊 2D RAG Vizualizáció</div>")

# Create tabs for different visualization views
viz_tabs = st.tabs(["📈 Pipeline Monitoring", "📄 Document Analysis", "⚡ Performance"])

with viz_tabs[0]:  # Pipeline Monitoring
    st.html("<div style='background:#f8fafc; padding:1rem; border-radius:8px; margin-bottom:1rem;'>")
    # Real-time pipeline status
    if st.session_state.get("node_timings"):
        # Create a timeline chart using plotly
        try:
            import plotly.graph_objects as go
            import pandas as pd
            # Prepare data for pipeline timeline
            pipeline_data = []
            for i, node in enumerate(st.session_state.node_timings):
                name = node["node_name"]
                # Find timing for this node
                timing = next((t["duration_seconds"] for t in st.session_state.node_timings if t.get("node_name") == name), 0)
                pipeline_data.append({
                    "Node": f"{i+1}. {name}",
                    "Duration": timing,
                    "Status": "Completed" if node.get("duration_seconds", 0) > 0 else "Pending",
                    "Order": i
                })
            df = pd.DataFrame(pipeline_data)
            # Create horizontal bar chart
            fig = go.Figure()
            for idx, row in df.iterrows():
                color = "#6366f1" if row["Status"] == "Completed" else "#fbbf24" if row["Status"] == "Pending" else "#e5e7eb"
                fig.add_trace(go.Bar(
                    y=[row["Node"]],
                    x=[row["Duration"]],
                    orientation='h',
                    marker_color=color,
                    text=f"{row['Duration']:.2f}s",
                    textposition='inside',
                    showlegend=False
                ))
            fig.update_layout(
                title="Pipeline Node Execution Times",
                xaxis_title="Duration (seconds)",
                yaxis_title="Pipeline Nodes",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True, key="pipeline_chart")
        except ImportError:
            st.warning("Plotly not available")
            for i, node in enumerate(st.session_state.node_timings):
                name = node["node_name"]
                timing = next((t["duration_seconds"] for t in st.session_state.node_timings if t.get("node_name") == name), 0)
                st.progress(min(timing, 1.0), text=f"{i+1}. {name}: {timing:.2f}s")
    else:
        st.info("Start a query to see pipeline visualization")
    st.html("</div>")

with viz_tabs[1]:  # Document Analysis
    st.html("<div style='background:#f8fafc; padding:1rem; border-radius:8px; margin-bottom:1rem;'>")
    if st.session_state.get("retrieved_docs"):
        docs = st.session_state.retrieved_docs
        if docs:
            try:
                import plotly.graph_objects as go
                import pandas as pd
                # Prepare document data
                doc_data = []
                for i, doc in enumerate(docs):
                    if isinstance(doc, dict):
                        source = doc.get("source", "unknown")
                        content = doc.get("content", doc.get("page_content", "")) or ""
                        score = doc.get("score", 0.0) or 0.0
                    else:
                        source = "unknown"
                        content = doc
                        score = 0.0
                    doc_data.append({
                        "Document": f"Doc {i+1}",
                        "Source": source,
                        "Score": score,
                        "Length": len(content),
                        "Preview": content[:250] + "..." if len(content) > 250 else content
                    })
                df = pd.DataFrame(doc_data)
                # Create scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["Length"],
                    y=df["Score"],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=df["Score"],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Similarity Score")
                    ),
                    text=df["Document"],
                    textposition="top center",
                    hovertemplate="<b>%{text}</b><br>Source: %{customdata[0]}<br>Score: %{y:.3f}<br>Length: %{x} chars<extra></extra>",
                    customdata=df[["Source"]].values
                ))
                # Set x-axis range to max document length
                max_length = df["Length"].max() if not df.empty else 0
                fig.update_layout(
                    title="Document Retrieval Analysis",
                    xaxis_title="Document Length (characters)",
                    yaxis_title="Similarity Score",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(range=[0, max_length * 1.1])
                )
                st.plotly_chart(fig, use_container_width=True, key="document_chart")
                # Show document details
                st.html("<div style='margin-top:1rem;'>")
                for i, doc in enumerate(docs):
                    if isinstance(doc, dict):
                        source = doc.get("source", "unknown")
                        content = doc.get("content", doc.get("page_content", "")) or ""
                        score = doc.get("score", 0.0) or 0.0
                    else:
                        source = "unknown"
                        content = doc
                        score = 0.0
                    with st.expander(f"📄 Doc {i+1} | {source} | Score: {float(score) if score is not None else 0:.3f}"):
                        st.write(content[:500] + ("..." if len(content) > 500 else ""))
                st.html("</div>")
            except ImportError:
                st.warning("Plotly not available")
        else:
            st.info("No documents to visualize")
    else:
        st.info("No retrieved documents")
    st.html("</div>")

with viz_tabs[1]:  # Document Analysis
    st.html("<div style='background:#f8fafc; padding:1rem; border-radius:8px; margin-bottom:1rem;'>")
    if st.session_state.get("retrieved_docs"):
        messages = st.session_state.messages
        user_messages = [m for m in messages if m["role"] == "user"]
        if user_messages:
            import re
            from collections import Counter
            all_words = []
            for msg in user_messages:
                words = re.findall(r'\b[a-záéíóöőúüű]+\b', msg["text"].lower())
                all_words.extend(words)
            if all_words:
                word_counts = Counter(all_words)
                top_words = word_counts.most_common(10)
                try:
                    import plotly.graph_objects as go
                    words, counts = zip(*top_words) if top_words else ([], [])
                    fig = go.Figure([go.Bar(x=list(words), y=list(counts))])
                    fig.update_layout(title="Top 10 Query Words", height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True, key="query_chart")
                except ImportError:
                    for word, count in top_words:
                        st.write(f"{word}: {count}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Queries", len(user_messages))
            col2.metric("Unique Queries", len(set(m["text"] for m in user_messages)))
            col3.metric("Avg Length", f"{sum(len(m['text']) for m in user_messages)/len(user_messages):.0f} chars")
    st.html("</div>")

with viz_tabs[2]:  # Performance
    st.html("<div style='background:#f8fafc; padding:1rem; border-radius:8px; margin-bottom:1rem;'>")
    if st.session_state.get("node_timings"):
        timings = [t.get("duration_seconds", 0) for t in st.session_state.node_timings if t.get("duration_seconds") is not None]
        if timings:
            try:
                import plotly.graph_objects as go
                import pandas as pd
                timing_data = [{"Node": f"{i+1}. {t.get('node_name', 'Node')}", "Duration": t.get("duration_seconds", 0)} for i, t in enumerate(st.session_state.node_timings)]
                df = pd.DataFrame(timing_data)
                total = sum(timings)
                fig = go.Figure(data=[go.Pie(labels=df["Node"], values=df["Duration"])])
                fig.update_layout(title="Time Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True, key="perf_pie")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", f"{total:.2f}s")
                col2.metric("Avg", f"{sum(timings)/len(timings):.2f}s")
                col3.metric("Max", f"{max(timings):.2f}s")
                slow = max(st.session_state.node_timings, key=lambda x: x.get("duration_seconds", 0)) if st.session_state.node_timings else None
                col4.metric("Slowest", f"{slow.get('node_name', 'N/A')}: {max(timings):.2f}s" if slow else "N/A")
            except ImportError:
                st.info("Plotly not available")
                st.write(f"Total: {sum(timings):.2f}s, Max: {max(timings):.2f}s")
    st.html("</div>")

if st.session_state.pending_query:
    q = st.session_state.pending_query.strip()
    if q:
        st.session_state.messages.append({"role": "user", "text": q})
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.nodes[0]["elapsed"] = 0.0
        st.session_state.node_timings = []
        st.session_state.bottleneck = ""
        st.session_state.retrieved_docs = []
        st.session_state.pending_query = ""
        # Store start time
        st.session_state.actual_node_timings = {}
        st.rerun()

elif st.session_state.test_running:
    # We are in the middle of a test.
    # We'll check if we have already started the test (we store the current index in session state)
    if 'test_current_idx' not in st.session_state:
        # Initialize test
        st.session_state.test_current_idx = 0
        st.session_state.test_latencies = []
        st.session_state.test_node_timings = []  # list of lists of timings per query
        # Load questions
        if st.session_state.test_mode == 'functional':
            questions = EVALUATION_QUESTIONS
            st.session_state.test_total = len(questions)
        else:  # load test
            # For load test, we'll use the same questions but repeat them to get 50-200
            # Let's say we want 100 queries.
            base_questions = EVALUATION_QUESTIONS
            repeats = (100 + len(base_questions) - 1) // len(base_questions)
            questions = base_questions * repeats
            st.session_state.test_total = 100  # we'll only take first 100
            questions = questions[:100]
        # Store questions in session state for later use
        st.session_state.test_questions = questions
        st.session_state.test_progress = "Starting functional test..." if st.session_state.test_mode == 'functional' else "Starting load test..."
        st.session_state.test_logs = []

    idx = st.session_state.test_current_idx
    if idx < st.session_state.test_total:
        # Process one query
        questions = st.session_state.test_questions
        q = questions[idx]
        question_text = q['question'] if isinstance(q, dict) else q
        st.session_state.test_progress = f'Processing query {idx+1}/{st.session_state.test_total}'
        st.session_state.test_logs.append(f'Query {idx+1}: {question_text[:50]}...')
        start = time.time()
        result = run_agent(question_text, rag=st.session_state.rag)
        end = time.time()
        latency = end - start
        st.session_state.test_latencies.append(latency)
        st.session_state.test_node_timings.append(result.get('node_timings', []))
        st.session_state.test_current_idx = idx + 1
        st.rerun()
    else:
        # Test is done
        st.session_state.test_running = False
        # Compute results
        latencies = st.session_state.test_latencies
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        # Compute average timings per node
        node_timings_all = st.session_state.test_node_timings
        # Initialize a dict for each node: list of durations
        node_durations = {n['name']: [] for n in st.session_state.nodes}
        for timings in node_timings_all:
            for t in timings:
                name = t.get('node_name')
                if name in node_durations:
                    node_durations[name].append(t.get('duration_seconds', 0))
        avg_node_timings = {}
        for name, durations in node_durations.items():
            if durations:
                avg_node_timings[name] = sum(durations) / len(durations)
            else:
                avg_node_timings[name] = 0
        # Identify bottleneck: the node with the highest average time
        bottleneck_node = max(avg_node_timings, key=avg_node_timings.get) if avg_node_timings else None
        bottleneck_time = avg_node_timings.get(bottleneck_node, 0) if bottleneck_node else 0

        st.session_state.test_results = {
            'mode': st.session_state.test_mode,
            'total_queries': st.session_state.test_total,
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'avg_node_timings': avg_node_timings,
            'bottleneck_node': bottleneck_node,
            'bottleneck_time': bottleneck_time,
            'latencies': latencies,  # maybe we don't need to store all
        }
        st.rerun()

elif st.session_state.running:
    current = st.session_state.current_node
    session_nodes = st.session_state.nodes
    total = len(session_nodes)
    
    if current < total:
        # Update elapsed time for running node
        if current < len(session_nodes):
            if current not in st.session_state.actual_node_timings:
                st.session_state.actual_node_timings[current] = time.time()
            else:
                elapsed = time.time() - st.session_state.actual_node_timings[current]
                session_nodes[current]["elapsed"] = max(session_nodes[current].get("elapsed", 0.0), elapsed)
        
        # Update node statuses
        for i, n in enumerate(session_nodes):
            if i == current:
                n["status"] = "running"
            elif i < current:
                n["status"] = "completed"
            else:
                n["status"] = "idle"
        
        st.session_state.current_node = current + 1
        st.rerun()
    else:
        # All nodes completed, process result
        try:
            result = run_agent(
                st.session_state.messages[-1]["text"]
                if st.session_state.messages
                else "Mennyi a munkaszerződés próbaideje?",
                rag=st.session_state.rag
            )

            answer = result.get("answer", "N/A")
            st.session_state.messages.append({"role": "bot", "text": answer})
            st.session_state.result = answer
            st.session_state.node_timings = result.get("node_timings", [])
            st.session_state.retrieved_docs = result.get("retrieved_docs", [])
            st.session_state.retrieved_doc_count = result.get("retrieved_docs_count", 0)
            st.session_state.prompt = result.get("prompt", "")

            # Update node elapsed times with actual timings from result
            if st.session_state.node_timings:
                for t in st.session_state.node_timings:
                    name = t.get("node_name")
                    dur = t.get("duration_seconds", 0.0)
                    if name:
                        idx = get_node_index_by_name(name)
                        if idx >= 0 and idx < len(st.session_state.nodes):
                            st.session_state.nodes[idx]["elapsed"] = dur

            # Identify bottleneck
            bottleneck_nodes = ["Vezérlés", "Döntés", "RAG Keresés", "Eszköz"]
            bottleneck = ""
            max_dur = 0
            for t in st.session_state.node_timings:
                if t.get("node_name", "") in bottleneck_nodes:
                    if t.get("duration_seconds", 0) > max_dur:
                        max_dur = t.get("duration_seconds", 0)
                        bottleneck = t.get("node_name", "")
            st.session_state.bottleneck = bottleneck

            # Mark all nodes as completed
            for n in st.session_state.nodes:
                n["status"] = "completed"

            st.session_state.running = False
            total_time = time.time() - st.session_state.start_time
            st.session_state.total_time = total_time
            st.session_state.current_node = -1
            st.rerun()
        except Exception as e:
            st.session_state.messages.append({"role": "bot", "text": "Hiba: " + str(e)[:100]})
            for n in st.session_state.nodes:
                n["status"] = "completed"
            st.session_state.running = False
            total_time = time.time() - st.session_state.start_time
            st.session_state.total_time = total_time
            st.session_state.current_node = -1
            st.rerun()

else:
    # Idle state - do nothing
    pass
