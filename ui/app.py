"""Streamlit UI - Agentic RAG Chatbot (Professional Light Theme)"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent import run_agent

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
        from rag import RagSubgraph, RagConfig
        try:
            _rag = RagSubgraph(RagConfig())
            _rag.load_documents()
            _rag.split_documents()
            _rag.build_index()
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
if "nodes" not in st.session_state:
    st.session_state.nodes = [
        {"id": 0, "name": "Vezérlés", "status": "idle", "desc": "Kérdés fogadása, szövegfeldolgozás", "elapsed": 0.0},
        {"id": 1, "name": "Döntés", "status": "idle", "desc": "Intent értékelése, útvonal döntés", "elapsed": 0.0},
        {"id": 2, "name": "RAG Keresés", "status": "idle", "desc": "TF-IDF index, vektorkeresés, relevancia", "elapsed": 0.0},
        {"id": 3, "name": "Eszköz", "status": "idle", "desc": "Számítás, dátum, egyszerű logikai műveletek", "elapsed": 0.0},
        {"id": 4, "name": "Generálás", "status": "idle", "desc": "GLM4 válasz generálás, stream", "elapsed": 0.0},
    ]
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
.status-done { background:rgba(99,102,241,0.12); color:#6366f1; }
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
"""


# Inject CSS
st.html(f"<style>{TEMPLATE}</style>")

# Remove empty Streamlit widget labels from DOM and auto-scroll chat to bottom
components.html("""<script>
(function(){var d=window.parent.document;function rm(){var l=d.querySelector('label[data-testid="stWidgetLabel"]');if(l&&!l.textContent.trim())l.remove();else setTimeout(rm,300)}rm();function scrollChat(){var chat=d.querySelector('.chat-messages');if(chat){chat.scrollTop=chat.scrollHeight;}setTimeout(scrollChat,500);}scrollChat();})();
</script>""", height=0, scrolling=False)

# Header
st.html("<h1 style='font-size:1.5rem; font-weight:700; margin-bottom:0.25rem;'>⚡ <span style='color:#6366f1'>Agentic RAG</span> Pipeline</h1>")

# Quick questions
st.html('<div class="quick-bar">')
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
st.html('</div>')

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
    st.html(chat_html)

    with st.container():
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                "",
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
            <div class="node-desc">{n["desc"]}</div>
            {ops_html}
            <div class="node-status {status_cls}">{status_text}</div>
        </div>'''
    nodes_html += '</div>'
    st.html(nodes_html)

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

    # Retrieved Documents
    if st.session_state.get("retrieved_docs"):
        st.html('<div class="section-title">📄 Retrievált Dokumentumok</div>')
        docs = st.session_state.retrieved_docs
        if docs:
            st.html(f"<span style='font-size:0.75rem; color:#6b7280;'>{len(docs)} dokumentumot használtunk fel a válaszhoz.</span>")
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
            docs_html += f"""
            <div class="retrieved-doc">
            <div class="retrieved-doc-head"><span>Dokumentum {i+1}</span><span>{source}{score_str}</span></div>
            <div class="retrieved-doc-content">{doc_preview}</div>
            </div>
            """
        st.html(docs_html)

    # Prompt (GLM4-hez küldve)
    if st.session_state.get("prompt"):
        st.html('<div class="section-title">📤 Prompt (GLM4-hez)</div>')
        with st.expander("Mutasd a teljes promptot", expanded=False):
            st.code(st.session_state.prompt, language="text", line_numbers=True)

# Query processing logic
if st.session_state.pending_query:
    q = st.session_state.pending_query.strip()
    if q:
        st.session_state.messages.append({"role": "user", "text": q})
        st.session_state.running = True
        st.session_state.current_node = 0
        st.session_state.nodes[0]["status"] = "running"
        st.session_state.nodes[0]["elapsed"] = 0.0
        st.session_state.node_timings = []
        st.session_state.bottleneck = ""
        st.session_state.retrieved_docs = []
        st.session_state.pending_query = ""
        # Store start time
        st.session_state.actual_node_timings = {}
        st.rerun()

if st.session_state.running:
    current = st.session_state.current_node
    nodes = st.session_state.nodes
    total = len(nodes)
    
    if current < total:
        # Update elapsed time for running node
        if current < len(nodes):
            if current not in st.session_state.actual_node_timings:
                st.session_state.actual_node_timings[current] = time.time()
            else:
                elapsed = time.time() - st.session_state.actual_node_timings[current]
                nodes[current]["elapsed"] = max(nodes[current].get("elapsed", 0.0), elapsed)
        
        # Update node statuses
        for i, n in enumerate(nodes):
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
            for n in nodes:
                n["status"] = "completed"
            
            st.session_state.running = False
            st.session_state.current_node = -1
            st.rerun()
        except Exception as e:
            st.session_state.messages.append({"role": "bot", "text": "Hiba: " + str(e)[:100]})
            for n in nodes:
                n["status"] = "completed"
            st.session_state.running = False
            st.session_state.current_node = -1
            st.rerun()
