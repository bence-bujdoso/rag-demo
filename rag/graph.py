"""RAG subgraph - LangGraph implementáció.

Ez a fájl a moduláris RAG algráfot definiálja LangGraph-al.
A RAG subgraph függetlenül is futtatható, de a fő agent
hívja.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.documents import Document

from rag import RagSubgraph, DocumentState, RagConfig


class RagGraphInput(TypedDict):
    """A RAG graph belépési adatstruktúrája."""
    query: str
    retrieve_only: bool  # Ha True, csak retrieval, nem generálás


class RagGraphOutput(TypedDict):
    """A RAG graph kimeneti adatstruktúrája."""
    query: str
    retrieved_docs: List[Document]
    context: str
    answer: Optional[str]
    error: Optional[str]


def create_rag_subgraph(config: Optional[RagConfig] = None) -> StateGraph:
    """
    Létrehozza a modularis RAG subgraph-et (LangGraph).
    
    A subgraph legalább 3-5 csomópontot tartalmaz:
    1. load_documents_node  - Dokumentumok betöltése
    2. chunk_node           - Szövegrészletekre bontás
    3. embed_node           - Embedding és indexelés
    4. search_node          - Vektorkeresés
    5. context_node         - Kontextum összeállítása
    
    A subgraphot a fő agent hívja, vagy önállóan is futtatható.
    """
    rag = RagSubgraph(config)

    # --- Node-ok definiálása ---

    def load_documents_node(state: RagGraphInput) -> dict:
        """1. Node: Dokumentumok betöltése az adatforrásból."""
        docs = rag.load_documents()
        return {"documents": docs, "error": None}

    def chunk_node(state: dict) -> dict:
        """2. Node: Szövegrészletekre bontás."""
        chunks = rag.split_documents(state.get("documents", []))
        return {"chunks": chunks}

    def embed_node(state: dict) -> dict:
        """3. Node: Embedding és FAISS indexelés."""
        vectorstore = rag.build_vectorstore(state.get("chunks", []))
        return {"vectorstore": vectorstore}

    def search_node(state: RagGraphInput) -> dict:
        """4. Node: Vektorkeresés a lekérdezésre."""
        query = state["query"]
        retrieved = rag.search(query)
        return {"retrieved_docs": retrieved}

    def context_node(state: dict) -> dict:
        """5. Node: Kontextum összeállítása."""
        retrieved = state.get("retrieved_docs", [])
        context = rag.get_context(retrieved)
        return {"context": context}

    def answer_node(state: dict) -> dict:
        """6. Node: LLM válaszgenerálás (opcionális, a fő workflow-ban)."""
        return {}

    # --- Graf felépítése ---

    workflow = StateGraph(RagGraphInput, RagGraphOutput)

    # Node-ok hozzáadása
    workflow.add_node("load_documents", load_documents_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("embed", embed_node)
    workflow.add_node("search", search_node)
    workflow.add_node("context", context_node)

    # Élek hozzáadása (sorrend)
    workflow.add_edge("load_documents", "chunk")
    workflow.add_edge("chunk", "embed")
    workflow.add_edge("embed", "search")
    workflow.add_edge("search", "context")

    # Végpont
    workflow.add_edge("context", END)

    # Kezdőpont
    workflow.set_entry_point("load_documents")

    # Kompilálás
    app = workflow.compile()
    
    # Előtelepítjük a RAG-t is (a subgraph futtatásakor)
    rag.load_documents()
    rag.split_documents()
    rag.build_vectorstore()
    
    return app, rag


# --- Egyszerű függvény a subgraph futtatásához ---

def run_rag_subgraph(query: str, config: Optional[RagConfig] = None) -> dict:
    """
    Egyszerű függvény a RAG subgraph futtatásához.
    Visszaadja a teljes RAG eredményt.
    """
    app, rag = create_rag_subgraph(config)
    
    # Futtatás
    initial_state = {"query": query, "retrieve_only": False}
    result = app.invoke(initial_state)
    
    return {
        "query": query,
        "context": result.get("context", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
        "error": result.get("error", None)
    }


if __name__ == "__main__":
    # Teszt futtatása
    print("RAG Subgraph teszt...")
    result = run_rag_subgraph("Mennyi a minimálbér 2026-ben?")
    print(f"Kontextum: {result['context'][:300]}...")
