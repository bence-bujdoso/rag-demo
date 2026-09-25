"""Moduláris RAG alrendszer (subgraph).

A RAG subgraph felelős a dokumentumok töltésével,
indexelésével és a vektorkereséssel. A fő workflow
ezt a subgraphot hívja, és nem számít bele a 3-5
csomópontba.

Használat:
    from rag import RagSubgraph, RagConfig, HybridRAG, HybridConfig
    rag = RagSubgraph(RagConfig())
    result = rag.run_pipeline("Mennyi a minimálbér?")
    
    # Hibrid RAG használata:
    hybrid = HybridRAG(HybridConfig())
    hybrid.initialize()
    result = hybrid.retrieve("Mennyi a minimálbér?")
"""

import os
import re
import math
from typing import List, TypedDict, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.base import TFIDFVectorizer, RagConfig
from rag.hybrid import HybridRAG, HybridConfig


class DocumentState(TypedDict):
    """Az RAG belső állapota."""
    documents: List[Document]
    chunks: List[Document]
    query: Optional[str]
    retrieved_docs: List[Document]
    context: str
    error: Optional[str]


@dataclass
class RagConfig:
    """RAG konfiguráció."""
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    chunk_size: int = 800
    chunk_overlap: int = 100
    search_k: int = 10


class TFIDFVectorizer:
    """Egyszerű TF-IDF vektorioló saját megvalósításban."""

    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.document_vectors = []
        self.chunk_texts = []

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        stopwords = {'a', 'az', 'és', 'vagy', 'de', 'nem', 'ez',
                     'ezek', 'amely', 'amelyik', 'ha', 'így', 'is',
                     'mind', 'minden', 'mindig', 'mert', 'hogy', 'ami',
                     'csak', 'egész', 'újra', 'másik',
                     'azt', 'belül', 'közt', 'valamint', 'való',
                     'aztà', 'fel', 'nélkül', 'kovalyolt', 'tételben',
                     'pl', 'például', 'jel', 'szó', 'összesen',
                     'adó', 'adók', 'jövedelem', 'jövedelm', 'bevétel',
                     'fizetés', 'fizet', 'mérték', 'százalék', 'arány',
                     'határol', 'hatály', 'hatóság', 'elő', 'utó',
                     'kapcsolatban', 'vonatkozás', 'kapcsol', 'szabály',
                     'szabályok', 'rendelkezés', 'rendelkezik', 'rendelkez',
                     'igen', 'sem', 'válasz', 'válaszol', 'válaszítan',
                     'hivatal', 'hivatalos', 'kiegészít', 'kiegészítő', 'kiegészítő',
                     'elfogad', 'elfogadott', 'elfogad', 'elfogadni',
                     'rend', 'rendelet', 'rendelkezés', 'rendelkezik',
                     'értelm', 'értelmében', 'értelmeként',
                     'jelent', 'jelentése', 'jelent', 'jelölö', 'jelöl',
                     'szám', 'számuj', 'számot', 'számít',
                     'év', 'éve', 'évben', 'évre',
                     'hónap', 'hónapban', 'havi',
                     'nap', 'napon', 'napi', 'napján',
                     'napra', 'napfény',
                     'kor', 'kori', 'kora', 'korú', 'korával',
                     'éves', 'évvel', 'évre', 'évvel',
                     'személy', 'személyi', 'személyek', 'személyünk',
                     'társas', 'társaság', 'társasági',
                     'jogi', 'jogszabály', 'jogszabálytár',
                     'könyv', 'közlöny', 'könyh',
                     'rend', 'rendelet', 'rendelkezés',
                     'igaz', 'igazán', 'igazan', 'igazság',
                     'való', 'valóban', 'valójában',
                     'valamilyen', 'valójában',
                     'tisz', 'tiszta', 'tiszta szabályok',
                     'kül', 'külön', 'különböz', 'különböző',
                     'másik', 'más', 'második',
                     'ismert', 'ismerték', 'ismert',
                     'lehet', 'létez', 'léte', 'létezik',
                     'kell', 'kell', 'kellett', 'kell-e',
                     'kell', 'kell', 'kell', 'kell',
                     'kell', 'kell', 'kell', 'kell',
                     'kell', 'kell', 'kell', 'kell',
                     'kell', 'kell', 'kell', 'kell',
                     'kell', 'kell', 'kell', 'kell',
                     'kell', 'ell',
                     'kell', 'kell', 'kell', 'kell'
                     }
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def fit(self, texts: List[str]):
        self.chunk_texts = texts
        tokenized_docs = [self._tokenize(t) for t in texts]
        all_tokens = set()
        for tokens in tokenized_docs:
            all_tokens.update(tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_tokens))}

        n_docs = len(texts)
        self.idf = {}
        for word in self.vocabulary:
            doc_count = sum(1 for tokens in tokenized_docs if word in tokens)
            self.idf[word] = math.log((n_docs + 1) / (doc_count + 1)) + 1

        self.document_vectors = []
        for tokens in tokenized_docs:
            vec = np.zeros(len(self.vocabulary))
            tf = Counter(tokens)
            total_tokens = max(len(tokens), 1)
            for word, count in tf.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    vec[idx] = (count / total_tokens) * self.idf[word]
            self.document_vectors.append(vec)

        norms = np.linalg.norm(self.document_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.document_vectors = self.document_vectors / norms

    def transform(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocabulary))
        tf = Counter(tokens)
        total_tokens = max(len(tokens), 1)
        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                vec[idx] = (count / total_tokens) * self.idf[word]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))


class RagSubgraph:
    """Moduláris RAG subgraph. Nem igényel Ollama-t vagy sentence-transformers-t."""

    def __init__(self, config: Optional[RagConfig] = None):
        self.config = config or RagConfig()
        self.vectorizer = TFIDFVectorizer()
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.query: Optional[str] = None
        self.retrieved_docs: List[Document] = []
        self.context: str = ""
        self.error: Optional[str] = None
        self._source_scores: dict = {}

    def load_documents(self) -> List[Document]:
        self.documents = []
        data_dir = self.config.data_dir
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Adatkönyvtár nem található: {data_dir}")
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                source = f"tudástár/{filename}"
                doc = Document(page_content=content, metadata={"source": source, "filename": filename})
                self.documents.append(doc)
        pass # print(f"[RAG] Betöltve {len(self.documents)} dokumentum.")
        return self.documents

    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        docs = documents or self.documents
        if not docs:
            docs = self.load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = splitter.split_documents(docs)
        pass # print(f"[RAG] Darabolva {len(self.chunks)} darabra.")
        return self.chunks

    def build_index(self, chunks: Optional[List[Document]] = None) -> TFIDFVectorizer:
        docs = chunks or self.chunks
        if not docs:
            docs = self.split_documents()
        texts = [doc.page_content for doc in docs]
        self.vectorizer.fit(texts)
        pass # print(f"[RAG] TF-IDF index kész: {len(self.vocabulary)} unikális szó, {len(self.document_vectors)} dokumentum.")
        return self.vectorizer

    @property
    def vocabulary(self) -> dict:
        return self.vectorizer.vocabulary

    @property
    def document_vectors(self) -> list:
        return self.vectorizer.document_vectors

    def _has_index(self) -> bool:
        return len(self.vectorizer.document_vectors) > 0

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self.config.search_k
        if not self._has_index():
            raise RuntimeError("Az index nincs inicializálva.")
        query_vec = self.vectorizer.transform(query)
        scores = []
        for i, doc_vec in enumerate(self.document_vectors):
            score = self.vectorizer.cosine_similarity(query_vec, doc_vec)
            source = self.chunks[i].metadata.get("source", "")
            # TF-IDF cosine similarity score
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(s[1] for s in scores)
            if max_score > 0:
                scores = [(i, s/max_score) for i, s in scores]
        top_k = scores[:k]
        self.retrieved_docs = [self.chunks[i] for i, _ in top_k]
        # Track per-source relevance
        self._source_scores = {}
        for i, s in top_k:
            src = self.chunks[i].metadata.get("source", "unknown")
            self._source_scores[src] = self._source_scores.get(src, 0) + s
        pass # print(f"[RAG] Keresett '{query}' -> {len(self.retrieved_docs)} eredmény.")
        return self.retrieved_docs

    def get_retrieval_stats(self) -> dict:
        """Visszaadja a dokumentum-forrásak szerű relevancia-összesítését."""
        return dict(sorted(self._source_scores.items(), key=lambda x: x[1], reverse=True))

    def get_context(self, retrieved_docs: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "ismeretlen")
            content = doc.page_content[:800]
            context_parts.append(f"[{source}]: {content}")
        self.context = "\n\n---\n\n".join(context_parts)
        return self.context

    def retrieve(self, query: str) -> dict:
        retrieved = self.search(query)
        context = self.get_context(retrieved)
        return {"retrieved_docs": retrieved, "context": context, "query": query, "num_results": len(retrieved)}

    def run_pipeline(self, query: str) -> DocumentState:
        state = DocumentState(documents=self.documents, chunks=self.chunks, query=query)
        try:
            if not self.documents:
                self.load_documents()
            if not self._has_index():
                self.split_documents()
                self.build_index()
            retrieved = self.search(query)
            context = self.get_context(retrieved)
            state.update({"retrieved_docs": retrieved, "context": context, "error": None})
        except Exception as e:
            state["error"] = str(e)
            print(f"[RAG] Hiba: {e}")
        return state


def run_rag_subgraph(query: str, config: Optional[RagConfig] = None) -> dict:
    rag = RagSubgraph(config)
    state = rag.run_pipeline(query)
    return {"query": query, "context": state.get("context", ""), "retrieved_docs": state.get("retrieved_docs", []), "error": state.get("error", None)}


if __name__ == "__main__":
    pass # print("RAG Subgraph teszt...")
    result = run_rag_subgraph("Mennyi a minimálbér 2024-ben?")
    pass # print(f"Kontextum: {result['context'][:300]}...")
    pass # print("✅ RAG Subgraph teszt kész!")
