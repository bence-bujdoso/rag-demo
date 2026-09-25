"""Hibrid RAG rendszer: TF-IDF + Dense Vector Search (RRF fusion)."""

import os
import math
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter

import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rag.base import TFIDFVectorizer, RagConfig


logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Hibrid RAG konfiguráció."""
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "data")
    chunk_size: int = 800
    chunk_overlap: int = 100
    search_k: int = 10
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    rrf_k: int = 60
    alpha: float = 0.5
    use_rrf: bool = True


class DenseVectorStore:
    """FAISS alapú sűrű vektortár."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_texts: List[str] = []
        self.chunk_metadatas: List[Dict] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict]):
        """Vektorok hozzáadása az indexhez."""
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.chunk_texts.extend(texts)
        self.chunk_metadatas.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Keresés az indexben."""
        if self.index.ntotal == 0:
            return []
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def __len__(self) -> int:
        return self.index.ntotal


class HybridRAG:
    """Hibrid RAG: TF-IDF + Dense Vector Search RRF-fel."""

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()
        self.tfidf_vectorizer = TFIDFVectorizer()
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_store: Optional[DenseVectorStore] = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self._initialized = False

    def _load_embedding_model(self) -> SentenceTransformer:
        """Embedding modell betöltése (lazy loading)."""
        if self.embedding_model is None:
            logger.info(f"Embedding modell betöltése: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        return self.embedding_model

    def load_documents(self) -> List[Document]:
        """Dokumentumok betöltése."""
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
        logger.info(f"[HybridRAG] Betöltve {len(self.documents)} dokumentum.")
        return self.documents

    def split_documents(self, documents: Optional[List[Document]] = None) -> List[Document]:
        """Szövegrészletekre bontás."""
        docs = documents or self.documents
        if not docs:
            docs = self.load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = splitter.split_documents(docs)
        logger.info(f"[HybridRAG] Darabolva {len(self.chunks)} darabra.")
        return self.chunks

    def build_tfidf_index(self, chunks: Optional[List[Document]] = None):
        """TF-IDF index építése."""
        docs = chunks or self.chunks
        if not docs:
            docs = self.split_documents()
        texts = [doc.page_content for doc in docs]
        self.tfidf_vectorizer.fit(texts)
        logger.info(f"[HybridRAG] TF-IDF index kész: {len(self.tfidf_vectorizer.vocabulary)} unikális szó.")

    def build_dense_index(self, chunks: Optional[List[Document]] = None):
        """Sűrű vektor index építése (FAISS + sentence-transformers)."""
        docs = chunks or self.chunks
        if not docs:
            docs = self.split_documents()

        model = self._load_embedding_model()
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        logger.info(f"[HybridRAG] Embeddingek generálása {len(texts)} szövegre...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        self.vector_store = DenseVectorStore(dimension)
        self.vector_store.add(embeddings, texts, metadatas)
        logger.info(f"[HybridRAG] FAISS index kész: {len(self.vector_store)} vektor, dimenzió: {dimension}.")

    def initialize(self):
        """Teljes inicializálás: dokumentumok, chunking, mindkét index."""
        if self._initialized:
            return
        self.load_documents()
        self.split_documents()
        self.build_tfidf_index()
        self.build_dense_index()
        self._initialized = True

    def tfidf_search(self, query: str, k: Optional[int] = None) -> List[Tuple[int, float, Document]]:
        """TF-IDF alapú keresés."""
        k = k or self.config.search_k
        if not hasattr(self.tfidf_vectorizer, 'document_vectors') or len(self.tfidf_vectorizer.document_vectors) == 0:
            raise RuntimeError("TF-IDF index nincs inicializálva.")

        query_vec = self.tfidf_vectorizer.transform(query)
        scores = []
        for i, doc_vec in enumerate(self.tfidf_vectorizer.document_vectors):
            score = self.tfidf_vectorizer.cosine_similarity(query_vec, doc_vec)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        if scores:
            max_score = max(s[1] for s in scores)
            if max_score > 0:
                scores = [(i, s / max_score) for i, s in scores]

        top_k = scores[:k]
        results = [(idx, score, self.chunks[idx]) for idx, score in top_k]
        return results

    def dense_search(self, query: str, k: Optional[int] = None) -> List[Tuple[int, float, Document]]:
        """Sűrű vektor alapú keresés (FAISS)."""
        k = k or self.config.search_k
        if self.vector_store is None or len(self.vector_store) == 0:
            raise RuntimeError("Dense index nincs inicializálva.")

        model = self._load_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        results = self.vector_store.search(query_embedding, k)

        dense_results = []
        for idx, score in results:
            if idx < len(self.chunks):
                dense_results.append((idx, score, self.chunks[idx]))
        return dense_results

    async def async_tfidf_search(self, query: str, k: Optional[int] = None) -> List[Tuple[int, float, Document]]:
        """Aszinkron TF-IDF keresés."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.tfidf_search, query, k)

    async def async_dense_search(self, query: str, k: Optional[int] = None) -> List[Tuple[int, float, Document]]:
        """Aszinkron Dense keresés."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.dense_search, query, k)

    async def parallel_search(self, query: str, k: Optional[int] = None) -> Tuple[List[Tuple[int, float, Document]], List[Tuple[int, float, Document]]]:
        """Párhuzamos keresés mindkét indexben."""
        tfidf_task = self.async_tfidf_search(query, k)
        dense_task = self.async_dense_search(query, k)
        tfidf_results, dense_results = await asyncio.gather(tfidf_task, dense_task)
        return tfidf_results, dense_results

    def rrf_fusion(self, tfidf_results: List[Tuple[int, float, Document]],
                   dense_results: List[Tuple[int, float, Document]],
                   k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF) kombinálás."""
        k = k or self.config.search_k
        rrf_k = self.config.rrf_k

        tfidf_ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(tfidf_results)}
        dense_ranks = {idx: rank + 1 for rank, (idx, _, _) in enumerate(dense_results)}

        all_indices = set(tfidf_ranks.keys()) | set(dense_ranks.keys())

        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in tfidf_ranks:
                score += 1.0 / (rrf_k + tfidf_ranks[idx])
            if idx in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[idx])
            rrf_scores[idx] = score

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_indices[:k]

        results = [(self.chunks[idx], score) for idx, score in top_k]
        return results

    def weighted_fusion(self, tfidf_results: List[Tuple[int, float, Document]],
                        dense_results: List[Tuple[int, float, Document]],
                        k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Súlyozott kombinálás (alpha paraméter)."""
        k = k or self.config.search_k
        alpha = self.config.alpha

        tfidf_scores = {idx: score for idx, score, _ in tfidf_results}
        dense_scores = {idx: score for idx, score, _ in dense_results}

        all_indices = set(tfidf_scores.keys()) | set(dense_scores.keys())

        combined_scores = {}
        for idx in all_indices:
            tfidf_s = tfidf_scores.get(idx, 0.0)
            dense_s = dense_scores.get(idx, 0.0)
            combined_scores[idx] = alpha * dense_s + (1 - alpha) * tfidf_s

        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_indices[:k]

        results = [(self.chunks[idx], score) for idx, score in top_k]
        return results

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Hibrid keresés (RRF vagy súlyozott)."""
        k = k or self.config.search_k
        if not self._initialized:
            self.initialize()

        tfidf_results = self.tfidf_search(query, k)
        dense_results = self.dense_search(query, k)

        if self.config.use_rrf:
            fused = self.rrf_fusion(tfidf_results, dense_results, k)
        else:
            fused = self.weighted_fusion(tfidf_results, dense_results, k)

        return [doc for doc, _ in fused]

    async def async_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Aszinkron hibrid keresés."""
        k = k or self.config.search_k
        if not self._initialized:
            self.initialize()

        tfidf_results, dense_results = await self.parallel_search(query, k)

        if self.config.use_rrf:
            fused = self.rrf_fusion(tfidf_results, dense_results, k)
        else:
            fused = self.weighted_fusion(tfidf_results, dense_results, k)

        return [doc for doc, _ in fused]

    def get_context(self, retrieved_docs: List[Document]) -> str:
        """Kontextum összeállítása a retrievált dokumentumokból."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "ismeretlen")
            content = doc.page_content[:800]
            context_parts.append(f"[{source}]: {content}")
        return "\n\n---\n\n".join(context_parts)

    def retrieve(self, query: str) -> Dict:
        """Teljes retrieval folyamat."""
        if not self._initialized:
            self.initialize()
        retrieved = self.search(query)
        context = self.get_context(retrieved)
        return {"retrieved_docs": retrieved, "context": context, "query": query, "num_results": len(retrieved)}


def create_hybrid_rag(config: Optional[HybridConfig] = None) -> HybridRAG:
    """Factory függvény HybridRAG létrehozásához."""
    return HybridRAG(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = create_hybrid_rag()
    rag.initialize()

    query = "Mennyi a minimálbér 2024-ben?"
    print(f"Kérdés: {query}")

    import time
    start = time.time()
    result = rag.retrieve(query)
    elapsed = time.time() - start
    print(f"Keresési idő: {elapsed:.3f}s")
    print(f"Találatok: {result['num_results']}")
    print(f"Kontextum hossza: {len(result['context'])} karakter")