"""Base classes for RAG system."""

import os
import re
import math
from typing import List, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
from langchain_core.documents import Document


class DocumentState(dict):
    """Az RAG belső állapota."""
    pass


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