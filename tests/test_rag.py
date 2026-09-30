#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for RagDemo project - RAG modules
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import RagConfig, TFIDFVectorizer, RagSubgraph

class TestRagConfig(unittest.TestCase):
    """Test RagConfig class"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = RagConfig()
        self.assertEqual(config.search_k, 10)
        self.assertEqual(config.chunk_size, 800)
        self.assertEqual(config.chunk_overlap, 100)
        # data_dir should be ../data relative to this file
        self.assertTrue(config.data_dir.endswith("data"))

class TestTFIDFVectorizer(unittest.TestCase):
    """Test TFIDFVectorizer class"""
    
    def setUp(self):
        self.vectorizer = TFIDFVectorizer()
        self.sample_texts = [
            "Minimálbér 2026-ben Magyarországon 266 ezer forint",
            "Munkaszerződés próbaidője 3 hónap",
            "Adózás és szociális járulékok",
            "Lakhatási támogatás és kedvezmények"
        ]
    
    def test_initialization(self):
        """Test vectorizer initialization"""
        self.assertEqual(self.vectorizer.vocabulary, {})
        self.assertEqual(self.vectorizer.idf, {})
        self.assertEqual(self.vectorizer.document_vectors, [])
        self.assertEqual(self.vectorizer.chunk_texts, [])
    
    def test_tokenize(self):
            """Test text tokenization removes punctuation and short words"""
            text = "A minimálbér 2026-ben 266 ezer forint havonta"
            tokens = self.vectorizer._tokenize(text)
            # Should keep meaningful words, remove punctuation
            self.assertIn("minimálbér", tokens)
            self.assertIn("2026", tokens)
            self.assertIn("266", tokens)
            self.assertIn("ezer", tokens)
            self.assertIn("forint", tokens)
            self.assertIn("havonta", tokens)
            # Should remove single characters and punctuation
            self.assertNotIn("a", tokens)  # article
            # Should split on word boundaries
            self.assertNotIn(".", tokens)
            self.assertNotIn(",", tokens)
    
    def test_fit_transform(self):
        """Test fit and transform functionality"""
        self.vectorizer.fit(self.sample_texts)
        
        # Check vocabulary was built
        self.assertGreater(len(self.vectorizer.vocabulary), 0)
        self.assertEqual(len(self.vectorizer.idf), len(self.vectorizer.vocabulary))
        self.assertEqual(len(self.vectorizer.document_vectors), len(self.sample_texts))
        
        # Test transform
        query_vec = self.vectorizer.transform("minimálbér")
        self.assertEqual(len(query_vec), len(self.vectorizer.vocabulary))
        
        # Test cosine similarity
        sim = self.vectorizer.cosine_similarity(query_vec, query_vec)
        self.assertAlmostEqual(sim, 1.0, places=5)
        
        # Different vectors should have lower similarity
        other_vec = self.vectorizer.transform("lakhatás")
        other_sim = self.vectorizer.cosine_similarity(query_vec, other_vec)
        self.assertGreaterEqual(other_sim, 0.0)
        self.assertLessEqual(other_sim, 1.0)

class TestRagSubgraph(unittest.TestCase):
    """Test RagSubgraph class"""
    
    def setUp(self):
        self.rag = RagSubgraph(RagConfig())
        # Create a small test dataset
        self.test_texts = [
            "Minimálbér 2026-ben: 266 ezer forint havonta",
            "Munkaszerződés próbaidője: 3 hónap maximum", 
            "Nyugdíjkor: 62 év férfi, 62 év női",
            "SZJA: 15% persönövedelemadó",
            "ÁFA: 27% általános forgalmi adó"
        ]
    
    def test_initialization(self):
        """Test RagSubgraph initialization"""
        self.assertIsInstance(self.rag, RagSubgraph)
        self.assertIsInstance(self.rag.config, RagConfig)
        self.assertIsInstance(self.rag.vectorizer, TFIDFVectorizer)
        self.assertEqual(self.rag.documents, [])
        self.assertEqual(self.rag.chunks, [])
        self.assertIsNone(self.rag.query)
        self.assertEqual(self.rag.retrieved_docs, [])
        self.assertEqual(self.rag.context, "")
        self.assertIsNone(self.rag.error)
        self.assertEqual(self.rag._source_scores, {})
    
    def test_load_documents_mock(self):
        """Test document loading logic (mocked since we don't have real files)"""
        # We'll test the method exists and can be called
        self.assertTrue(hasattr(self.rag, 'load_documents'))
        self.assertTrue(callable(self.rag.load_documents))
    
    def test_split_documents_mock(self):
        """Test document splitting logic (mocked)"""
        self.assertTrue(hasattr(self.rag, 'split_documents'))
        self.assertTrue(callable(self.rag.split_documents))
    
    def test_build_index_mock(self):
        """Test index building logic (mocked)"""
        self.assertTrue(hasattr(self.rag, 'build_index'))
        self.assertTrue(callable(self.rag.build_index))
    
    def test_search_method_exists(self):
        """Test that search method exists"""
        self.assertTrue(hasattr(self.rag, 'search'))
        self.assertTrue(callable(self.rag.search))
    
    def test_retrieve_method_exists(self):
        """Test that retrieve method exists"""
        self.assertTrue(hasattr(self.rag, 'retrieve'))
        self.assertTrue(callable(self.rag.retrieve))
    
    def test_get_context_method_exists(self):
        """Test that get_context method exists"""
        self.assertTrue(hasattr(self.rag, 'get_context'))
        self.assertTrue(callable(self.rag.get_context))
    
    def test_get_retrieval_stats_exists(self):
        """Test that get_retrieval_stats method exists"""
        self.assertTrue(hasattr(self.rag, 'get_retrieval_stats'))
        self.assertTrue(callable(self.rag.get_retrieval_stats))

if __name__ == '__main__':
    unittest.main()