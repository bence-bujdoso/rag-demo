#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for RagDemo project - Configuration and Constants
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from constants import *

class TestConstants(unittest.TestCase):
    """Test configuration constants"""
    
    def test_rag_configuration(self):
        """Test RAG configuration constants"""
        self.assertEqual(RAG_SEARCH_K, 10)
        self.assertEqual(RAG_CHUNK_SIZE, 800)
        self.assertEqual(RAG_CHUNK_OVERLAP, 100)
    
    def test_embedding_model(self):
        """Test embedding model constant"""
        self.assertEqual(EMBEDDING_MODEL, "karsar/paraphrase-multilingual-MiniLM-L12-hu-v3")
    
    def test_rrf_parameters(self):
        """Test RRF fusion parameters"""
        self.assertEqual(RRF_K, 60)
        self.assertEqual(ALPHA, 0.5)
    
    def test_model_configuration(self):
        """Test LLM model configuration"""
        self.assertEqual(LLM_MODEL, "glm4:9b")
        self.assertEqual(LLM_TEMPERATURE, 0.7)
        self.assertEqual(LLM_MAX_TOKENS, 4096)
    
    def test_evaluation_thresholds(self):
        """Test evaluation threshold"""
        self.assertEqual(EVALUATION_PASS_THRESHOLD, 6.0)
    
    def test_ui_configuration(self):
        """Test UI configuration"""
        self.assertEqual(UI_PORT, 8501)
        self.assertEqual(UI_HOST, "localhost")

if __name__ == '__main__':
    unittest.main()