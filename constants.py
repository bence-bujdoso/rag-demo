#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants module for RagDemo project.
Centralizes hardcoded values for easier configuration and maintenance.
"""

# RAG Configuration
RAG_SEARCH_K = 10
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 100

# Embedding Model
EMBEDDING_MODEL = "karsar/paraphrase-multilingual-MiniLM-L12-hu-v3"

# RRF Parameters
RRF_K = 60
ALPHA = 0.5  # Weight for dense vs TF-IDF in weighted fusion

# Model Configuration
LLM_MODEL = "glm4:9b"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096

# Evaluation Thresholds
EVALUATION_PASS_THRESHOLD = 6.0  # Out of 10

# UI Configuration
UI_PORT = 8501
UI_HOST = "localhost"