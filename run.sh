#!/bin/bash
# Run script for RagDemo
# Usage: ./run.sh

set -e

echo "🚀 RagDemo Agentic RAG Chatbot - Start"
echo "======================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️ Ollama nem fut! Indítás..."
    ollama serve &
    sleep 3
    echo "✅ Ollama indítva"
fi

# Check if model is available
if ! ollama list 2>/dev/null | grep -q "glm4:9b"; then
    echo "📥 glm4:9b letöltése..."
    ollama pull glm4:9b
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment létrehozása..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt --quiet
else
    source venv/bin/activate
fi

# Run Streamlit
echo "🎨 Streamlit UI indulása..."
echo "🌐 Nyisd meg: http://localhost:8501"
streamlit run ui/app.py --server.port=8501 --server.headless=true
