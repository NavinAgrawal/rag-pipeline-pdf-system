#!/bin/bash
# Clean launch script for RAG Pipeline PDF System
# Filters out PyTorch/Streamlit compatibility warnings for cleaner development experience

echo "🚀 Starting RAG Pipeline PDF System..."
echo "   App will be available at: http://localhost:8501"
echo ""

streamlit run enhanced_rag_demo_app.py 2>&1 | grep -v -E "(torch\.classes|RuntimeError|Traceback|_get_custom_class_python_wrapper|__path__\._path|Examining the path)"
