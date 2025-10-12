#!/bin/bash
# Docker container integration test
# Verifies all components are working correctly

set -e

echo "=================================="
echo "  PDF Compare - Container Test"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
test_step() {
    local name=$1
    local command=$2

    echo ""
    echo -n "Testing: $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo ""
echo "1. Python Environment"
echo "-----------------------------------"
test_step "Python version (3.11+)" "python --version | grep -q 'Python 3.1[1-9]'"
test_step "pip installed" "pip --version"

echo ""
echo "2. Core Python Packages"
echo "-----------------------------------"
test_step "pymupdf" "python -c 'import fitz'"
test_step "shapely" "python -c 'import shapely'"
test_step "numpy" "python -c 'import numpy'"
test_step "streamlit" "python -c 'import streamlit'"
test_step "opencv" "python -c 'import cv2'"
test_step "pytesseract" "python -c 'import pytesseract'"
test_step "sqlalchemy" "python -c 'import sqlalchemy'"

echo ""
echo "3. RAG/AI Packages"
echo "-----------------------------------"
test_step "langchain" "python -c 'import langchain'"
test_step "langchain_community" "python -c 'import langchain_community'"
test_step "chromadb" "python -c 'import chromadb'"
test_step "pypdf" "python -c 'import pypdf'"

echo ""
echo "4. Project Modules"
echo "-----------------------------------"
test_step "pdf_compare module" "python -c 'import pdf_compare'"
test_step "pdf_extract_server" "python -c 'from pdf_compare.pdf_extract_server import pdf_to_vectormap_server'"
test_step "rag_simple" "python -c 'from pdf_compare.rag_simple import SimplePDFChat'"
test_step "db_backend" "python -c 'from pdf_compare.db_backend import DatabaseBackend'"

echo ""
echo "5. System Dependencies"
echo "-----------------------------------"
test_step "tesseract-ocr" "which tesseract"
test_step "curl" "which curl"
test_step "sqlite3" "which sqlite3"

echo ""
echo "6. Ollama Connection"
echo "-----------------------------------"

# Check if OLLAMA_HOST is set
if [ -z "$OLLAMA_HOST" ]; then
    echo -e "${YELLOW}⚠ OLLAMA_HOST not set, using default${NC}"
    export OLLAMA_HOST="http://localhost:11434"
fi

echo "   Ollama host: $OLLAMA_HOST"

if test_step "Ollama service reachable" "curl -f -s $OLLAMA_HOST/api/version"; then
    test_step "llama3.2 model available" "curl -s $OLLAMA_HOST/api/tags | grep -q llama3.2"
    test_step "nomic-embed-text model available" "curl -s $OLLAMA_HOST/api/tags | grep -q nomic-embed-text"

    # Test Ollama LLM
    if test_step "Ollama LLM functional" "python -c 'from langchain_community.llms import Ollama; llm = Ollama(model=\"llama3.2\"); print(llm.invoke(\"test\"))'"; then
        echo "   LLM response successful"
    fi

    # Test embeddings
    if test_step "Ollama embeddings functional" "python -c 'from langchain_community.embeddings import OllamaEmbeddings; e = OllamaEmbeddings(model=\"nomic-embed-text\"); print(len(e.embed_query(\"test\")))'"; then
        echo "   Embeddings response successful"
    fi
else
    echo -e "${RED}   ✗ Cannot reach Ollama service${NC}"
    echo "   Make sure Ollama container is running"
    ((TESTS_FAILED+=3))
fi

echo ""
echo "7. Database Connection"
echo "-----------------------------------"

if [ -n "$DATABASE_URL" ]; then
    echo "   Database URL: ${DATABASE_URL%%@*}@***"  # Hide credentials
    test_step "Database connection" "python -c 'from sqlalchemy import create_engine; engine = create_engine(\"$DATABASE_URL\"); engine.connect()'"
else
    echo -e "${YELLOW}   ⚠ DATABASE_URL not set, skipping database tests${NC}"
fi

echo ""
echo "=================================="
echo "  Test Summary"
echo "=================================="
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Container is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Access UI: http://localhost:8501"
    echo "  - Run RAG test: docker exec pdf-compare-ui python test_rag.py"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check the errors above.${NC}"
    exit 1
fi
