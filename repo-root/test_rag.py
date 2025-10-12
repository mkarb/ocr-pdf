#!/usr/bin/env python3
"""
Quick test script for RAG PDF analysis.
Run this to verify Ollama and RAG are working correctly.
"""

import sys
from pathlib import Path


def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    print("Testing Ollama connection...")
    try:
        from langchain_community.llms import Ollama

        llm = Ollama(model="llama3.2")
        response = llm.invoke("Say 'OK' if you can hear me.")

        if "ok" in response.lower():
            print("  ✓ Ollama connection successful")
            return True
        else:
            print(f"  ✗ Unexpected response: {response}")
            return False
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        print("\n  Please ensure:")
        print("    1. Ollama is installed (https://ollama.com)")
        print("    2. Run: ollama pull llama3.2")
        print("    3. Run: ollama pull nomic-embed-text")
        return False


def test_embeddings():
    """Test if embedding model is available."""
    print("\nTesting embedding model...")
    try:
        from langchain_community.embeddings import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_text = ["This is a test"]
        result = embeddings.embed_documents(test_text)

        if result and len(result) > 0:
            print(f"  ✓ Embeddings working (dimension: {len(result[0])})")
            return True
        else:
            print("  ✗ No embeddings generated")
            return False
    except Exception as e:
        print(f"  ✗ Embedding test failed: {e}")
        print("\n  Run: ollama pull nomic-embed-text")
        return False


def test_pdf_chat(pdf_path: str):
    """Test chatting with a PDF."""
    print(f"\nTesting PDF chat with: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"  ✗ PDF not found: {pdf_path}")
        return False

    try:
        from pdf_compare.rag_simple import SimplePDFChat

        print("  Loading PDF and creating embeddings...")
        chat = SimplePDFChat(pdf_path)

        print("  Asking test question...")
        answer = chat.ask("What is this document about? Give a one sentence summary.")

        print(f"\n  Answer: {answer}")
        print("\n  ✓ PDF chat working!")
        return True
    except Exception as e:
        print(f"  ✗ PDF chat failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_symbol_extraction(pdf_path: str):
    """Test symbol legend extraction."""
    print(f"\nTesting symbol extraction from: {pdf_path}")

    if not Path(pdf_path).exists():
        print(f"  ✗ PDF not found: {pdf_path}")
        return False

    try:
        from pdf_compare.rag_simple import SimplePDFChat

        chat = SimplePDFChat(pdf_path)

        print("  Extracting symbol legend...")
        legend = chat.extract_symbol_legend()

        if legend.get('symbols'):
            print(f"\n  Found {len(legend['symbols'])} symbols:")
            for symbol in legend['symbols'][:5]:  # Show first 5
                print(f"    - {symbol.get('name', 'Unknown')}: {symbol.get('description', 'No description')}")

            if len(legend['symbols']) > 5:
                print(f"    ... and {len(legend['symbols']) - 5} more")

            print("\n  ✓ Symbol extraction working!")
            return True
        else:
            print("  ℹ No symbols found (document may not have a legend)")
            print(f"  Raw response: {legend.get('raw_response', 'N/A')[:200]}")
            return True  # Not an error, just no legend
    except Exception as e:
        print(f"  ✗ Symbol extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(pdf_path: str = None):
    """Run all tests."""
    print("="*60)
    print("RAG PDF ANALYSIS - TEST SUITE")
    print("="*60)

    results = []

    # Test 1: Ollama connection
    results.append(("Ollama Connection", test_ollama_connection()))

    # Test 2: Embeddings
    results.append(("Embedding Model", test_embeddings()))

    # Test 3 & 4: PDF tests (if PDF provided)
    if pdf_path:
        results.append(("PDF Chat", test_pdf_chat(pdf_path)))
        results.append(("Symbol Extraction", test_symbol_extraction(pdf_path)))
    else:
        print("\n⚠ No PDF provided - skipping PDF tests")
        print("  Usage: python test_rag.py <pdf_file>")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! RAG system is ready to use.")
        return True
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = None

    success = run_all_tests(pdf_file)
    sys.exit(0 if success else 1)
