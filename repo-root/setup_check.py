#!/usr/bin/env python3
"""
Setup verification and guided first-run script.
Checks that all dependencies are installed and walks through first analysis.
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.11+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.11+)"


def check_package(package: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        __import__(package)
        return True, "Installed"
    except ImportError:
        return False, "Missing"


def check_ollama() -> Tuple[bool, str]:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, version
        return False, "Not responding"
    except FileNotFoundError:
        return False, "Not installed"
    except Exception as e:
        return False, str(e)


def check_ollama_model(model: str) -> Tuple[bool, str]:
    """Check if an Ollama model is available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and model in result.stdout:
            return True, "Available"
        return False, "Not pulled"
    except Exception as e:
        return False, str(e)


def print_header(text: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_check(name: str, status: bool, detail: str):
    """Print a check result."""
    symbol = "[OK]" if status else "[!!]"
    color = "\033[92m" if status else "\033[91m"  # Green or red
    reset = "\033[0m"
    print(f"{color}{symbol:6}{reset} {name:30} {detail}")


def run_checks() -> dict:
    """Run all system checks and return results."""
    results = {}

    print_header("System Requirements")

    # Python version
    ok, detail = check_python_version()
    print_check("Python Version", ok, detail)
    results['python'] = ok

    print_header("Core Dependencies")

    # Core packages
    packages = [
        ('pymupdf', 'PyMuPDF (PDF processing)'),
        ('shapely', 'Shapely (geometry)'),
        ('numpy', 'NumPy (numerical)'),
        ('streamlit', 'Streamlit (UI)'),
        ('cv2', 'OpenCV (raster comparison)'),
        ('pytesseract', 'Tesseract (OCR)'),
        ('sqlalchemy', 'SQLAlchemy (database)'),
    ]

    all_core_ok = True
    for pkg, desc in packages:
        ok, detail = check_package(pkg)
        print_check(desc, ok, detail)
        if not ok:
            all_core_ok = False

    results['core_packages'] = all_core_ok

    print_header("AI/RAG Dependencies")

    # RAG packages
    rag_packages = [
        ('langchain', 'LangChain'),
        ('langchain_community', 'LangChain Community'),
        ('chromadb', 'ChromaDB'),
        ('pypdf', 'PyPDF'),
    ]

    all_rag_ok = True
    for pkg, desc in rag_packages:
        ok, detail = check_package(pkg)
        print_check(desc, ok, detail)
        if not ok:
            all_rag_ok = False

    results['rag_packages'] = all_rag_ok

    print_header("Ollama Setup")

    # Ollama
    ok, detail = check_ollama()
    print_check("Ollama Installation", ok, detail)
    results['ollama'] = ok

    if ok:
        # Check models
        ok_llm, detail_llm = check_ollama_model("llama3.2")
        print_check("LLM Model (llama3.2)", ok_llm, detail_llm)
        results['llm_model'] = ok_llm

        ok_embed, detail_embed = check_ollama_model("nomic-embed-text")
        print_check("Embed Model (nomic-embed-text)", ok_embed, detail_embed)
        results['embed_model'] = ok_embed
    else:
        results['llm_model'] = False
        results['embed_model'] = False

    return results


def print_summary(results: dict):
    """Print summary and next steps."""
    print_header("Summary")

    all_ok = all(results.values())

    if all_ok:
        print("\n✓ All checks passed! System is ready to use.\n")
        print("Next steps:")
        print("  1. Test RAG with a PDF:")
        print("     python test_rag.py your_diagram.pdf")
        print("\n  2. Launch Streamlit UI:")
        print("     streamlit run ui/streamlit_app.py")
        print("\n  3. Try interactive PDF chat:")
        print("     python pdf_compare/rag_simple.py your_diagram.pdf")
        return True
    else:
        print("\n× Some components are missing. Follow the instructions below:\n")

        if not results.get('python'):
            print("1. Install Python 3.11 or newer:")
            print("   https://www.python.org/downloads/")

        if not results.get('core_packages'):
            print("\n2. Install core dependencies:")
            print("   pip install -r requirements.txt")

        if not results.get('rag_packages'):
            print("\n3. Install RAG dependencies (included in requirements.txt):")
            print("   pip install -r requirements.txt")

        if not results.get('ollama'):
            print("\n4. Install Ollama:")
            print("   See: INSTALL_OLLAMA_WINDOWS.md")
            print("   Quick: https://ollama.com/download/windows")

        if results.get('ollama') and not results.get('llm_model'):
            print("\n5. Pull LLM model:")
            print("   ollama pull llama3.2")

        if results.get('ollama') and not results.get('embed_model'):
            print("\n6. Pull embedding model:")
            print("   ollama pull nomic-embed-text")

        print("\nAfter fixing the issues, run this script again:")
        print("  python setup_check.py")
        return False


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  PDF COMPARE - SETUP VERIFICATION")
    print("=" * 70)
    print("\nThis script will check if your system is ready to use PDF Compare.")
    print("It will verify:")
    print("  - Python version")
    print("  - Required packages")
    print("  - Ollama installation")
    print("  - AI models")

    results = run_checks()
    all_ok = print_summary(results)

    print("\n" + "=" * 70)
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
