"""
Simple RAG implementation for PDF analysis using Ollama.
Simplified version optimized for symbol recognition and diagram analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class SimplePDFChat:
    """Simple RAG chatbot for PDF documents."""

    def __init__(
        self,
        pdf_path: str,
        llm_model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
    ):
        """
        Initialize PDF chatbot.

        Args:
            pdf_path: Path to PDF file
            llm_model: Ollama LLM model name
            embed_model: Ollama embedding model name
        """
        self.pdf_path = pdf_path
        self.base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"

        # Load PDF
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(self.chunks)} chunks")

        # Create embeddings
        print("Creating embeddings...")
        embeddings = OllamaEmbeddings(model=embed_model, base_url=self.base_url)

        # Create vector store
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        print("Vector store created")

        # Create LLM
        self.llm = Ollama(model=llm_model, base_url=self.base_url)

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
        )

    def ask(self, question: str) -> str:
        """
        Ask a question about the PDF.

        Args:
            question: Question to ask

        Returns:
            Answer from the LLM
        """
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

    def extract_symbol_legend(self) -> Dict[str, Any]:
        """Extract symbol legend from the PDF."""
        question = """
        Look at the first few pages of this document.
        Extract all symbol definitions from the symbol legend.

        For each symbol, provide:
        - Symbol name
        - Description
        - Reference code (if any)

        Format the response as JSON with this structure:
        {
            "symbols": [
                {"name": "...", "description": "...", "reference": "..."}
            ]
        }
        """
        response = self.ask(question)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "symbols": [],
                "raw_response": response,
                "error": "Failed to parse JSON"
            }

    def extract_page_names(self) -> List[str]:
        """Extract names/titles of all pages."""
        question = """
        List all the page titles or section names in this document.
        Return just the list of titles, one per line.
        """
        response = self.ask(question)
        return [line.strip() for line in response.split('\n') if line.strip()]

    def find_symbol_instances(self, symbol_name: str) -> str:
        """Find all instances where a symbol is mentioned."""
        question = f"""
        Find all mentions of the "{symbol_name}" symbol in this document.
        For each mention, provide:
        - Page number
        - Context (what's happening around it)
        """
        return self.ask(question)


class SymbolComparator:
    """Compare symbols between two PDFs using LLM understanding."""

    def __init__(
        self,
        pdf1_path: str,
        pdf2_path: str,
        llm_model: str = "llama3.2",
        base_url: Optional[str] = None
    ):
        """
        Initialize symbol comparator.

        Args:
            pdf1_path: Path to first PDF (old version)
            pdf2_path: Path to second PDF (new version)
            llm_model: Ollama model to use
        """
        self.base_url = base_url or os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.llm = Ollama(model=llm_model, base_url=self.base_url)
        self.pdf1_chat = SimplePDFChat(pdf1_path, llm_model, base_url=self.base_url)
        self.pdf2_chat = SimplePDFChat(pdf2_path, llm_model, base_url=self.base_url)

        # Extract legends from both
        print("Extracting legends...")
        self.legend1 = self.pdf1_chat.extract_symbol_legend()
        self.legend2 = self.pdf2_chat.extract_symbol_legend()

    def compare_symbols(self) -> str:
        """Compare symbols between the two PDFs."""
        prompt = f"""
        Compare the symbols between these two documents:

        Document 1 symbols:
        {json.dumps(self.legend1, indent=2)}

        Document 2 symbols:
        {json.dumps(self.legend2, indent=2)}

        Identify:
        1. New symbols (in document 2 but not in document 1)
        2. Removed symbols (in document 1 but not in document 2)
        3. Changed symbols (present in both but with different descriptions)
        4. Unchanged symbols

        Provide a clear summary of the differences.
        """

        return self.llm.invoke(prompt)

    def is_same_symbol(self, bbox1: tuple, context1: str, bbox2: tuple, context2: str) -> Dict[str, Any]:
        """
        Determine if two symbols are the same, regardless of size.

        Args:
            bbox1: Bounding box from first PDF (x0, y0, x1, y1)
            context1: Text context around first symbol
            bbox2: Bounding box from second PDF
            context2: Text context around second symbol

        Returns:
            Dictionary with match result
        """
        # Calculate sizes
        width1, height1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        width2, height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        prompt = f"""
        Are these two symbols the same, even though they may be different sizes?

        Symbol 1:
        - Size: {width1:.1f} x {height1:.1f} points
        - Context: {context1}

        Symbol 2:
        - Size: {width2:.1f} x {height2:.1f} points
        - Context: {context2}

        Consider:
        - Do they represent the same thing semantically?
        - Are they in similar positions (same general area)?
        - Is the context similar?

        Respond in JSON format:
        {{
            "are_same": true or false,
            "confidence": 0.0 to 1.0,
            "reasoning": "brief explanation"
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "are_same": False,
                "confidence": 0.0,
                "reasoning": "Failed to parse response",
                "raw_response": response
            }


# Quick helper functions
def chat_with_pdf(pdf_path: str, *, base_url: Optional[str] = None) -> SimplePDFChat:
    """
    Quick setup to chat with a PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        SimplePDFChat instance

    Example:
        >>> chat = chat_with_pdf("diagram.pdf")
        >>> print(chat.ask("What symbols are in the legend?"))
    """
    return SimplePDFChat(pdf_path, base_url=base_url)


def compare_pdfs(old_pdf: str, new_pdf: str, *, base_url: Optional[str] = None) -> SymbolComparator:
    """
    Quick setup to compare two PDFs.

    Args:
        old_pdf: Path to old PDF
        new_pdf: Path to new PDF

    Returns:
        SymbolComparator instance

    Example:
        >>> comp = compare_pdfs("old.pdf", "new.pdf")
        >>> print(comp.compare_symbols())
    """
    return SymbolComparator(old_pdf, new_pdf, base_url=base_url)


def interactive_pdf_chat(pdf_path: str, *, base_url: Optional[str] = None):
    """
    Interactive command-line chat with a PDF.

    Args:
        pdf_path: Path to PDF file

    Example:
        >>> interactive_pdf_chat("diagram.pdf")
        Ask a question (or 'quit'): What valves are shown?
        Answer: The document shows...
    """
    chat = SimplePDFChat(pdf_path, base_url=base_url)

    print(f"\nChatting with: {pdf_path}")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        # Special commands
        if question.lower() == 'legend':
            print("\nExtracting symbol legend...")
            legend = chat.extract_symbol_legend()
            print(json.dumps(legend, indent=2))
            continue

        if question.lower() == 'pages':
            print("\nExtracting page names...")
            pages = chat.extract_page_names()
            for i, page in enumerate(pages, 1):
                print(f"{i}. {page}")
            continue

        # Regular question
        print("\nThinking...")
        answer = chat.ask(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Interactive chat:  python rag_simple.py <pdf_file>")
        print("  Compare PDFs:      python rag_simple.py <old_pdf> <new_pdf>")
        print("\nSpecial commands in interactive mode:")
        print("  legend  - Extract symbol legend")
        print("  pages   - List page names")
        print("  quit    - Exit")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Single PDF - interactive chat
        interactive_pdf_chat(sys.argv[1])
    else:
        # Two PDFs - comparison
        old_pdf, new_pdf = sys.argv[1], sys.argv[2]
        print(f"Comparing:\n  Old: {old_pdf}\n  New: {new_pdf}\n")

        comparator = compare_pdfs(old_pdf, new_pdf)
        print("\n" + "="*60)
        print("SYMBOL COMPARISON")
        print("="*60)
        print(comparator.compare_symbols())
