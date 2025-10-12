"""
RAG-based symbol recognition for engineering diagrams.
Uses LangChain + Ollama for intelligent PDF analysis and symbol matching.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class SymbolLegendExtractor:
    """Extract and parse symbol legend from PDF first pages."""

    def __init__(self, llm_model: str = "llama3.2"):
        """
        Initialize symbol legend extractor.

        Args:
            llm_model: Ollama model to use (default: llama3.2)
        """
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)

    def extract_legend_from_page(self, pdf_path: str, page_number: int = 0) -> Dict[str, Any]:
        """
        Extract symbol legend from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (0-based) containing legend

        Returns:
            Dictionary with legend information
        """
        # Extract text from the page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        text = page.get_text()

        # Extract images/diagrams from the page
        images = []
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "index": img_index,
                "width": base_image["width"],
                "height": base_image["height"],
                "colorspace": base_image["colorspace"]
            })

        doc.close()

        # Use LLM to identify legend sections
        prompt = f"""
        Analyze this text from a PDF page that may contain a symbol legend.
        Extract all symbol definitions, their names, and descriptions.

        Text:
        {text}

        Please identify:
        1. Symbol names (e.g., "Valve", "Pump", "Tank")
        2. Symbol descriptions
        3. Symbol reference numbers or codes
        4. Any size or scale information

        Return the information in JSON format with this structure:
        {{
            "symbols": [
                {{"name": "...", "description": "...", "reference": "...", "size_info": "..."}}
            ],
            "page_title": "...",
            "has_legend": true/false
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            legend_data = json.loads(response)
        except json.JSONDecodeError:
            # LLM didn't return valid JSON, parse manually
            legend_data = {
                "symbols": [],
                "page_title": "Unknown",
                "has_legend": False,
                "raw_response": response
            }

        # Add image metadata
        legend_data["images_count"] = len(images)
        legend_data["images"] = images

        return legend_data

    def extract_page_names(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract page names/titles from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries with page information
        """
        doc = fitz.open(pdf_path)
        pages_info = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()

            # Use LLM to extract page title
            prompt = f"""
            Extract the main title or name of this page from the following text.
            Return only the title text, nothing else.

            Text (first 500 characters):
            {text[:500]}
            """

            title = self.llm.invoke(prompt).strip()

            pages_info.append({
                "page_number": page_num + 1,  # 1-based for display
                "title": title,
                "text_length": len(text)
            })

        doc.close()
        return pages_info


class RAGPDFAnalyzer:
    """RAG-based PDF analyzer using LangChain and Ollama."""

    def __init__(
        self,
        pdf_path: str,
        llm_model: str = "llama3.2",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize RAG PDF analyzer.

        Args:
            pdf_path: Path to PDF file
            llm_model: Ollama model name
            embedding_model: HuggingFace embedding model
            persist_directory: Directory for vector store persistence
        """
        self.pdf_path = pdf_path
        self.llm_model = llm_model
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Load PDF
        self.loader = PyPDFLoader(pdf_path)
        self.documents = self.loader.load()

        # Split documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.splits = self.text_splitter.split_documents(self.documents)

        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )

        # Create LLM
        self.llm = OllamaLLM(model=llm_model, temperature=0.2)

        # Create retrieval chain
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> RetrievalQA:
        """Create retrieval QA chain."""
        template = """
        Use the following context from a PDF document to answer the question.
        If you don't know the answer based on the context, say so.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )

    def query(self, question: str) -> str:
        """
        Query the PDF using RAG.

        Args:
            question: Question about the PDF content

        Returns:
            Answer from the LLM
        """
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    def find_similar_symbols(self, symbol_description: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar symbols in the document.

        Args:
            symbol_description: Description of the symbol to find
            k: Number of similar results to return

        Returns:
            List of similar symbol references
        """
        docs = self.vectorstore.similarity_search(symbol_description, k=k)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "source": doc.metadata.get("source", "")
            })

        return results


class SymbolMatcher:
    """Match symbols across different diagrams regardless of size."""

    def __init__(self, llm_model: str = "llama3.2"):
        """
        Initialize symbol matcher.

        Args:
            llm_model: Ollama model to use
        """
        self.llm = OllamaLLM(model=llm_model, temperature=0.1)
        self.known_symbols = {}

    def learn_symbols_from_legend(self, legend_data: Dict[str, Any]) -> None:
        """
        Learn symbols from extracted legend.

        Args:
            legend_data: Legend data from SymbolLegendExtractor
        """
        if "symbols" in legend_data:
            for symbol in legend_data["symbols"]:
                name = symbol.get("name", "")
                if name:
                    self.known_symbols[name] = symbol

    def match_symbol(
        self,
        symbol_bbox: Tuple[float, float, float, float],
        page_context: str,
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Match a symbol to known symbols from the legend.

        Args:
            symbol_bbox: Bounding box of the symbol (x0, y0, x1, y1)
            page_context: Text context around the symbol
            image_data: Optional image data of the symbol

        Returns:
            Match information with confidence score
        """
        # Create prompt for LLM
        symbol_list = "\n".join([f"- {name}: {data.get('description', '')}"
                                 for name, data in self.known_symbols.items()])

        prompt = f"""
        Given this context from a diagram page and a list of known symbols,
        identify which symbol is most likely at this location.

        Known symbols:
        {symbol_list}

        Context around symbol location:
        {page_context}

        Symbol bounding box: {symbol_bbox}

        Which symbol from the list is this most likely to be?
        Provide your answer in JSON format:
        {{
            "matched_symbol": "symbol name",
            "confidence": 0.0-1.0,
            "reasoning": "why you think this is the match"
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            match_data = json.loads(response)
        except json.JSONDecodeError:
            match_data = {
                "matched_symbol": "Unknown",
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM response",
                "raw_response": response
            }

        return match_data

    def compare_symbols(
        self,
        symbol1_data: Dict[str, Any],
        symbol2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two symbols to determine if they are the same.

        Args:
            symbol1_data: Data about first symbol
            symbol2_data: Data about second symbol

        Returns:
            Comparison result with similarity score
        """
        prompt = f"""
        Compare these two symbols and determine if they represent the same thing,
        even if they are different sizes or have minor visual differences.

        Symbol 1:
        {json.dumps(symbol1_data, indent=2)}

        Symbol 2:
        {json.dumps(symbol2_data, indent=2)}

        Are these the same symbol? Consider:
        - Semantic meaning (e.g., both are valves)
        - Position/location on the page
        - Surrounding context

        Provide your answer in JSON format:
        {{
            "are_same": true/false,
            "similarity_score": 0.0-1.0,
            "reasoning": "explanation"
        }}
        """

        response = self.llm.invoke(prompt)

        try:
            comparison = json.loads(response)
        except json.JSONDecodeError:
            comparison = {
                "are_same": False,
                "similarity_score": 0.0,
                "reasoning": "Failed to parse LLM response",
                "raw_response": response
            }

        return comparison


# Helper functions
def setup_rag_for_pdf(pdf_path: str, llm_model: str = "llama3.2") -> RAGPDFAnalyzer:
    """
    Quick setup of RAG analyzer for a PDF.

    Args:
        pdf_path: Path to PDF file
        llm_model: Ollama model to use

    Returns:
        Configured RAGPDFAnalyzer instance
    """
    return RAGPDFAnalyzer(pdf_path, llm_model=llm_model)


def extract_and_learn_symbols(pdf_path: str, legend_page: int = 0, llm_model: str = "llama3.2") -> SymbolMatcher:
    """
    Extract legend and create symbol matcher in one step.

    Args:
        pdf_path: Path to PDF file
        legend_page: Page number with symbol legend (0-based)
        llm_model: Ollama model to use

    Returns:
        SymbolMatcher with learned symbols
    """
    # Extract legend
    extractor = SymbolLegendExtractor(llm_model=llm_model)
    legend_data = extractor.extract_legend_from_page(pdf_path, legend_page)

    # Create matcher and learn symbols
    matcher = SymbolMatcher(llm_model=llm_model)
    matcher.learn_symbols_from_legend(legend_data)

    return matcher
