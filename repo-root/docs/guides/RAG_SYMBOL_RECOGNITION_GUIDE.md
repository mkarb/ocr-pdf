# RAG-Based Symbol Recognition Guide

## Overview

This system uses RAG (Retrieval Augmented Generation) with LangChain and Ollama to:
1. Extract symbol legends from PDF first pages
2. Recognize symbols regardless of size
3. Intelligently match symbols across different diagrams
4. Answer questions about PDF content using natural language

## Prerequisites

### 1. Install Ollama

Download and install Ollama from https://ollama.com

```bash
# Windows: Download installer from website

# After installation, pull a model
ollama pull llama3.2
```

### 2. Verify Installation

```bash
ollama list  # Should show llama3.2
```

### 3. Python Dependencies

Already installed:
- langchain
- langchain-community
- langchain-ollama
- chromadb
- sentence-transformers

## Quick Start

### 1. Extract Symbol Legend

```python
from pdf_compare.rag_symbol_recognition import SymbolLegendExtractor

# Extract legend from first page
extractor = SymbolLegendExtractor(llm_model="llama3.2")
legend = extractor.extract_legend_from_page("diagram.pdf", page_number=0)

print(f"Found {len(legend['symbols'])} symbols")
for symbol in legend['symbols']:
    print(f"  - {symbol['name']}: {symbol['description']}")
```

### 2. Ask Questions About Your PDF

```python
from pdf_compare.rag_symbol_recognition import RAGPDFAnalyzer

# Setup RAG for your PDF
analyzer = RAGPDFAnalyzer("engineering_diagram.pdf")

# Ask questions
answer = analyzer.query("What symbols are shown on page 1?")
print(answer)

answer = analyzer.query("What type of valve is used in the main line?")
print(answer)

answer = analyzer.query("List all page names and their contents")
print(answer)
```

### 3. Match Symbols Across Documents

```python
from pdf_compare.rag_symbol_recognition import extract_and_learn_symbols, SymbolMatcher

# Learn symbols from legend
matcher = extract_and_learn_symbols("diagram.pdf", legend_page=0)

# Match a symbol found in another document
match = matcher.match_symbol(
    symbol_bbox=(100, 200, 150, 250),
    page_context="Main water supply line, flow control section"
)

print(f"Matched: {match['matched_symbol']}")
print(f"Confidence: {match['confidence']}")
print(f"Reasoning: {match['reasoning']}")
```

### 4. Compare Symbols Between Documents

```python
# Compare two symbols to see if they're the same type
comparison = matcher.compare_symbols(
    symbol1_data={"name": "Valve", "location": "page 2, top right", "size": "large"},
    symbol2_data={"name": "Valve", "location": "page 5, center", "size": "small"}
)

if comparison['are_same']:
    print(f"These are the same symbol (similarity: {comparison['similarity_score']})")
    print(f"Reasoning: {comparison['reasoning']}")
```

## Use Cases

### Use Case 1: Extract All Page Names

```python
from pdf_compare.rag_symbol_recognition import SymbolLegendExtractor

extractor = SymbolLegendExtractor()
pages = extractor.extract_page_names("multi_page_diagram.pdf")

for page in pages:
    print(f"Page {page['page_number']}: {page['title']}")
```

Output:
```
Page 1: SYMBOL LEGEND - PIPING AND INSTRUMENTATION
Page 2: MAIN WATER SUPPLY SYSTEM
Page 3: ELECTRICAL CONTROL PANEL
Page 4: HVAC DISTRIBUTION LAYOUT
...
```

### Use Case 2: Find Similar Symbols

```python
analyzer = RAGPDFAnalyzer("diagram.pdf")

# Find all mentions of a specific symbol type
similar = analyzer.find_similar_symbols(
    "pressure relief valve with manual override",
    k=5  # Return top 5 matches
)

for result in similar:
    print(f"Page {result['page']}: {result['content']}")
```

### Use Case 3: Intelligent Symbol Comparison

```python
# This is the key feature for recognizing the same symbol at different sizes!

matcher = extract_and_learn_symbols("old_diagram.pdf")

# When comparing two documents, use the matcher
symbol_from_old_doc = {
    "bbox": (100, 200, 120, 220),
    "context": "Main line, near tank T-101"
}

symbol_from_new_doc = {
    "bbox": (150, 300, 200, 350),  # Different size!
    "context": "Main line, near tank T-101"
}

# LLM will understand they're the same symbol based on context
# even though the size is different (20x20 vs 50x50 pixels)
comparison = matcher.compare_symbols(symbol_from_old_doc, symbol_from_new_doc)
print(f"Same symbol: {comparison['are_same']}")  # True
```

## Integration with Existing Comparison

### Enhanced PDF Comparison with Symbol Recognition

```python
from pdf_compare.rag_symbol_recognition import extract_and_learn_symbols
from pdf_compare.compare_new import diff_documents
from pdf_compare.store_new import open_db

# Learn symbols from legend
matcher = extract_and_learn_symbols("diagram.pdf", legend_page=0)

# Traditional vector comparison
backend = open_db("postgresql://...")
diffs = diff_documents(backend, "old_id", "new_id")

# Now enhance with symbol recognition
for page_diff in diffs:
    page_num = page_diff["page"]

    # For each detected change
    for added_geom in page_diff["geometry"]["added"]:
        # Use LLM to identify what symbol this is
        match = matcher.match_symbol(
            symbol_bbox=added_geom,
            page_context=f"Page {page_num} added geometry"
        )

        print(f"Added: {match['matched_symbol']} (confidence: {match['confidence']})")

    for removed_geom in page_diff["geometry"]["removed"]:
        match = matcher.match_symbol(
            symbol_bbox=removed_geom,
            page_context=f"Page {page_num} removed geometry"
        )

        print(f"Removed: {match['matched_symbol']} (confidence: {match['confidence']})")
```

## Advanced Features

### Custom Prompts

```python
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")

# Custom prompt for specific domain
template = """
You are an expert in P&ID (Piping and Instrumentation Diagrams).
Analyze this symbol and identify its type according to ISA standards.

Symbol information:
{symbol_info}

Provide identification in JSON format with ISA symbol code.
"""

prompt = PromptTemplate(template=template, input_variables=["symbol_info"])
result = llm.invoke(prompt.format(symbol_info="..."))
```

### Batch Processing

```python
import glob
from pathlib import Path

# Process all PDFs in a directory
pdf_files = glob.glob("diagrams/*.pdf")

all_legends = {}
for pdf_path in pdf_files:
    extractor = SymbolLegendExtractor()
    legend = extractor.extract_legend_from_page(pdf_path, 0)

    if legend['has_legend']:
        all_legends[Path(pdf_path).name] = legend

# Now you have a database of all symbols across all documents
```

### Performance Optimization

```python
# Use smaller, faster model for simple tasks
extractor_fast = SymbolLegendExtractor(llm_model="llama3.2:1b")  # 1B parameter model

# Use larger model for complex analysis
analyzer_accurate = RAGPDFAnalyzer("complex_diagram.pdf", llm_model="llama3.2:8b")
```

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama service
ollama serve

# In another terminal, verify
ollama list
```

### Model Not Found

```bash
# Download required model
ollama pull llama3.2

# List available models
ollama list
```

### Slow Performance

```python
# Use smaller embedding model
analyzer = RAGPDFAnalyzer(
    "diagram.pdf",
    embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
)

# Reduce chunk size
analyzer.text_splitter.chunk_size = 500  # Default is 1000
```

### Memory Issues

```python
# Process pages individually instead of whole PDF
for page_num in range(page_count):
    legend = extractor.extract_legend_from_page(pdf_path, page_num)
    # Process and discard
```

## Benefits Over Traditional Comparison

### Traditional Vector Comparison
- Detects geometric differences
- Pixel-perfect matching
- Size-dependent
- No semantic understanding

### RAG + LLM Comparison
- Understands symbol meaning
- Size-independent matching
- Context-aware
- Can explain differences
- Handles renamed but equivalent symbols

### Example Scenario

**Situation**: Valve symbol changed from 10mm to 15mm between revisions

**Traditional Result**:
```python
{
    "removed": [{"bbox": [100, 200, 110, 210]}],  # 10mm valve
    "added": [{"bbox": [100, 200, 115, 215]}]     # 15mm valve
}
# Appears as: valve removed and different valve added
```

**RAG Result**:
```python
{
    "interpretation": "Same valve, increased size from 10mm to 15mm",
    "confidence": 0.95,
    "reasoning": "Both identified as 'main shutoff valve' based on context"
}
# Appears as: valve resized, not replaced
```

## Next Steps

1. Integrate with Streamlit UI for interactive symbol recognition
2. Build symbol database across all your diagrams
3. Create automated reports of symbol changes
4. Export symbol inventory for documentation

## Resources

- LangChain Docs: https://python.langchain.com/docs/get_started/introduction
- Ollama Models: https://ollama.com/library
- Chroma Vector DB: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
