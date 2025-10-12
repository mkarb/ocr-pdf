# Quick Start: RAG for PDF Analysis

Simple, few-lines-of-code approach to chat with your PDFs and recognize symbols.

## Installation (Windows)

### 1. Install Ollama

```powershell
# Download and install from https://ollama.com/download/windows
# Or use winget
winget install Ollama.Ollama
```

### 2. Pull Required Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Verify Installation

```bash
ollama list
```

Should show:
```
NAME                    ID              SIZE
llama3.2:latest         a80c4f17acd5    2.0 GB
nomic-embed-text:latest 0a109f422b47    274 MB
```

### 4. Python Dependencies (Already Installed)

```bash
pip install langchain-community pypdf faiss-cpu
```

## Quick Test

### Test 1: Interactive Chat with PDF

```bash
cd h:\repo-root\ocr-pdf\repo-root
python -m pdf_compare.rag_simple your_diagram.pdf
```

Then ask questions:
```
Ask a question: What symbols are in the legend?
Ask a question: What is on page 2?
Ask a question: legend        # Special command - extracts legend
Ask a question: pages         # Special command - lists all pages
Ask a question: quit
```

### Test 2: Compare Two PDFs

```bash
python -m pdf_compare.rag_simple old_diagram.pdf new_diagram.pdf
```

Automatically compares symbols and shows differences.

### Test 3: Python Script (3 Lines!)

```python
from pdf_compare.rag_simple import chat_with_pdf

chat = chat_with_pdf("diagram.pdf")
print(chat.ask("What symbols are shown in the legend?"))
```

### Test 4: Symbol Recognition (Size-Independent)

```python
from pdf_compare.rag_simple import SymbolComparator

# Compare two diagrams
comp = SymbolComparator("old_diagram.pdf", "new_diagram.pdf")

# Check if two symbols are the same despite size difference
result = comp.is_same_symbol(
    bbox1=(100, 200, 110, 210),  # 10x10 symbol in old diagram
    context1="Main water valve at tank T-101",
    bbox2=(150, 300, 175, 325),  # 25x25 symbol in new diagram
    context2="Main water valve at tank T-101"
)

print(f"Same symbol: {result['are_same']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

## Common Use Cases

### Extract Symbol Legend

```python
from pdf_compare.rag_simple import SimplePDFChat

chat = SimplePDFChat("diagram.pdf")
legend = chat.extract_symbol_legend()

for symbol in legend.get('symbols', []):
    print(f"{symbol['name']}: {symbol['description']}")
```

Output:
```
Ball Valve: Used for on/off flow control
Check Valve: Prevents backflow
Pressure Relief: Safety device for overpressure
```

### Get All Page Names

```python
chat = SimplePDFChat("diagram.pdf")
pages = chat.extract_page_names()

for i, page in enumerate(pages, 1):
    print(f"Page {i}: {page}")
```

Output:
```
Page 1: SYMBOL LEGEND
Page 2: MAIN WATER SUPPLY SYSTEM
Page 3: ELECTRICAL CONTROL PANEL
```

### Find All Instances of a Symbol

```python
chat = SimplePDFChat("diagram.pdf")
instances = chat.find_symbol_instances("Ball Valve")
print(instances)
```

Output:
```
Ball Valve appears on:
- Page 2: Main supply line, near tank T-101
- Page 3: Return line, before pump P-203
- Page 5: Bypass circuit, control section
```

### Compare Symbol Changes Between Revisions

```python
from pdf_compare.rag_simple import compare_pdfs

comp = compare_pdfs("rev_A.pdf", "rev_B.pdf")
changes = comp.compare_symbols()
print(changes)
```

Output:
```
Symbol Changes Between Revisions:

NEW SYMBOLS (in Rev B):
- Emergency Shutoff Valve: Added to main line
- Flow Meter: Added at outlet

REMOVED SYMBOLS (from Rev A):
- Manual Override Valve: Replaced with automated version

CHANGED SYMBOLS:
- Main Ball Valve: Size increased from 2" to 3"

UNCHANGED SYMBOLS:
- Check Valve, Pressure Relief, All tank symbols
```

## Integration with Existing Code

### Use with Traditional Vector Comparison

```python
from pdf_compare.compare_new import diff_documents
from pdf_compare.store_new import open_db
from pdf_compare.rag_simple import SymbolComparator

# Get geometric differences (traditional)
backend = open_db("sqlite:///vectormap.sqlite")
diffs = diff_documents(backend, "old_id", "new_id")

# Enhance with LLM understanding
comp = SymbolComparator("old.pdf", "new.pdf")

for page_diff in diffs:
    # For each geometric change, ask LLM what it means
    for added_bbox in page_diff["geometry"]["added"]:
        # Get context from page
        context = f"Page {page_diff['page']}, added geometry"

        # Use LLM to identify the symbol
        # (This is where the magic happens - recognizes symbols despite size)
        result = comp.is_same_symbol(
            bbox1=(0, 0, 0, 0),  # No old symbol
            context1="",
            bbox2=added_bbox,
            context2=context
        )

        print(f"Added: {result['reasoning']}")
```

## Performance Tips

### Fast Mode (Smaller Model)

```python
# Use smaller 1B parameter model for speed
chat = SimplePDFChat("diagram.pdf", llm_model="llama3.2:1b")
```

### Accurate Mode (Larger Model)

```python
# Use larger 8B parameter model for complex diagrams
chat = SimplePDFChat("diagram.pdf", llm_model="llama3.2:8b")
```

### Cache Embeddings

```python
# FAISS vector store is automatically cached in memory
# For persistence, save it:
chat.vector_store.save_local("vector_store_cache")

# Load later:
from langchain_community.vectorstores import FAISS
vector_store = FAISS.load_local("vector_store_cache", embeddings)
```

## Troubleshooting

### Ollama Not Found

```bash
# Verify Ollama is installed
ollama --version

# Start Ollama service (usually auto-starts)
ollama serve
```

### Model Not Found

```bash
# List installed models
ollama list

# Pull missing model
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Slow Performance

```python
# Reduce chunk size for faster processing
chat = SimplePDFChat(
    "diagram.pdf",
    llm_model="llama3.2:1b"  # Use 1B model instead of 3B
)
```

### Memory Issues

```bash
# If running out of memory, use smaller model
ollama pull llama3.2:1b

# Or increase Ollama memory limit
# Edit Ollama config (location varies by OS)
```

## Examples

### Complete Example: Symbol Inventory

```python
from pdf_compare.rag_simple import SimplePDFChat

# Load diagram
chat = SimplePDFChat("P&ID_Drawing.pdf")

# Extract legend
print("=== SYMBOL LEGEND ===")
legend = chat.extract_symbol_legend()
for symbol in legend.get('symbols', []):
    print(f"- {symbol['name']}")

# Get page structure
print("\n=== PAGE STRUCTURE ===")
pages = chat.extract_page_names()
for i, page in enumerate(pages, 1):
    print(f"Page {i}: {page}")

# Find critical components
print("\n=== SAFETY DEVICES ===")
safety = chat.ask("List all safety-related symbols and where they appear")
print(safety)
```

### Complete Example: Revision Comparison

```python
from pdf_compare.rag_simple import SymbolComparator

# Compare revisions
comp = SymbolComparator("RevA.pdf", "RevB.pdf")

# Get summary
print("=== CHANGES SUMMARY ===")
summary = comp.compare_symbols()
print(summary)

# Check specific symbol
result = comp.is_same_symbol(
    bbox1=(100, 200, 120, 220),
    context1="Main line control valve",
    bbox2=(100, 200, 150, 250),  # Different size!
    context2="Main line control valve"
)

if result['are_same']:
    print(f"\n✓ Same symbol (confidence: {result['confidence']})")
    print(f"  {result['reasoning']}")
else:
    print(f"\n✗ Different symbol")
```

## Next Steps

1. Test with your actual diagram PDFs
2. Build a symbol database across all drawings
3. Automate revision comparison reports
4. Integrate with your existing workflow

## Resources

- Ollama: https://ollama.com
- LangChain: https://python.langchain.com
- FAISS: https://github.com/facebookresearch/faiss
