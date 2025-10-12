# Install Ollama on Windows - Step by Step

## Method 1: Direct Download (Recommended)

### Step 1: Download Ollama

1. Open your browser
2. Go to: **https://ollama.com/download/windows**
3. Click "Download for Windows"
4. Wait for `OllamaSetup.exe` to download

### Step 2: Install Ollama

1. Double-click `OllamaSetup.exe`
2. Click "Install"
3. Wait for installation to complete
4. Ollama will start automatically (you'll see a llama icon in system tray)

### Step 3: Verify Installation

Open PowerShell and run:
```powershell
ollama --version
```

You should see something like:
```
ollama version is 0.1.x
```

### Step 4: Pull Required Models

In PowerShell:
```powershell
# Pull the LLM (this will take a few minutes, ~2GB download)
ollama pull llama3.2

# Pull the embedding model (smaller, ~274MB)
ollama pull nomic-embed-text
```

### Step 5: Verify Models

```powershell
ollama list
```

Should show:
```
NAME                    ID              SIZE
llama3.2:latest         a80c4f17acd5    2.0 GB
nomic-embed-text:latest 0a109f422b47    274 MB
```

### Step 6: Test It

```powershell
ollama run llama3.2 "Say hello"
```

Should respond with something like: "Hello! How can I help you today?"

## Method 2: Using Winget (If You Have It)

```powershell
# Check if winget is available
winget --version

# If available, install Ollama
winget install Ollama.Ollama

# Then pull models
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Method 3: Manual Installation

If the installer doesn't work:

1. Download from: https://github.com/ollama/ollama/releases
2. Get the latest `ollama-windows-amd64.exe`
3. Rename it to `ollama.exe`
4. Move it to `C:\Program Files\Ollama\`
5. Add `C:\Program Files\Ollama\` to your PATH

## Troubleshooting

### "ollama: command not found"

**Option A: Restart PowerShell**
Close and reopen PowerShell after installation.

**Option B: Check Installation Path**
```powershell
# Check if Ollama is installed
Get-Command ollama

# If not found, check these locations:
Test-Path "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe"
Test-Path "C:\Program Files\Ollama\ollama.exe"
```

**Option C: Add to PATH Manually**
1. Open "Environment Variables"
2. Add Ollama path to System PATH
3. Restart PowerShell

### "Failed to pull model"

**Check Internet Connection:**
```powershell
Test-Connection ollama.com
```

**Check Disk Space:**
Models are large (llama3.2 is ~2GB). Ensure you have enough space:
```powershell
Get-PSDrive C
```

**Try Alternative Model (Smaller):**
```powershell
# Use 1B parameter model (much smaller, ~1GB)
ollama pull llama3.2:1b
```

### Ollama Service Not Running

**Start Ollama Service:**
```powershell
# Check if running
Get-Process ollama

# If not, start it
Start-Process ollama serve
```

Or just click the Ollama icon in your system tray.

### Firewall Issues

If you get connection errors:
1. Open Windows Defender Firewall
2. Click "Allow an app through firewall"
3. Click "Change settings"
4. Find "Ollama" and check both Private and Public
5. Click OK

## After Installation - Test RAG System

Once Ollama is installed and models are pulled:

```powershell
# Navigate to your project
cd h:\repo-root\ocr-pdf\repo-root

# Run the test script
python test_rag.py

# If you have a PDF, test with it
python test_rag.py "path\to\your\diagram.pdf"
```

## Quick Reference

### Common Ollama Commands

```powershell
# Check version
ollama --version

# List installed models
ollama list

# Pull a model
ollama pull <model-name>

# Run a model interactively
ollama run <model-name>

# Remove a model
ollama rm <model-name>

# Show model info
ollama show <model-name>

# Check service status
Get-Process ollama
```

### Recommended Models for This Project

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `llama3.2:1b` | ~1GB | Fast | Good | Quick tests, simple queries |
| `llama3.2` (3B) | ~2GB | Medium | Better | General use (recommended) |
| `llama3.2:8b` | ~4.7GB | Slower | Best | Complex diagrams, high accuracy |

### Model Management

```powershell
# Use different model sizes
ollama pull llama3.2:1b   # Small, fast
ollama pull llama3.2      # Medium (default, recommended)
ollama pull llama3.2:8b   # Large, most accurate

# Free up space by removing unused models
ollama rm llama3.2:8b
```

## Storage Location

Models are stored at:
```
C:\Users\<YourUsername>\.ollama\models
```

Check disk usage:
```powershell
Get-ChildItem "$env:USERPROFILE\.ollama\models" -Recurse |
    Measure-Object -Property Length -Sum |
    Select-Object @{Name="Size(GB)";Expression={[math]::Round($_.Sum/1GB,2)}}
```

## Performance Tips

### Use GPU Acceleration (If Available)

Ollama automatically uses NVIDIA GPU if available. Check:
```powershell
nvidia-smi
```

### Adjust Ollama Settings

Create/edit `C:\Users\<YourUsername>\.ollama\config.json`:
```json
{
  "num_ctx": 2048,
  "num_gpu": 1,
  "num_thread": 8
}
```

## Next Steps

After installation:

1. **Test Ollama:**
   ```powershell
   ollama run llama3.2 "What is 2+2?"
   ```

2. **Test with Python:**
   ```powershell
   python -c "from langchain_community.llms import Ollama; print(Ollama(model='llama3.2').invoke('Hello'))"
   ```

3. **Run RAG Tests:**
   ```powershell
   python test_rag.py your_diagram.pdf
   ```

4. **Start Using RAG:**
   ```powershell
   python -m pdf_compare.rag_simple your_diagram.pdf
   ```

## Support

- Ollama Documentation: https://github.com/ollama/ollama
- Model Library: https://ollama.com/library
- Windows Issues: https://github.com/ollama/ollama/issues

## Alternative: Use Cloud API (No Installation)

If installation fails, you can use cloud-based LLMs instead:

```python
# Use OpenAI API instead
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", api_key="your-key")
```

But Ollama is free and runs locally, so it's worth the installation effort!
