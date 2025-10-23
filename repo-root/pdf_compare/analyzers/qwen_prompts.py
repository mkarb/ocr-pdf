"""Prompt presets for Qwen2-VL OCR modes."""

from __future__ import annotations

from typing import Dict, Tuple


PROMPT_PRESETS: Dict[str, str] = {
    "layout": """Analyze this document layout and extract text:
1. Identify region types (table/diagram/text)
2. For tables: extract cell-by-cell
3. For diagrams: extract scattered labels
4. For text blocks: maintain reading order

Return JSON: {"texts": [{"text": "...", "bbox": [...], "region_type": "table|diagram|text", "confidence": ...}]}""",
    "sparse": """Extract ALL text from this technical drawing, including:
- Scattered labels and dimensions
- Individual numbers and symbols
- Text at any angle or orientation
- Text in margins and corners

Return JSON: {"texts": [{"text": "...", "bbox": [x0,y0,x1,y1], "confidence": 0.95}]}
Focus on INDIVIDUAL text elements, not paragraphs.""",
    "table": """Extract text from this table/structured document:
- Detect table cells individually
- Preserve row/column structure
- Extract headers separately
- Include all cell content

Return JSON with cell-level granularity.""",
}

DEFAULT_PROMPT_MODE = "sparse"
PROMPT_MODE_CHOICES: Tuple[str, ...] = tuple(sorted(PROMPT_PRESETS.keys()))


def get_prompt(mode: str) -> str:
    """Return the prompt string for the given mode."""
    key = mode.lower()
    if key not in PROMPT_PRESETS:
        raise ValueError(
            f"Unknown Qwen prompt mode '{mode}'. "
            f"Valid options: {', '.join(PROMPT_PRESETS)}"
        )
    return PROMPT_PRESETS[key]

