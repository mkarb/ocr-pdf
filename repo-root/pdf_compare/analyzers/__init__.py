from .highres_ocr import HighResOCRConfig, highres_ocr, tiled_ocr
from .enhanced_ocr import EnhancedOCRConfig, SymbolLibrary, enhanced_ocr
from .legend_extractor import LegendEntry, LegendExtractor, validate_ocr_against_legend
from .table_extractor import (
    TableCell,
    TableRow,
    Table,
    TableExtractionConfig,
    TableExtractor,
)

# Optional Qwen2-VL OCR (requires vLLM)
try:
    from .qwen_vl_ocr import QwenVLOCR, VLOCRConfig, get_qwen_vl_ocr
    HAVE_QWEN_VL = True
except ImportError:
    HAVE_QWEN_VL = False
    QwenVLOCR = None
    VLOCRConfig = None
    get_qwen_vl_ocr = None

__all__ = [
    # High-res OCR
    "HighResOCRConfig",
    "highres_ocr",
    "tiled_ocr",
    # Enhanced OCR
    "EnhancedOCRConfig",
    "SymbolLibrary",
    "enhanced_ocr",
    # Legend extraction
    "LegendEntry",
    "LegendExtractor",
    "validate_ocr_against_legend",
    # Table extraction
    "TableCell",
    "TableRow",
    "Table",
    "TableExtractionConfig",
    "TableExtractor",
    # Qwen2-VL OCR (optional)
    "QwenVLOCR",
    "VLOCRConfig",
    "get_qwen_vl_ocr",
    "HAVE_QWEN_VL",
]
