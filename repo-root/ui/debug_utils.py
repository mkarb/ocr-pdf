"""Shared helpers for Streamlit OCR debug configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

DEFAULT_OCR_DEBUG_DIR = "./debug/ocr"
DEFAULT_OCR_DEBUG_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "output_dir": DEFAULT_OCR_DEBUG_DIR,
    "confidence_threshold_low": 70,
    "confidence_threshold_high": 90,
}


def _build_config(
    enabled: bool,
    output_dir: str | None,
    low_threshold: int,
    high_threshold: int,
) -> Dict[str, Any]:
    config = DEFAULT_OCR_DEBUG_CONFIG.copy()
    config.update(
        {
            "enabled": bool(enabled),
            "output_dir": output_dir or DEFAULT_OCR_DEBUG_DIR,
            "confidence_threshold_low": int(low_threshold),
            "confidence_threshold_high": int(high_threshold),
        }
    )
    return config


def update_ocr_debug_session(
    enabled: bool,
    output_dir: str | None,
    low_threshold: int,
    high_threshold: int,
) -> Dict[str, Any]:
    """Persist OCR debug configuration in Streamlit session_state."""
    config = _build_config(enabled, output_dir, low_threshold, high_threshold)
    st.session_state["ocr_debug_config"] = config
    return config


def get_ocr_debug_config() -> Dict[str, Any]:
    """Fetch OCR debug configuration with defaults applied."""
    stored = st.session_state.get("ocr_debug_config")
    if isinstance(stored, dict):
        merged = DEFAULT_OCR_DEBUG_CONFIG.copy()
        merged.update({k: stored.get(k, merged[k]) for k in merged})
        merged["enabled"] = bool(stored.get("enabled", False))
        merged["output_dir"] = stored.get("output_dir", DEFAULT_OCR_DEBUG_DIR) or DEFAULT_OCR_DEBUG_DIR
        return merged
    return DEFAULT_OCR_DEBUG_CONFIG.copy()


def resolve_debug_output_dir(config: Dict[str, Any]) -> Path:
    """Return a Path object for the debug output directory."""
    return Path(config.get("output_dir") or DEFAULT_OCR_DEBUG_DIR)

