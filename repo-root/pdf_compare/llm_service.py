"""
LLM Service Manager for PDF Analysis.

Manages multiple LLM backends:
- Ollama: Fast, lightweight, good for development and quick queries
- vLLM: High-performance inference with AMD ROCm support, ideal for batch processing

Usage:
    # Initialize service
    service = LLMServiceManager()

    # Quick query via Ollama
    response = service.quick_query("What symbols are in this diagram?")

    # Batch processing via vLLM (when available)
    responses = service.batch_analyze([prompt1, prompt2, prompt3])

    # Check vLLM availability
    if service.vllm_available:
        layout = service.analyze_layout(page_vectors)
"""

from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import vLLM (optional dependency)
try:
    from vllm import LLM, SamplingParams
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False
    logger.info("vLLM not available. Install with: pip install vllm")

# Try to import Ollama (existing dependency)
try:
    from langchain_community.llms import Ollama
    HAVE_OLLAMA = True
except ImportError:
    HAVE_OLLAMA = False
    logger.warning("Ollama not available. Install langchain-community.")


class LLMServiceConfig:
    """Configuration for LLM services."""

    def __init__(
        self,
        # Ollama settings
        ollama_model: str = "qwen2.5:7b",
        ollama_base_url: Optional[str] = None,

        # vLLM settings
        vllm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        vllm_tensor_parallel: int = 2,  # Use both AMD GPUs
        vllm_gpu_memory_util: float = 0.85,
        vllm_max_model_len: int = 4096,
        vllm_dtype: str = "float16",

        # Feature flags
        enable_vllm: bool = True,
        vllm_lazy_init: bool = True,  # Only load when first needed
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        self.vllm_model = vllm_model
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.vllm_gpu_memory_util = vllm_gpu_memory_util
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_dtype = vllm_dtype

        self.enable_vllm = enable_vllm and HAVE_VLLM and os.getenv("VLLM_ENABLED", "true").lower() == "true"
        self.vllm_lazy_init = vllm_lazy_init


class LLMServiceManager:
    """
    Unified LLM service manager supporting Ollama and vLLM backends.

    Automatically routes queries to the appropriate backend based on:
    - Query type (single vs batch)
    - Complexity requirements
    - Backend availability
    """

    def __init__(self, config: Optional[LLMServiceConfig] = None):
        """
        Initialize LLM service manager.

        Args:
            config: Service configuration. If None, uses defaults from environment.
        """
        self.config = config or LLMServiceConfig()

        # Initialize Ollama (lightweight, always available)
        self._ollama = None
        if HAVE_OLLAMA:
            try:
                self._ollama = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url
                )
                logger.info(f"Ollama initialized: {self.config.ollama_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")

        # Initialize vLLM (high-performance, lazy-loaded)
        self._vllm = None
        self._vllm_initialized = False

        if not self.config.vllm_lazy_init and self.config.enable_vllm:
            self._init_vllm()

    def _init_vllm(self) -> bool:
        """
        Initialize vLLM backend.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._vllm_initialized:
            return True

        if not HAVE_VLLM:
            logger.warning("vLLM not available. Install with: pip install vllm")
            return False

        if not self.config.enable_vllm:
            logger.info("vLLM disabled via configuration")
            return False

        try:
            logger.info(f"Initializing vLLM: {self.config.vllm_model}")
            logger.info(f"  Tensor parallel size: {self.config.vllm_tensor_parallel}")
            logger.info(f"  GPU memory utilization: {self.config.vllm_gpu_memory_util}")

            self._vllm = LLM(
                model=self.config.vllm_model,
                tensor_parallel_size=self.config.vllm_tensor_parallel,
                gpu_memory_utilization=self.config.vllm_gpu_memory_util,
                max_model_len=self.config.vllm_max_model_len,
                dtype=self.config.vllm_dtype,
                trust_remote_code=True,
            )

            self._vllm_initialized = True
            logger.info(f"vLLM initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.info("Falling back to Ollama for all queries")
            self._vllm = None
            self._vllm_initialized = False
            return False

    @property
    def vllm(self) -> Optional[LLM]:
        """Get vLLM instance, initializing if needed (lazy loading)."""
        if self._vllm is None and not self._vllm_initialized:
            self._init_vllm()
        return self._vllm

    @property
    def ollama(self) -> Optional[Ollama]:
        """Get Ollama instance."""
        return self._ollama

    @property
    def vllm_available(self) -> bool:
        """Check if vLLM backend is available."""
        if self._vllm is not None:
            return True
        if not self._vllm_initialized and self.config.enable_vllm:
            return self._init_vllm()
        return False

    @property
    def ollama_available(self) -> bool:
        """Check if Ollama backend is available."""
        return self._ollama is not None

    def quick_query(self, prompt: str, **kwargs) -> str:
        """
        Fast single query via Ollama.

        Best for:
        - Interactive user queries
        - Simple questions
        - Development/debugging

        Args:
            prompt: Query prompt
            **kwargs: Additional arguments for Ollama

        Returns:
            Response text

        Raises:
            RuntimeError: If no backend is available
        """
        if not self.ollama_available:
            raise RuntimeError("Ollama backend not available. Check installation and configuration.")

        logger.debug(f"Ollama query: {prompt[:100]}...")
        return self._ollama.invoke(prompt, **kwargs)

    def batch_analyze(
        self,
        prompts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 512,
        **kwargs
    ) -> List[str]:
        """
        Batch processing via vLLM (faster for multiple prompts).

        Falls back to Ollama if vLLM unavailable.

        Best for:
        - Processing multiple pages
        - Layout analysis at scale
        - Batch table extraction

        Args:
            prompts: List of prompts to process
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            **kwargs: Additional vLLM sampling parameters

        Returns:
            List of response texts
        """
        # Try vLLM first (much faster for batches)
        if self.vllm_available:
            logger.info(f"vLLM batch processing: {len(prompts)} prompts")

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            outputs = self.vllm.generate(prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]

        # Fallback to Ollama (slower but works)
        elif self.ollama_available:
            logger.warning(f"vLLM not available, using Ollama for batch (slower): {len(prompts)} prompts")
            return [self.quick_query(prompt) for prompt in prompts]

        else:
            raise RuntimeError("No LLM backend available")

    def structured_query(
        self,
        prompt: str,
        response_format: str = "json",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Query expecting structured output (JSON, etc.).

        Uses vLLM if available for better structured output support.

        Args:
            prompt: Query prompt (should specify desired format)
            response_format: Expected format ("json", "yaml", etc.)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response length

        Returns:
            Response text (caller should parse based on format)
        """
        # vLLM preferred for structured output (Qwen2.5 is excellent at JSON)
        if self.vllm_available:
            logger.debug(f"vLLM structured query ({response_format})")

            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )

            outputs = self.vllm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text

        # Fallback to Ollama
        elif self.ollama_available:
            logger.debug(f"Ollama structured query ({response_format})")
            return self.quick_query(prompt)

        else:
            raise RuntimeError("No LLM backend available")

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about available backends.

        Returns:
            Dictionary with backend status and configuration
        """
        return {
            "ollama": {
                "available": self.ollama_available,
                "model": self.config.ollama_model if self.ollama_available else None,
                "base_url": self.config.ollama_base_url if self.ollama_available else None,
            },
            "vllm": {
                "available": self.vllm_available,
                "model": self.config.vllm_model if self.config.enable_vllm else None,
                "tensor_parallel_size": self.config.vllm_tensor_parallel,
                "initialized": self._vllm_initialized,
            },
            "default_backend": "vllm" if self.vllm_available else "ollama",
        }

    def __repr__(self) -> str:
        backends = []
        if self.ollama_available:
            backends.append(f"Ollama({self.config.ollama_model})")
        if self.vllm_available:
            backends.append(f"vLLM({self.config.vllm_model})")
        elif self.config.enable_vllm and not self._vllm_initialized:
            backends.append("vLLM(lazy)")

        return f"LLMServiceManager({', '.join(backends)})"


# Global service instance (lazy-initialized)
_global_service: Optional[LLMServiceManager] = None


def get_llm_service(config: Optional[LLMServiceConfig] = None) -> LLMServiceManager:
    """
    Get global LLM service instance (singleton pattern).

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        Global LLM service manager
    """
    global _global_service

    if _global_service is None:
        _global_service = LLMServiceManager(config)
        logger.info(f"Initialized global LLM service: {_global_service}")

    return _global_service


def reset_llm_service():
    """Reset global LLM service (useful for testing)."""
    global _global_service
    _global_service = None


__all__ = [
    "LLMServiceManager",
    "LLMServiceConfig",
    "get_llm_service",
    "reset_llm_service",
    "HAVE_VLLM",
    "HAVE_OLLAMA",
]
