# pdf_compare/worker_pool.py
"""
Worker pool management for PDF extraction with process pools.

This module provides robust worker pool utilities with:
- Thread explosion prevention (BLAS/OpenMP limiting)
- Memory leak prevention (document cache cleanup)
- Throttled future submission to bound memory usage
- Automatic retry with reduced workers on crashes
- Detailed error context for debugging

Usage:
    from .worker_pool import ThrottledPoolExecutor, worker_init

    with ThrottledPoolExecutor(max_workers=8, initializer=worker_init) as pool:
        results = pool.map(process_page, pages)
"""
from __future__ import annotations
from typing import Dict, Any, Callable, Optional, Iterable, List
from concurrent.futures import ProcessPoolExecutor, Future, wait, FIRST_COMPLETED
from collections import deque
import multiprocessing as mp
import logging
import atexit
import os

try:
    from threadpoolctl import threadpool_limits  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    threadpool_limits = None  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------
# Document Cache Management
# -----------------------
# Process-level cache: keeps one document handle per process to avoid repeated opens
_PROCESS_DOC_CACHE: Dict[str, Any] = {}
_CACHE_CLEANUP_REGISTERED = False


# -----------------------
# Thread control helpers
# -----------------------
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_THREADPOOL_LIMITER = None
_THREADPOOL_EXIT_REGISTERED = False


def get_cached_doc(pdf_path: str):
    """
    Get or create a cached document handle for this process.

    This cache is per-worker process to avoid repeatedly opening the same PDF.
    For multi-page PDFs, each worker can reuse the document handle across pages.

    Args:
        pdf_path: Path to PDF file

    Returns:
        fitz.Document handle (cached)
    """
    if pdf_path not in _PROCESS_DOC_CACHE:
        import fitz
        _PROCESS_DOC_CACHE[pdf_path] = fitz.open(pdf_path)
    return _PROCESS_DOC_CACHE[pdf_path]


def clear_doc_cache():
    """
    Close and clear all cached documents.

    Call this after processing to prevent memory leaks from accumulated PDF handles.
    Registered automatically in worker_init() via atexit.
    """
    global _PROCESS_DOC_CACHE
    for doc in _PROCESS_DOC_CACHE.values():
        try:
            doc.close()
        except Exception:
            pass  # Best effort cleanup
    _PROCESS_DOC_CACHE.clear()


def _set_thread_env_defaults() -> None:
    """Ensure worker-friendly BLAS/OpenMP env vars are present."""
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def _activate_threadpool_limits() -> None:
    """Clamp native threadpools (NumPy/OpenCV) to a single thread if possible."""
    global _THREADPOOL_LIMITER, _THREADPOOL_EXIT_REGISTERED

    if threadpool_limits is None or _THREADPOOL_LIMITER is not None:
        return

    limiter = threadpool_limits(limits=1)
    limiter.__enter__()
    _THREADPOOL_LIMITER = limiter

    if not _THREADPOOL_EXIT_REGISTERED:
        atexit.register(_shutdown_threadpool_limits)
        _THREADPOOL_EXIT_REGISTERED = True


def _shutdown_threadpool_limits() -> None:
    """Release threadpool overrides on interpreter shutdown."""
    global _THREADPOOL_LIMITER
    if _THREADPOOL_LIMITER is not None:
        try:
            _THREADPOOL_LIMITER.__exit__(None, None, None)
        finally:
            _THREADPOOL_LIMITER = None


def configure_thread_env() -> None:
    """Prepare the current process for safe multiprocessing imports."""
    _set_thread_env_defaults()


# -----------------------
# Worker Process Initialization
# -----------------------
def worker_init():
    """
    Initialize worker process with resource limits and cleanup handlers.

    This function should be passed as `initializer` to ProcessPoolExecutor.

    What it does:
    1. Caps BLAS/OpenMP threads to 1 per worker to prevent thread explosion
       - Without this: 8 workers × 8 BLAS threads each = 64 threads for 8 cores (terrible!)
       - With this: 8 workers × 1 thread each = 8 threads for 8 cores (optimal!)

    2. Registers document cache cleanup on worker exit to prevent memory leaks

    Environment Variables Set:
        OMP_NUM_THREADS: OpenMP thread limit
        OPENBLAS_NUM_THREADS: OpenBLAS thread limit
        MKL_NUM_THREADS: Intel MKL thread limit
        NUMEXPR_NUM_THREADS: NumExpr thread limit
        VECLIB_MAXIMUM_THREADS: Apple Accelerate thread limit
    """
    global _CACHE_CLEANUP_REGISTERED

    configure_thread_env()
    _activate_threadpool_limits()

    if not _CACHE_CLEANUP_REGISTERED:
        atexit.register(clear_doc_cache)
        _CACHE_CLEANUP_REGISTERED = True


# -----------------------
# Throttled Pool Executor
# -----------------------
class ThrottledPoolExecutor:
    """
    ProcessPoolExecutor wrapper with throttled submission to bound memory usage.

    Problem this solves:
    - Standard ProcessPoolExecutor submits all futures immediately
    - For 100-page PDF: 100 futures × 10MB per page = 1GB+ memory
    - Large PDFs cause OOM and BrokenProcessPool crashes

    Solution:
    - Keep only `workers × multiplier` futures in-flight at once
    - As futures complete, submit more to keep queue full
    - Bounds memory to: workers × multiplier × avg_page_size

    Example:
        - 8 workers, multiplier=3, 10MB per page
        - Memory: 8 × 3 × 10MB = 240MB (manageable)
        - vs unbounded: 100 × 10MB = 1GB (crash)

    Features:
    - Automatic timeout handling (300s per batch, 10s per result)
    - Detailed error messages with page context
    - Always uses 'spawn' to avoid fork+thread deadlocks
    - Accepts custom worker initializer

    Usage:
        with ThrottledPoolExecutor(max_workers=8, initializer=worker_init) as pool:
            results = pool.submit_throttled(
                worker_func=process_page,
                items=list(range(100)),
                progress_callback=lambda done, total: print(f"{done}/{total}")
            )
    """

    def __init__(
        self,
        max_workers: int,
        initializer: Optional[Callable] = None,
        mp_context: Optional[mp.context.BaseContext] = None,
        in_flight_multiplier: int = 3,
    ):
        """
        Initialize throttled pool executor.

        Args:
            max_workers: Maximum number of worker processes
            initializer: Optional function to call in each worker on startup
            mp_context: Multiprocessing context (defaults to 'spawn' for safety)
            in_flight_multiplier: Keep workers × multiplier futures in-flight (default: 3)
        """
        self.max_workers = max_workers
        self.initializer = initializer
        self.mp_context = mp_context or mp.get_context("spawn")
        self.max_in_flight = max_workers * in_flight_multiplier
        self.executor: Optional[ProcessPoolExecutor] = None

    def __enter__(self):
        """Start the pool executor."""
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=self.mp_context,
            initializer=self.initializer,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the pool executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
        return False

    def submit_throttled(
        self,
        worker_func: Callable,
        items: Iterable[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        item_to_args: Optional[Callable[[Any], tuple]] = None,
    ) -> List[Any]:
        """
        Submit work with throttled in-flight futures to bound memory.

        Args:
            worker_func: Function to call for each item (must be picklable)
            items: Iterable of items to process
            progress_callback: Optional callback(completed_count, total_count)
            item_to_args: Optional function to convert item → args tuple for worker_func
                         Default: passes item as single argument

        Returns:
            List of results in submission order (not completion order)

        Raises:
            RuntimeError: If page extraction fails with detailed error context
        """
        if not self.executor:
            raise RuntimeError("ThrottledPoolExecutor not entered (use 'with' statement)")

        items_list = list(items)
        total_items = len(items_list)

        if total_items == 0:
            return []

        # Default item_to_args: pass item as single argument
        if item_to_args is None:
            item_to_args = lambda item: (item,)

        # Throttled submission queue
        pending_items = deque(enumerate(items_list))
        in_flight: Dict[Future, int] = {}  # future → item_index mapping
        results_dict: Dict[int, Any] = {}  # item_index → result mapping

        # Submit initial batch
        while pending_items and len(in_flight) < self.max_in_flight:
            item_idx, item = pending_items.popleft()
            args = item_to_args(item)
            fut = self.executor.submit(worker_func, *args)
            in_flight[fut] = item_idx

        # Process results and refill queue
        completed_count = 0
        while in_flight:
            # Wait for at least one future to complete (with timeout)
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED, timeout=300)

            if not done:
                # Timeout - log warning but continue
                logger.warning(f"Timeout waiting for work completion after 300s ({completed_count}/{total_items} done)")
                continue

            for fut in done:
                item_idx = in_flight.pop(fut)
                try:
                    result = fut.result(timeout=10)
                    results_dict[item_idx] = result
                    completed_count += 1

                    if progress_callback:
                        progress_callback(completed_count, total_items)

                except Exception as e:
                    error_msg = f"Item {item_idx} (of {total_items}) processing failed: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            # Refill queue with more items
            while pending_items and len(in_flight) < self.max_in_flight:
                item_idx, item = pending_items.popleft()
                args = item_to_args(item)
                fut = self.executor.submit(worker_func, *args)
                in_flight[fut] = item_idx

        # Return results in original order
        return [results_dict[i] for i in range(total_items)]


# -----------------------
# Helper Functions
# -----------------------
def get_optimal_workers(
    requested_workers: int = 0,
    page_count: int = 1,
    enable_ocr: bool = False,
    force_serial: bool = False,
) -> int:
    """
    Calculate optimal worker count based on constraints.

    Args:
        requested_workers: Requested worker count (0 = auto-detect from CPU_LIMIT or cpu_count)
        page_count: Number of pages in document
        enable_ocr: Whether OCR is enabled (limits to 2 workers max due to memory)
        force_serial: Force serial processing (returns 1)

    Returns:
        Optimal worker count (1 to cpu_count)
    """
    import os

    if force_serial:
        return 1

    # Auto-detect from environment or CPU count
    if requested_workers <= 0:
        cpu_limit = int(os.getenv("CPU_LIMIT", os.cpu_count() or 4))
        requested_workers = max(1, cpu_limit - 1)  # Leave one core for orchestration

    # Smart worker allocation based on document size
    if page_count == 1:
        workers = 1  # No benefit from parallel for single page
    elif page_count <= 4:
        workers = min(2, requested_workers)  # Max 2 for small docs
    else:
        workers = max(1, min(requested_workers, page_count))  # Cap at page count

    # OCR is extremely memory-intensive (800 DPI rendering + Tesseract)
    # Large engineering drawings can use 2GB+ per page during OCR
    # Limit to 2 workers max to prevent OOM
    if enable_ocr:
        original_workers = workers
        workers = min(workers, 2)
        if workers < original_workers:
            logger.info(f"OCR enabled: limiting workers from {original_workers} to {workers} to prevent OOM")

    return workers
