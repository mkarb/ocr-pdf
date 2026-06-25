#!/usr/bin/env python3
"""
Benchmark script to compare Standard Mode vs Server Mode extraction performance.

Usage:
    python benchmark_modes.py <pdf_path>

Example:
    python benchmark_modes.py ./uploads/drawing.pdf
"""
import sys
import time
import os
from pathlib import Path


def benchmark_standard_mode(pdf_path: str):
    """Benchmark standard mode extraction."""
    from pdf_compare.pdf_extract import pdf_to_vectormap

    start = time.perf_counter()
    vm = pdf_to_vectormap(pdf_path)
    elapsed = time.perf_counter() - start

    return {
        "mode": "Standard",
        "elapsed": elapsed,
        "pages": vm.meta.page_count,
        "geoms": sum(len(p.geoms) for p in vm.pages),
        "texts": sum(len(p.texts) for p in vm.pages),
        "pages_per_sec": vm.meta.page_count / elapsed if elapsed > 0 else 0,
    }


def benchmark_server_mode(pdf_path: str):
    """Benchmark server mode extraction."""
    from pdf_compare.pdf_extract_server import pdf_to_vectormap_server

    start = time.perf_counter()
    vm = pdf_to_vectormap_server(pdf_path)
    elapsed = time.perf_counter() - start

    return {
        "mode": "Server",
        "elapsed": elapsed,
        "pages": vm.meta.page_count,
        "geoms": sum(len(p.geoms) for p in vm.pages),
        "texts": sum(len(p.texts) for p in vm.pages),
        "pages_per_sec": vm.meta.page_count / elapsed if elapsed > 0 else 0,
    }


def print_results(results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'='*60}")
    print(f"Mode: {results['mode']}")
    print(f"{'='*60}")
    print(f"Pages:         {results['pages']}")
    print(f"Geometries:    {results['geoms']:,}")
    print(f"Text Runs:     {results['texts']:,}")
    print(f"Time:          {results['elapsed']:.2f}s")
    print(f"Pages/sec:     {results['pages_per_sec']:.2f}")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"\nBenchmarking PDF: {pdf_path}")
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"CPU_LIMIT env: {os.getenv('CPU_LIMIT', 'not set')}")

    # Benchmark standard mode
    print("\n[1/2] Running Standard Mode...")
    try:
        standard_results = benchmark_standard_mode(pdf_path)
        print_results(standard_results)
    except Exception as e:
        print(f"Standard mode failed: {e}")
        standard_results = None

    # Benchmark server mode
    print("[2/2] Running Server Mode...")
    try:
        server_results = benchmark_server_mode(pdf_path)
        print_results(server_results)
    except Exception as e:
        print(f"Server mode failed: {e}")
        server_results = None

    # Comparison
    if standard_results and server_results:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        speedup = standard_results['elapsed'] / server_results['elapsed']
        time_saved = standard_results['elapsed'] - server_results['elapsed']

        print(f"Standard Mode:  {standard_results['elapsed']:.2f}s ({standard_results['pages_per_sec']:.2f} pages/sec)")
        print(f"Server Mode:    {server_results['elapsed']:.2f}s ({server_results['pages_per_sec']:.2f} pages/sec)")
        print(f"\nSpeedup:        {speedup:.2f}x")
        print(f"Time Saved:     {time_saved:.2f}s")

        if speedup > 1.5:
            print(f"\n✅ Server mode is {speedup:.1f}x faster!")
            print("   Recommended for production deployment.")
        elif speedup > 1.0:
            print(f"\n⚠️  Server mode is {speedup:.1f}x faster (modest improvement)")
            print("   Consider for large batch processing.")
        else:
            print("\n⚠️  No significant speedup detected.")
            print("   PDF may be too small to benefit from parallelization.")

        print(f"{'='*60}\n")

        # Recommendations
        print("RECOMMENDATIONS:")
        if standard_results['pages'] < 10:
            print("- PDF has few pages - speedup limited by overhead")
            print("- Server mode benefits increase with page count")
        else:
            print("- Use Server Mode for Docker deployments")
            print("- Use Standard Mode for Streamlit UI")

        print(f"\nEnvironment tuning:")
        print(f"  export CPU_LIMIT={os.cpu_count()}  # Use all cores")
        print(f"  export PDF_BEZIER_SAMPLES=48       # Higher quality (slower)")
        print(f"  export PDF_SIMPLIFY_TOL=0.1        # Faster (lower quality)")


if __name__ == "__main__":
    main()