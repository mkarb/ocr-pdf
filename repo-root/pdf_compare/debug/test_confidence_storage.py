"""
Test script to verify OCR confidence values are preserved throughout the pipeline.

Usage:
    python test_confidence_storage.py path/to/test.pdf

This script will:
1. Ingest a PDF with OCR enabled
2. Verify confidence values are stored in database
3. Retrieve the document and check confidence values
4. Print statistics about confidence distribution
"""

import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent / "repo-root"
sys.path.insert(0, str(repo_root))

from pdf_compare.pdf_extract import pdf_to_vectormap
from pdf_compare.db_backend import DatabaseBackend
from pdf_compare.store_new import upsert_vectormap


def test_confidence_storage(pdf_path: str, database_url: str):
    """Test confidence value roundtrip through the pipeline."""

    print("=" * 60)
    print("OCR Confidence Storage Test")
    print("=" * 60)
    print(f"PDF: {pdf_path}")
    print(f"Database: {database_url}")
    print()

    # Step 1: Extract PDF with OCR
    print("[1/4] Extracting PDF with OCR enabled...")
    vectormap = pdf_to_vectormap(
        pdf_path,
        enable_ocr=True,
        ocr_dpi=400,
        ocr_engine="tesseract",  # or "easyocr" or "qwen-vl"
        workers=1  # Serial for testing
    )

    print(f"  ✓ Extracted {vectormap.meta.page_count} page(s)")
    print(f"  ✓ Doc ID: {vectormap.meta.doc_id}")

    # Count OCR vs native text
    total_texts = 0
    ocr_texts = 0
    native_texts = 0
    confidences = []

    for page in vectormap.pages:
        for text in page.texts:
            total_texts += 1
            if text.confidence is not None:
                ocr_texts += 1
                confidences.append(text.confidence)
            else:
                native_texts += 1

    print(f"  ✓ Total text spans: {total_texts}")
    print(f"    - Native: {native_texts}")
    print(f"    - OCR: {ocr_texts}")

    if confidences:
        print(f"  ✓ OCR Confidence stats:")
        print(f"    - Min: {min(confidences)}")
        print(f"    - Max: {max(confidences)}")
        print(f"    - Avg: {sum(confidences) / len(confidences):.1f}")
    else:
        print("  ⚠ No OCR confidence values found!")
        print("    (This is normal if the PDF has lots of native text)")

    # Step 2: Store in database
    print()
    print("[2/4] Storing in database...")
    db = DatabaseBackend(database_url)
    db.upsert_vectormap(vectormap)
    print("  ✓ Stored successfully")

    # Step 3: Retrieve from database
    print()
    print("[3/4] Retrieving from database...")
    retrieved_vm = db.get_vectormap(vectormap.meta.doc_id)

    if not retrieved_vm:
        print("  ✗ Failed to retrieve document!")
        return False

    print("  ✓ Retrieved successfully")

    # Step 4: Verify confidence values
    print()
    print("[4/4] Verifying confidence values...")

    retrieved_ocr = 0
    retrieved_confidences = []

    for page in retrieved_vm.pages:
        for text in page.texts:
            if text.confidence is not None:
                retrieved_ocr += 1
                retrieved_confidences.append(text.confidence)

    if retrieved_ocr != ocr_texts:
        print(f"  ✗ OCR text count mismatch!")
        print(f"    - Original: {ocr_texts}")
        print(f"    - Retrieved: {retrieved_ocr}")
        return False

    if len(retrieved_confidences) != len(confidences):
        print(f"  ✗ Confidence count mismatch!")
        print(f"    - Original: {len(confidences)}")
        print(f"    - Retrieved: {len(retrieved_confidences)}")
        return False

    # Check if values match
    original_avg = sum(confidences) / len(confidences) if confidences else 0
    retrieved_avg = sum(retrieved_confidences) / len(retrieved_confidences) if retrieved_confidences else 0

    print(f"  ✓ OCR text count matches: {retrieved_ocr}")
    print(f"  ✓ Confidence count matches: {len(retrieved_confidences)}")
    print(f"  ✓ Original avg confidence: {original_avg:.1f}")
    print(f"  ✓ Retrieved avg confidence: {retrieved_avg:.1f}")

    if abs(original_avg - retrieved_avg) > 0.1:
        print(f"  ✗ Confidence average mismatch!")
        return False

    # Print confidence distribution
    print()
    print("Confidence Distribution:")
    high_conf = sum(1 for c in retrieved_confidences if c >= 90)
    medium_conf = sum(1 for c in retrieved_confidences if 70 <= c < 90)
    low_conf = sum(1 for c in retrieved_confidences if c < 70)

    print(f"  - High (>=90):   {high_conf:4d} ({high_conf/len(retrieved_confidences)*100 if retrieved_confidences else 0:.1f}%)")
    print(f"  - Medium (70-89): {medium_conf:4d} ({medium_conf/len(retrieved_confidences)*100 if retrieved_confidences else 0:.1f}%)")
    print(f"  - Low (<70):      {low_conf:4d} ({low_conf/len(retrieved_confidences)*100 if retrieved_confidences else 0:.1f}%)")

    # Show sample low confidence texts
    if low_conf > 0:
        print()
        print("Sample Low Confidence Texts (<70):")
        low_samples = [(t.text, t.confidence) for page in retrieved_vm.pages
                       for t in page.texts if t.confidence and t.confidence < 70][:5]
        for text, conf in low_samples:
            print(f"  - [{conf}%] \"{text[:50]}\"")

    print()
    print("=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_confidence_storage.py <pdf_path>")
        print()
        print("Example:")
        print("  python test_confidence_storage.py samples/test.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL environment variable not set")
        print()
        print("Set it like this:")
        print("  export DATABASE_URL=postgresql://user:pass@localhost:5432/pdfcompare")
        sys.exit(1)

    try:
        success = test_confidence_storage(pdf_path, database_url)
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
