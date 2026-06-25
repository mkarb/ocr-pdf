#!/usr/bin/env python3
"""
Demo script for table extraction from engineering drawings.

Tests the TableExtractor on PDF files to extract:
- Bills of Materials (BOM) / Parts Lists
- Symbol/Legend tables
- Line type tables
"""

from pathlib import Path
from pdf_compare.analyzers import TableExtractor, TableExtractionConfig


def demo_table_extraction(pdf_path: str, page_num: int = 0):
    """
    Demo: Extract tables from a PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-based) to extract from
    """
    print(f"\n{'='*60}")
    print(f"Extracting tables from: {Path(pdf_path).name}")
    print(f"Page: {page_num + 1}")
    print(f"{'='*60}\n")

    # Create configuration
    config = TableExtractionConfig(
        dpi=400,
        enable_line_detection=True,
        enable_whitespace_detection=True,
        ocr_min_conf=50,
        min_rows=2,
        min_cols=2
    )

    # Create extractor
    extractor = TableExtractor(config)

    # Extract table from specific page
    print("🔍 Detecting tables...")
    table = extractor.extract_table(pdf_path, page_num)

    if not table:
        print("❌ No table found on this page.")
        return

    print(f"✅ Table found!")
    print(f"   Type: {table.table_type}")
    print(f"   Headers: {table.headers}")
    print(f"   Rows: {len(table.rows)}")
    print(f"   Columns: {len(table.headers)}")
    print(f"   Detection: {table.metadata.get('detection_method', 'unknown')}")
    print()

    # Display table contents
    print("📊 Table Contents:")
    print("-" * 80)

    # Print headers
    print(" | ".join(f"{h:20s}" for h in table.headers))
    print("-" * 80)

    # Print rows (limit to first 10)
    for row in table.rows[:10]:
        row_text = []
        for cell in row.cells:
            text = cell.text[:20] if cell.text else ""
            row_text.append(f"{text:20s}")
        print(" | ".join(row_text))

    if len(table.rows) > 10:
        print(f"... ({len(table.rows) - 10} more rows)")

    print("-" * 80)
    print()

    # Save results
    output_dir = Path("./table_extraction_results")
    output_dir.mkdir(exist_ok=True)

    # Save as JSON
    json_path = output_dir / f"table_page_{page_num + 1}.json"
    extractor.save_tables_json(str(json_path))
    print(f"💾 Saved JSON: {json_path}")

    # Try to save as CSV (requires pandas)
    try:
        df = table.to_dataframe()
        csv_path = output_dir / f"table_page_{page_num + 1}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV: {csv_path}")
    except ImportError:
        print("⚠️  Pandas not installed - skipping CSV export")
        print("   Install with: pip install pandas")

    print()


def demo_extract_all_tables(pdf_path: str):
    """
    Demo: Extract all tables from a PDF.

    Args:
        pdf_path: Path to PDF file
    """
    print(f"\n{'='*60}")
    print(f"Extracting ALL tables from: {Path(pdf_path).name}")
    print(f"{'='*60}\n")

    # Create extractor
    config = TableExtractionConfig(
        dpi=400,
        enable_line_detection=True,
        enable_whitespace_detection=True
    )
    extractor = TableExtractor(config)

    # Extract all tables
    print("🔍 Scanning all pages...")
    tables = extractor.extract_all_tables(pdf_path)

    print(f"✅ Found {len(tables)} tables\n")

    # Summarize findings
    bom_tables = extractor.get_bom_tables()
    symbol_tables = extractor.get_symbol_tables()

    print(f"📋 BOMs/Parts Lists: {len(bom_tables)}")
    print(f"🔣 Symbol/Legend Tables: {len(symbol_tables)}")
    print(f"📊 Other Tables: {len(tables) - len(bom_tables) - len(symbol_tables)}")
    print()

    # Display details
    for i, table in enumerate(tables, 1):
        print(f"Table {i}: {table.table_type} (Page {table.page})")
        print(f"  Headers: {', '.join(table.headers[:5])}")
        if len(table.headers) > 5:
            print(f"           ... and {len(table.headers) - 5} more")
        print(f"  Rows: {len(table.rows)}")
        print()

    # Save all tables
    output_dir = Path("./table_extraction_results")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "all_tables.json"
    extractor.save_tables_json(str(json_path))
    print(f"💾 Saved all tables to: {json_path}")

    try:
        extractor.save_tables_csv(str(output_dir))
        print(f"💾 Saved CSV files to: {output_dir}/")
    except ImportError:
        print("⚠️  Pandas not installed - skipping CSV export")

    print()


def demo_bom_extraction(pdf_path: str):
    """
    Demo: Extract and display BOM specifically.

    Args:
        pdf_path: Path to PDF file
    """
    print(f"\n{'='*60}")
    print(f"Extracting BOM from: {Path(pdf_path).name}")
    print(f"{'='*60}\n")

    # Configure for BOM extraction
    config = TableExtractionConfig(
        dpi=400,
        enable_line_detection=True,
        bom_keywords=[
            "PARTS LIST", "BILL OF MATERIALS", "BOM", "MATERIAL LIST",
            "ITEM", "QTY", "QUANTITY", "PART NUMBER", "DESCRIPTION", "PART NO"
        ]
    )

    extractor = TableExtractor(config)
    tables = extractor.extract_all_tables(pdf_path)

    bom_tables = extractor.get_bom_tables()

    if not bom_tables:
        print("❌ No BOM found in this drawing")
        return

    print(f"✅ Found {len(bom_tables)} BOM table(s)\n")

    for bom in bom_tables:
        print(f"📋 BOM on Page {bom.page}")
        print(f"   Columns: {', '.join(bom.headers)}")
        print(f"   Parts: {len(bom.rows)}")
        print()

        # Display parts list
        print("Parts List:")
        print("-" * 80)

        # Find quantity and description columns
        qty_col = None
        desc_col = None
        part_col = None

        for i, header in enumerate(bom.headers):
            h_upper = header.upper()
            if "QTY" in h_upper or "QUANTITY" in h_upper:
                qty_col = i
            elif "DESC" in h_upper:
                desc_col = i
            elif "PART" in h_upper and "NO" in h_upper:
                part_col = i

        # Display rows
        for row in bom.rows[:20]:  # Show first 20 parts
            parts = []

            if part_col is not None and part_col < len(row.cells):
                parts.append(f"Part: {row.cells[part_col].text}")

            if qty_col is not None and qty_col < len(row.cells):
                parts.append(f"Qty: {row.cells[qty_col].text}")

            if desc_col is not None and desc_col < len(row.cells):
                parts.append(f"Desc: {row.cells[desc_col].text}")

            if parts:
                print(" | ".join(parts))

        if len(bom.rows) > 20:
            print(f"... and {len(bom.rows) - 20} more parts")

        print("-" * 80)
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_table_extractor.py <pdf_path> [page_num]")
        print("  python test_table_extractor.py <pdf_path> --all")
        print("  python test_table_extractor.py <pdf_path> --bom")
        print()
        print("Examples:")
        print("  python test_table_extractor.py drawing.pdf")
        print("  python test_table_extractor.py drawing.pdf 0")
        print("  python test_table_extractor.py drawing.pdf --all")
        print("  python test_table_extractor.py drawing.pdf --bom")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Determine mode
    if len(sys.argv) >= 3:
        if sys.argv[2] == "--all":
            demo_extract_all_tables(pdf_path)
        elif sys.argv[2] == "--bom":
            demo_bom_extraction(pdf_path)
        else:
            try:
                page_num = int(sys.argv[2])
                demo_table_extraction(pdf_path, page_num)
            except ValueError:
                print(f"Error: Invalid page number: {sys.argv[2]}")
                sys.exit(1)
    else:
        # Default: extract from first page
        demo_table_extraction(pdf_path, 0)
