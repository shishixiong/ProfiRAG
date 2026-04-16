"""Convert PDF files to Markdown using DocumentLoader with pymupdf4llm."""

import argparse
import sys
from pathlib import Path

from profirag.ingestion.loaders import DocumentLoader


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown preserving table formats"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="./documents",
        help="Directory containing PDF files (default: ./documents)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./markdown",
        help="Output directory for Markdown files (default: ./markdown)"
    )
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Extract images from PDFs and save them"
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Specific pages to extract (e.g., '1-5,10,15-20')"
    )
    parser.add_argument(
        "--exclude-header-footer",
        action="store_true",
        help="Filter out repeating header/footer content (e.g., copyright notices, page numbers)"
    )
    parser.add_argument(
        "--header-footer-min-occurrences",
        type=int,
        default=3,
        help="Minimum occurrences for auto-detecting header/footer (default: 3)"
    )
    parser.add_argument(
        "--header-footer-pattern",
        type=str,
        action="append",
        default=None,
        help="Custom pattern to filter (can be used multiple times)"
    )
    parser.add_argument(
        "--extract-tables",
        action="store_true",
        help="Extract tables to separate files and use index links in main MD file"
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=None,
        help="Directory to save extracted tables (default: output_dir/tables)"
    )

    args = parser.parse_args()

    # Parse pages argument
    pages = None
    if args.pages:
        pages = parse_pages(args.pages)

    # Parse header footer patterns
    header_footer_patterns = args.header_footer_pattern if args.header_footer_pattern else None

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    loader = DocumentLoader(
        use_pymupdf4llm=True,
        pdf_write_images=args.write_images,
        pdf_image_path=str(output_dir / "images") if args.write_images else None,
        pdf_pages=pages,
        exclude_header_footer=args.exclude_header_footer,
        header_footer_patterns=header_footer_patterns,
        header_footer_auto_detect=True,
        header_footer_min_occurrences=args.header_footer_min_occurrences,
    )

    # Find PDF files
    doc_dir = Path(args.documents)
    pdf_files = list(doc_dir.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {args.documents}")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF files")

    # Convert each PDF
    for pdf_file in pdf_files:
        print(f"\nConverting: {pdf_file.name}")

        # Generate output filename
        output_name = pdf_file.stem + ".md"
        output_path = output_dir / output_name

        try:
            # Convert PDF to Markdown file
            saved_path, table_paths = loader.pdf_to_markdown_file(
                pdf_path=str(pdf_file),
                output_md_path=str(output_path),
                extract_tables=args.extract_tables,
                tables_output_dir=args.tables_dir,
            )
            print(f"  Saved to: {saved_path}")

            if args.extract_tables and table_paths:
                print(f"  Extracted {len(table_paths)} tables to tables/ directory")
                for table_path in table_paths:
                    print(f"    - {Path(table_path).name}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nConversion complete. Output directory: {output_dir}")


def parse_pages(page_spec: str) -> list[int]:
    """Parse page specification like '1-5,10,15-20' to list of page numbers.

    Args:
        page_spec: Page specification string

    Returns:
        List of page numbers (0-indexed for pymupdf4llm)
    """
    pages = []
    for part in page_spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))

    # Convert to 0-indexed (pymupdf4llm uses 0-indexed pages)
    return [p - 1 for p in pages]


if __name__ == "__main__":
    main()