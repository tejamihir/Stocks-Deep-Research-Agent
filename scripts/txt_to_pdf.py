import argparse
import os
from pathlib import Path
from typing import Iterable

from fpdf import FPDF


def iter_txt_files(root: Path) -> Iterable[Path]:
    # Typical pattern: <root>/<TICKER>/10-K/<accession>/full-submission.txt
    for path in root.rglob("*.txt"):
        # only convert 10-K related txt files (optional heuristic)
        parts_lower = {part.lower() for part in path.parts}
        if "10-k" in parts_lower or "10-k" in path.as_posix().lower():
            yield path


def txt_to_pdf(txt_path: Path, pdf_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            # Use multi_cell to wrap long lines
            pdf.multi_cell(0, 5, line)

    pdf.output(str(pdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SEC 10-K .txt files to PDF")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Data" / "Annual Reports" / "sec-edgar-filings"),
        help="Root directory containing downloaded 10-K folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Data" / "Annual Reports" / "pdfs"),
        help="Directory to write generated PDFs",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for txt_path in iter_txt_files(input_root):
        # Derive ticker from relative path if possible (folder layout)
        try:
            rel = txt_path.relative_to(input_root)
            # Expect: <TICKER>/10-K/<accession>/file.txt
            ticker = rel.parts[0]
        except Exception:
            ticker = txt_path.stem

        pdf_name = f"{ticker}_latest_10k.pdf"
        pdf_path = output_root / pdf_name

        try:
            txt_to_pdf(txt_path, pdf_path)
            print(f"Converted: {txt_path} -> {pdf_path}")
            count += 1
        except Exception as e:
            print(f"Failed: {txt_path} ({e})")

    print(f"Done. Converted {count} file(s).")


if __name__ == "__main__":
    main()
