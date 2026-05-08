import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.table_extractor import TableExtractor


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "document"


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def export_all_tables(pdf_dir: Path, output_dir: Path, clean: bool = False) -> Dict[str, List[Dict]]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = TableExtractor(str(pdf_dir))

    summary: Dict[str, List[Dict]] = {"documents": []}
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    for pdf_file in pdf_files:
        doc_name = pdf_file.name
        doc_folder = output_dir / _safe_name(pdf_file.stem)
        doc_folder.mkdir(parents=True, exist_ok=True)

        doc_entry = {
            "pdf": doc_name,
            "output_folder": str(doc_folder),
            "tables": {},
        }

        for table_type in ("raci", "sipoc"):
            result = extractor.extract_table(
                f"show {table_type} table",
                matched_pdf=doc_name,
                forced_table_type=table_type,
            )

            table_text = result.get("table", "") or ""
            table_error = result.get("error", "") or ""
            pages = result.get("pages", []) or []
            sources = result.get("sources", []) or []

            output_payload = {
                "pdf": doc_name,
                "table_type": table_type,
                "pages": pages,
                "sources": sources,
                "error": table_error,
                "table": table_text,
            }

            _write_json(doc_folder / f"{table_type}.json", output_payload)

            if table_text:
                _write_text(doc_folder / f"{table_type}.md", table_text.rstrip() + "\n")
            else:
                note = table_error or f"No valid {table_type.upper()} table extracted from {doc_name}."
                _write_text(doc_folder / f"{table_type}.md", note + "\n")

            doc_entry["tables"][table_type] = {
                "pages": pages,
                "sources": sources,
                "error": table_error,
                "has_table": bool(table_text),
            }

        _write_json(doc_folder / "manifest.json", doc_entry)
        summary["documents"].append(doc_entry)
        print(f"[OK] {doc_name}")

    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RACI and SIPOC tables from all SOP PDFs.")
    parser.add_argument("--pdf-dir", default="pdfs", help="Folder containing the source PDF files.")
    parser.add_argument("--output-dir", default="extracted_tables", help="Folder where extracted tables are written.")
    parser.add_argument("--clean", action="store_true", help="Delete the output folder before exporting.")
    args = parser.parse_args()

    pdf_dir = _resolve_path(args.pdf_dir)
    output_dir = _resolve_path(args.output_dir)

    summary = export_all_tables(pdf_dir, output_dir, clean=args.clean)
    print(f"Exported {len(summary['documents'])} PDFs to {output_dir}")


if __name__ == "__main__":
    main()