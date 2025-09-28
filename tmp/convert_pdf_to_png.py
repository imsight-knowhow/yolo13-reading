import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("ERROR: PyMuPDF (pymupdf) is required. Install with: pip install pymupdf", file=sys.stderr)
    raise


def convert_pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> None:
    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError(f"No pages in PDF: {pdf_path}")
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(png_path))


def main(argv):
    if len(argv) < 3:
        print("Usage: python tmp/convert_pdf_to_png.py <input.pdf> <output.png> [dpi]", file=sys.stderr)
        return 2
    pdf = Path(argv[1])
    png = Path(argv[2])
    dpi = int(argv[3]) if len(argv) > 3 else 200
    convert_pdf_to_png(pdf, png, dpi)
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

