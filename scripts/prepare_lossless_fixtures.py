"""Optional parity validation: preserve Pillow-decoded pixels as PNG.
Only used during development, never by the library or distributed application.
Run download_fixtures.sh first. Requires Pillow.
"""
from pathlib import Path
from PIL import Image
root = Path(__file__).resolve().parent.parent / "tests/fixtures"
(root / "lossless").mkdir(exist_ok=True)
for path in sorted((root / "images").glob("*.jpg")):
    Image.open(path).convert("RGB").save(root / "lossless" / f"{path.stem}.png")
