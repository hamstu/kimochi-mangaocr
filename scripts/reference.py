"""Optional development parity reference; never used by the Rust library.
Install torch>=2.6, transformers>=4.45,<5, pillow, numpy, jaconv.
Usage: python scripts/reference.py models image.jpg [beams]
"""
import json
import re
import sys
from pathlib import Path
import jaconv
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

model_dir, image_path = sys.argv[1:3]
beams = int(sys.argv[3]) if len(sys.argv) > 3 else 4
model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True).eval()
image = Image.open(image_path).convert("L").convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
with torch.inference_mode():
    ids = model.generate(pixels, max_length=300, num_beams=beams)[0].tolist()
vocab = Path(model_dir, "vocab.txt").read_text().splitlines()
text = "".join(vocab[i] for i in ids if i > 4)
text = "".join(text.split()).replace("…", "...")
text = re.sub("[・.]{2,}", lambda m: "." * len(m[0]), text)
text = jaconv.h2z(text, ascii=True, digit=True)
print(json.dumps({"text": text, "token_ids": [i for i in ids[1:] if i != 3]}, ensure_ascii=False, indent=2))
