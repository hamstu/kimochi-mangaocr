# mangaocr (Rust)

A Rust port of [Manga OCR](https://github.com/kha-white/manga-ocr) (Apache-2.0) with a native port of [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (GPL-3.0): Japanese manga OCR on Candle (CPU, Metal or CUDA), with no Python, PyTorch, ONNX Runtime or OpenCV.

Used by [Kimochi](https://github.com/hamstu/kimochi-tauri).

```toml
[dependencies]
mangaocr = { git = "https://github.com/hamstu/kimochi-mangaocr", features = ["metal"] }
```

```rust,no_run
use mangaocr::{MangaOcr, ModelPaths, OcrOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = MangaOcr::new(ModelPaths::from_dir("models"))?;
    let result = engine.ocr_file("selection.png", &OcrOptions::default())?;
    println!("{}", result.text);
    Ok(())
}
```

For whole pages, `ComicTextDetector` finds text blocks and `MangaOcr::recognize_page` reads each one.

Models are not included. Download them (pinned and hash-checked, about 510 MiB) with:

```sh
cargo run --release --example download_models -- models
cargo run --release --example ocr -- models selection.png cpu    # or: --features metal ... metal
cargo test
```

The recognizer weights are [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) (Apache-2.0); the detector weights come from [manga-image-translator](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) (GPL-3.0). Pins and checksums are in `scripts/model-manifest.json`.

## Licence

GPL-3.0-only (see [LICENSE](LICENSE)), because the detector is ported from GPL-3.0 comic-text-detector. The Manga OCR parts keep their Apache-2.0 notice in [LICENSE-APACHE](LICENSE-APACHE) and [NOTICE](NOTICE).
