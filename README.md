# mangaocr

Native Rust recognition for Japanese manga, using the original [Manga OCR](https://github.com/kha-white/manga-ocr) weights. The ViT encoder, causal BERT decoder, beam search, image processing, and text normalization run in Rust. No Python, PyTorch, ONNX Runtime, OpenCV, or external inference service is required. CPU is portable; Metal accelerates Apple Silicon and CUDA supports NVIDIA GPUs.

Recognize a selected text bubble or screenshot region. For full pages, `ComicTextDetector` locates text blocks and `recognize_page` returns independently recognized groups with boxes, direction, line polygons and generation metadata. Caller-supplied regions and custom `TextDetector` implementations remain supported.

```sh
cargo run --release --example download_models -- models
cargo run --release --example ocr -- models selection.png cpu
# Full manga page (detector + recognizer)
cargo run --release --features metal --example page -- models page.png metal
# Apple Silicon
cargo run --release --features metal --example ocr -- models selection.png metal
# NVIDIA (requires CUDA toolkit/drivers for Candle's CUDA backend)
cargo run --release --features cuda --example ocr -- models selection.png cuda
```

The Rust downloader pins `kha-white/manga-ocr-base` revision `aa6573bd10b0d446cbf622e29c3e084914df9741`, verifies SHA-256 for every file, and installs each verified file by rename. Recognizer weights are about 424 MiB; the pinned detector adds about 90 MiB. Existing valid files are reused; inference never accesses the network. If a download is interrupted, remove its `.part` file before retrying. Files remain in `models/`, excluded from Git.

## Library API

```toml
[dependencies]
mangaocr = { path = "../kimochi-mangaocr", features = ["metal"] }
```

Omit `metal` on other platforms or enable `cuda` for NVIDIA. Features are opt-in so the default build has no native GPU build requirements. `accelerate` optionally uses Apple's Accelerate BLAS for CPU computation on macOS.

```rust,no_run
use mangaocr::{MangaOcr, ModelPaths, OcrOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = MangaOcr::new(ModelPaths::from_dir("models"))?;
    println!("{:?}", engine.backend_info());
    let result = engine.ocr_file("selection.png", &OcrOptions::default())?;
    println!("{}", result.text);
    Ok(())
}
```

Load once and reuse the engine. `recognize(&RgbImage, &OcrOptions)` accepts decoded RGB screenshots. `decode_image(&[u8])` accepts PNG/JPEG/WebP bytes; `ocr_file` combines decoding and recognition. Recognition results implement Serde and contain text, generated token IDs, a mean sequence log probability, and a truncation flag. The score is **not calibrated OCR accuracy**. The model provides no character coordinates. Region boxes are original image pixels `[left, top, right, bottom]`, with exclusive right/bottom edges.

Defaults reproduce upstream's four beams, length penalty 2, no repeated trigrams, and maximum total length 300 (299 new tokens). Set `beam_width: 1` for faster greedy recognition. Generation is bounded to the decoder's position capacity; invalid options, boxes, empty images, and incompatible model architectures return errors. Exact uniform-color images return empty text by default, avoiding one obvious hallucination case. Disable `skip_uniform_images` to run them through the model.

`ModelPaths` accepts the original `pytorch_model.bin`, loaded using Candle's Rust tensor reader. It also accepts a `.safetensors` file with the same tensor names/shapes; set `paths.weights` explicitly. The pinned model's config and character vocabulary remain required. This implementation supports the original fixed architecture, not arbitrary Hugging Face models.

## GPU behavior

| Backend | Feature | Status |
| --- | --- | --- |
| CPU | none | Validated locally on Apple M5 Max |
| Apple GPU / Metal | `metal` | Real inference and corpus parity validated on Apple Silicon |
| NVIDIA / CUDA | `cuda` | Implemented via Candle; requires validation on NVIDIA hardware |
| AMD/Intel GPU | — | No backend included |

`Config::default()` requests `Backend::Auto`, preferring compiled-in Metal on macOS, then compiled-in CUDA, then CPU. `backend_info()` records the selected device and device-initialization fallback reasons. Explicit device requests return errors when unavailable. Auto fallback covers device initialization; subsequent model-loading or inference errors are returned rather than hidden. Metal uses the GPU, not Apple's Neural Engine.

On this Apple M5 Max, the upstream example takes roughly **64–65 ms warm on Metal versus about 470 ms on CPU** with four beams and F32 inference. These are local measurements, not cross-device performance guarantees. The first Metal use can take seconds for kernel compilation. Use release builds, warm up the engine, and measure your own crops:

```sh
cargo run --release --features metal --example benchmark -- models selection.png metal
```

Attention state is cached across generated tokens; cross-attention projections are computed once per crop. Beam selection reorders only decoder caches and selects the top candidates without sorting the entire vocabulary. Multiple page regions run sequentially to bound memory use; regions are not batch-encoded.

## Tauri integration

The sibling `../kimochi-tauri` exposes this crate as **Local (Manga OCR)** in its OCR settings. Its build stages verified models and licenses, enables Metal on macOS, and reuses a lazily initialized engine on the retained-frame blocking worker. Results use the shared grouped OCR contract: each text block gets its own overlay and transcript row, with explicit direction and geometric fallback for other engines. For another host, resolve model files from the app's resource/cache directory, initialize once, and run decoding/inference on a blocking worker:

```rust,ignore
use std::sync::{Arc, Mutex};
use mangaocr::{MangaOcr, ModelPaths, OcrOptions, Recognition};

struct MangaState(Arc<Mutex<MangaOcr>>);

#[tauri::command]
async fn manga_ocr(
    state: tauri::State<'_, MangaState>,
    png: Vec<u8>,
) -> Result<Recognition, String> {
    let engine = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || {
        let image = mangaocr::decode_image(&png).map_err(|e| e.to_string())?;
        engine.lock().map_err(|e| e.to_string())?
            .recognize(&image, &OcrOptions::default())
            .map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}
```

Use a bounded queue or coalesce pending captures for continuous scanning: generation is synchronous and does not currently support cancellation mid-crop. Bundle the weights, vocabulary, config, and applicable licenses; the host owns provisioning and download UI. No separate native inference runtime needs bundling.

## Detector assessment and recognition limits

[comic-text-detector](https://github.com/dmMaze/comic-text-detector) is integrated as a native Candle graph. Its pinned ONNX file is parsed using Rust protobuf definitions and executed with Rust tensor operations; ONNX Runtime and OpenCV are not used. YOLO block confidence/NMS and DBNet connected-component lines provide block boxes and orientation. Line polygons are axis-aligned bounds, not rotated contours. Blocks generally correspond to speech bubbles, captions or sound effects, but this is text-block detection, not segmentation of bubble outlines. Panel order is a deterministic local ordering rather than an inferred manga reading order.

```rust,no_run
use mangaocr::{Backend, ComicTextDetector, DetectionOptions, MangaOcr, ModelPaths, OcrOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = mangaocr::decode_image(&std::fs::read("page.png")?)?;
    let mut recognizer = MangaOcr::new(ModelPaths::from_dir("models"))?;
    let mut detector = ComicTextDetector::new("models/comic-text-detector.onnx", Backend::Auto)?;
    let page = recognizer.recognize_page(&image, &mut detector, &OcrOptions::default(), &DetectionOptions::default())?;
    println!("{}", serde_json::to_string_pretty(&page)?);
    Ok(())
}
```

Detection uses a 1024-pixel letterbox; recognition crops the original-resolution image, with small padding. Very small text on huge pages can be missed by detector downsampling. Configure confidence/NMS/line thresholds and a bounded region limit through `DetectionOptions`. Detection confidence and recognition sequence scores are separate and neither is calibrated transcription accuracy.

Manga OCR handles vertical/horizontal text and multi-line bubbles without line splitting. A whole page reduced to 224×224 loses text detail; recognize detected or supplied blocks instead. The generative model can hallucinate text on drawings/empty regions and omit or repeat text in long crops. Beam search does not remove those model limits. Transparent input is converted to RGB without compositing; composite against the intended background before recognition when needed.

## Validation

```sh
cargo test
cargo clippy --all-targets --features metal -- -D warnings
./scripts/download_fixtures.sh
cargo test --release --test inference -- --ignored --nocapture
MANGAOCR_TEST_BACKEND=metal cargo test --release --features metal \
    --test inference -- --ignored --nocapture
```

The optional inference test loads real weights, checks the 12 pinned upstream cases, and validates region geometry, serialization, uniform-image skipping, greedy decoding, and truncation. Artwork is downloaded for local testing and is not checked into the repository. Unit tests verify Pillow-compatible grayscale/antialiased bilinear processing using reference hashes for upsampling, downsampling, and identity resize, plus Unicode normalization, generation option validation, and repeated-ngram suppression.

On both CPU and Metal, **all 12 cases match upstream when the decoded input pixels are identical**. On the original JPEGs, 11 of 12 match upstream. Rust JPEG decoding and Pillow/libjpeg differ by up to one pixel level on a sensitive crop (`01.jpg`); this changes its quotation marks/ending, and its known native result is tracked explicitly in the test. PNG screenshots avoid JPEG decoder differences. This is not a claim of perfect transcription: the test measures upstream parity, including upstream's own errors.

For exact pixel parity, these optional development scripts require Pillow (and the Python reference additionally needs PyTorch, Transformers 4.x, NumPy, and jaconv). Python is never used by the Rust library, downloader, or deployed app:

```sh
python scripts/prepare_lossless_fixtures.py
MANGAOCR_LOSSLESS_FIXTURES=1 cargo test --release --test inference -- --ignored
MANGAOCR_LOSSLESS_FIXTURES=1 MANGAOCR_TEST_BACKEND=metal \
    cargo test --release --features metal --test inference -- --ignored
python scripts/reference.py models tests/fixtures/images/00.jpg
```

The combined library is GPL-3.0-only because the detector is GPL-3.0. The recognizer retains its Apache-2.0 provenance in `LICENSE-APACHE` and `NOTICE`. See the [model card](https://huggingface.co/kha-white/manga-ocr-base).

Detector validation uses a synthetic input and sampled ONNX Runtime reference outputs in `tests/fixtures/detector-reference.json`. It checks both CPU and Metal, including a workaround for Candle 0.9.2's CPU convolution layout bug when channels equal both spatial dimensions. For real page validation:

```sh
cargo test --release detector_graph_matches_reference -- --ignored
MANGAOCR_TEST_BACKEND=metal cargo test --release --features metal detector_graph_matches_reference -- --ignored
cargo test --release --features metal --test page -- --ignored
```

On the downloaded detector example page, Metal detected and recognized 15 blocks in about 1.54 seconds on Apple M5 Max. CUDA uses native Candle operations but still requires validation on NVIDIA hardware.
