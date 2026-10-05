use mangaocr::{Backend, ComicTextDetector, DetectionOptions, MangaOcr, ModelPaths, OcrOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() < 3 {
        return Err("usage: page MODEL_DIR IMAGE [cpu|metal|cuda]".into());
    }
    let b = match a.get(3).map(String::as_str) {
        Some("metal") => Backend::Metal,
        Some("cuda") => Backend::Cuda,
        Some("cpu") => Backend::Cpu,
        _ => Backend::Auto,
    };
    let mut detector = ComicTextDetector::new(
        std::path::Path::new(&a[1]).join("comic-text-detector.onnx"),
        b,
    )?;
    let mut ocr =
        MangaOcr::with_config(ModelPaths::from_dir(&a[1]), mangaocr::Config { backend: b })?;
    let im = mangaocr::decode_image(&std::fs::read(&a[2])?)?;
    let start = std::time::Instant::now();
    let result = ocr.recognize_page(
        &im,
        &mut detector,
        &OcrOptions::default(),
        &DetectionOptions::default(),
    )?;
    eprintln!(
        "{} blocks in {:?} on {:?}",
        result.groups.len(),
        start.elapsed(),
        detector.backend_info()
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
