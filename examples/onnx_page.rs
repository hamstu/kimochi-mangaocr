//! `onnx_page MODEL_DIR IMAGE cpu|directml ONNXRUNTIME_LIBRARY [ROUNDS]`: a page through ONNX
//! Runtime, its timings on stderr and its result as JSON on stdout (as `page` prints it).
use mangaocr::onnx::{OnnxPage, Provider};
use mangaocr::{DetectionOptions, ModelPaths, OcrOptions, PageDetector};
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() < 5 {
        return Err(
            "usage: onnx_page MODEL_DIR IMAGE cpu|directml ONNXRUNTIME_LIBRARY [ROUNDS]".into(),
        );
    }
    ort::init_from(&a[4]).commit()?;
    let provider = if a[3] == "directml" {
        Provider::DirectMl
    } else {
        Provider::Cpu
    };
    let rounds: usize = a.get(5).map_or(2, |r| r.parse().unwrap());
    let dir = std::path::Path::new(&a[1]);
    let start = Instant::now();
    let mut engine = OnnxPage::new(
        &ModelPaths::from_dir(dir),
        dir.join("comic-text-detector.onnx"),
        provider,
        OcrOptions::default().beam_width,
    )?;
    eprintln!("load {:?} on {provider:?}", start.elapsed());
    let image = mangaocr::decode_image(&std::fs::read(&a[2])?)?;
    let mut result = None;
    for round in 0..rounds {
        let start = Instant::now();
        let (groups, _) = engine
            .detector
            .detect_page_with_mask(&image, &DetectionOptions::default())?;
        let detect = start.elapsed();
        let start = Instant::now();
        let page = engine.recognize_region(
            &image,
            &OcrOptions::default(),
            &DetectionOptions::default(),
        )?;
        let total = start.elapsed();
        let lines: usize = page.groups.iter().map(|g| g.lines.len()).sum();
        eprintln!(
            "round {round}: detect {detect:?}; page {total:?} => read ~{:?}; {} groups, {lines} lines (detector {} groups)",
            total.saturating_sub(detect),
            page.groups.len(),
            groups.len()
        );
        result = Some(page);
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
