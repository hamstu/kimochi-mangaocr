use mangaocr::{Backend, Config, MangaOcr, ModelPaths, OcrOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: cargo run --release --example ocr -- MODELS IMAGE [cpu|metal|cuda] [BEAMS]"
                .into(),
        );
    }
    let backend = match args.get(3).map(String::as_str) {
        None | Some("auto") => Backend::Auto,
        Some("cpu") => Backend::Cpu,
        Some("metal") => Backend::Metal,
        Some("cuda") => Backend::Cuda,
        _ => return Err("unknown backend".into()),
    };
    let start = std::time::Instant::now();
    let mut engine = MangaOcr::with_config(ModelPaths::from_dir(&args[1]), Config { backend })?;
    eprintln!(
        "Loaded in {:?}; {:?}",
        start.elapsed(),
        engine.backend_info()
    );
    let options = OcrOptions {
        beam_width: args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(4),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = engine.ocr_file(&args[2], &options)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    eprintln!("Recognized in {:?}", start.elapsed());
    Ok(())
}
