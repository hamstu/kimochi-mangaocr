use mangaocr::{Backend, Config, MangaOcr, ModelPaths, OcrOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: benchmark MODELS IMAGE cpu|metal|cuda".into());
    }
    let backend = match args[3].as_str() {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        _ => return Err("invalid backend".into()),
    };
    let mut engine = MangaOcr::with_config(ModelPaths::from_dir(&args[1]), Config { backend })?;
    let image = image::open(&args[2])?.to_rgb8();
    for i in 0..6 {
        let start = std::time::Instant::now();
        let text = engine.recognize(&image, &OcrOptions::default())?;
        println!(
            "{} {:.1} ms {}",
            if i == 0 { "cold" } else { "warm" },
            start.elapsed().as_secs_f64() * 1000.,
            text.text
        );
    }
    Ok(())
}
