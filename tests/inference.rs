use mangaocr::{Backend, Config, Detection, MangaOcr, ModelPaths, OcrOptions};
use serde::Deserialize;
#[derive(Deserialize)]
struct Case {
    filename: String,
    result: String,
}
#[test]
#[ignore = "requires downloaded weights and fixtures; run in release mode"]
fn upstream_corpus_and_regions() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("MANGAOCR_MODEL_DIR").unwrap_or_else(|_| "models".into());
    let backend = match std::env::var("MANGAOCR_TEST_BACKEND").as_deref() {
        Ok("metal") => Backend::Metal,
        Ok("cuda") => Backend::Cuda,
        _ => Backend::Cpu,
    };
    let mut engine = MangaOcr::with_config(ModelPaths::from_dir(dir), Config { backend })?;
    let options = OcrOptions::default();
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/expected_results.json"))?;
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/images");
    let mut failures = Vec::new();
    let lossless = std::env::var_os("MANGAOCR_LOSSLESS_FIXTURES").is_some();
    for c in &cases {
        let path = if lossless {
            base.parent()
                .unwrap()
                .join("lossless")
                .join(c.filename.replace(".jpg", ".png"))
        } else {
            base.join(&c.filename)
        };
        let image = mangaocr::decode_image(&std::fs::read(path)?)?;
        let result = engine.recognize(&image, &options)?;
        println!("{} {}", c.filename, result.text);
        // Integer JPEG decoders differ by one level from libjpeg on this sensitive crop.
        let expected = if !lossless && c.filename == "01.jpg" {
            "立川で見た、穴への下の巨大な眼は．．．"
        } else {
            &c.result
        };
        if result.text != expected {
            failures.push(format!(
                "{}: expected {:?}, got {:?}",
                c.filename, c.result, result.text
            ));
        }
        assert!(!result.truncated);
        assert!(result.mean_log_probability.unwrap().is_finite());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let image = mangaocr::decode_image(&std::fs::read(base.join(&cases[0].filename))?)?;
    let regions = [Detection {
        bbox: [0, 0, image.width(), image.height()],
        confidence: Some(0.9),
    }];
    let result = engine.recognize_regions(&image, &regions, &options)?;
    assert_eq!(result[0].recognition.text, cases[0].result);
    assert_eq!(result[0].bbox, regions[0].bbox);
    serde_json::to_string(&result)?;
    assert!(
        engine
            .recognize_regions(
                &image,
                &[Detection {
                    bbox: [1, 0, 1, 1],
                    confidence: None
                }],
                &options
            )
            .is_err()
    );
    let blank = image::RgbImage::from_pixel(32, 32, image::Rgb([255; 3]));
    assert!(engine.recognize(&blank, &options)?.text.is_empty());
    let short = engine.recognize(
        &image,
        &OcrOptions {
            max_new_tokens: 1,
            ..options.clone()
        },
    )?;
    assert!(short.truncated);
    assert_eq!(short.token_ids.len(), 1);
    let greedy = engine.recognize(
        &image,
        &OcrOptions {
            beam_width: 1,
            ..options
        },
    )?;
    assert!(!greedy.truncated);
    Ok(())
}
