use mangaocr::{
    Backend, ComicTextDetector, DetectionOptions, MangaOcr, ModelPaths, OcrOptions, Orientation,
};
#[test]
#[ignore = "requires downloaded model files and page fixture; use release mode"]
fn full_page_preserves_independent_text_groups() {
    let image = mangaocr::decode_image(&std::fs::read("tests/fixtures/page.jpg").unwrap()).unwrap();
    let backend = if std::env::var("MANGAOCR_TEST_BACKEND").as_deref() == Ok("metal") {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    let mut detector = ComicTextDetector::new("models/comic-text-detector.onnx", backend).unwrap();
    let mut recognizer = MangaOcr::with_config(
        ModelPaths::from_dir("models"),
        mangaocr::Config {
            backend,
            ..Default::default()
        },
    )
    .unwrap();
    let page = recognizer
        .recognize_page(
            &image,
            &mut detector,
            &OcrOptions::default(),
            &DetectionOptions::default(),
        )
        .unwrap();
    // The detector's blocks, and, as mokuro reads them, blocks of the lines no block took: the
    // page's margin notes, set in type too small for the block head to find.
    let detected = page
        .groups
        .iter()
        .filter(|g| g.detection_confidence > 0.)
        .count();
    assert!((12..=18).contains(&detected), "{detected} detected blocks");
    assert!(page.groups.len() <= 24, "{} blocks", page.groups.len());
    assert!(
        page.groups
            .iter()
            .any(|g| g.orientation == Orientation::Vertical)
    );
    assert_eq!(
        page.full_text,
        page.groups
            .iter()
            .map(|g| g.recognition.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mut ids = std::collections::HashSet::new();
    for group in page.groups {
        assert!(ids.insert(group.id));
        let [l, t, r, b] = group.bbox;
        assert!(l < r && t < b && r <= image.width() && b <= image.height());
        assert!(!group.recognition.text.trim().is_empty());
        assert_eq!(group.lines.len(), group.line_polygons.len());
        assert_eq!(group.lines.concat(), group.recognition.text);
        assert!(!group.recognition.truncated);
    }
}

/// A snip of one or two lines — what a reader draws around a single bubble — must still be
/// read. Upstream's recognition crops are exactly that. Stretched to fill the detector's
/// square, as the reference preprocessing does, one of these twelve came back right.
#[test]
#[ignore = "requires downloaded model files; use release mode"]
fn snips_are_read_like_the_crops_they_are() {
    #[derive(serde::Deserialize)]
    struct Case {
        filename: String,
        result: String,
    }
    // Punctuation is where JPEG decoders and the two paths legitimately differ.
    let words = |s: &str| {
        s.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
    };
    let backend = if std::env::var("MANGAOCR_TEST_BACKEND").as_deref() == Ok("metal") {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    let mut detector = ComicTextDetector::new("models/comic-text-detector.onnx", backend).unwrap();
    let mut recognizer = MangaOcr::with_config(
        ModelPaths::from_dir("models"),
        mangaocr::Config {
            backend,
            ..Default::default()
        },
    )
    .unwrap();
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/expected_results.json")).unwrap();
    let mut detected = 0;
    let mut exact = 0;
    for c in &cases {
        let path = format!("tests/fixtures/images/{}", c.filename);
        let image = mangaocr::decode_image(&std::fs::read(path).unwrap()).unwrap();
        let options = (OcrOptions::default(), DetectionOptions::default());
        let page = recognizer
            .recognize_page(&image, &mut detector, &options.0, &options.1)
            .unwrap();
        detected += usize::from(!page.groups.is_empty());
        let region = recognizer
            .recognize_region(&image, &mut detector, &options.0, &options.1)
            .unwrap();
        println!("{} {:?} / {:?}", c.filename, region.full_text, c.result);
        assert!(!region.groups.is_empty(), "{} read nothing", c.filename);
        exact += usize::from(words(&region.full_text) == words(&c.result));
    }
    assert!(
        detected >= 9,
        "the detector found text in {detected} of {} snips",
        cases.len()
    );
    assert!(exact >= 9, "{exact} of {} snips read exactly", cases.len());
}

/// A page with nothing the detector calls text is not read as one block of invented text.
#[test]
#[ignore = "requires downloaded model files; use release mode"]
fn a_textless_image_reads_as_nothing() {
    let backend = if std::env::var("MANGAOCR_TEST_BACKEND").as_deref() == Ok("metal") {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    let mut detector = ComicTextDetector::new("models/comic-text-detector.onnx", backend).unwrap();
    let mut recognizer = MangaOcr::with_config(
        ModelPaths::from_dir("models"),
        mangaocr::Config {
            backend,
            ..Default::default()
        },
    )
    .unwrap();
    let image = image::RgbImage::from_fn(400, 300, |x, y| {
        let v = if ((x / 40) + (y / 40)) % 2 == 0 {
            30
        } else {
            220
        };
        image::Rgb([v, v, v])
    });
    let page = recognizer
        .recognize_region(
            &image,
            &mut detector,
            &OcrOptions::default(),
            &DetectionOptions::default(),
        )
        .unwrap();
    assert!(page.groups.is_empty(), "{:?}", page.full_text);
}
