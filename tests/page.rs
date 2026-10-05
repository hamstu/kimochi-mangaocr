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
    assert!(
        page.groups.len() >= 12 && page.groups.len() <= 18,
        "{} blocks",
        page.groups.len()
    );
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
        assert!(!group.recognition.truncated);
    }
}
