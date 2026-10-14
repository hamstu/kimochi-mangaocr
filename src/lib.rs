//! Native Rust Japanese manga recognition. Reuse [`MangaOcr`] on a worker thread.
//! Recognize selections or detected page groups with native CPU/Metal/CUDA inference.
mod detector;
mod detector_graph;
mod generation;
pub use detector::{ComicTextDetector, DetectedGroup, DetectionOptions, Orientation};
mod model;
mod preprocess;
mod text;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
pub use text::post_process;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("unsupported model contract: {0}")]
    ModelContract(String),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub weights: PathBuf,
    pub vocab: PathBuf,
    pub config: PathBuf,
}
impl ModelPaths {
    pub fn from_dir(dir: impl AsRef<Path>) -> Self {
        let d = dir.as_ref();
        Self {
            weights: d.join("pytorch_model.bin"),
            vocab: d.join("vocab.txt"),
            config: d.join("config.json"),
        }
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    #[default]
    Auto,
    Cpu,
    Metal,
    Cuda,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub requested: Backend,
    pub selected: Backend,
    pub fallback_reasons: Vec<String>,
}
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub backend: Backend,
}
#[derive(Debug, Clone)]
pub struct OcrOptions {
    /// Number of active beams, 1..=16. One selects greedy decoding.
    pub beam_width: usize,
    /// Excludes the initial CLS token. Upstream max_length=300 means 299 here.
    pub max_new_tokens: usize,
    pub length_penalty: f64,
    /// Zero disables n-gram blocking.
    pub no_repeat_ngram_size: usize,
    /// Exact solid-color images return empty text without running the model.
    pub skip_uniform_images: bool,
}
impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_new_tokens: 299,
            length_penalty: 2.,
            no_repeat_ngram_size: 3,
            skip_uniform_images: true,
        }
    }
}
impl OcrOptions {
    fn validate(&self) -> Result<()> {
        if !(1..=16).contains(&self.beam_width)
            || !(1..=511).contains(&self.max_new_tokens)
            || !self.length_penalty.is_finite()
            || self.length_penalty < 0.
            || self.no_repeat_ngram_size > 512
        {
            return Err(Error::InvalidInput("beam width must be 1..=16, token limit 1..=511, length penalty finite and nonnegative, n-gram size <=512".into()));
        }
        Ok(())
    }
}
/// Original pixels, exclusive right/bottom coordinates.
pub type BoundingBox = [u32; 4];
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub confidence: Option<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recognition {
    pub text: String,
    /// Generated vocabulary IDs, excluding CLS and EOS. Includes any emitted special tokens.
    pub token_ids: Vec<u32>,
    /// Mean log probability of the selected sequence, including EOS when present.
    /// This is a model score, not calibrated OCR accuracy.
    pub mean_log_probability: Option<f64>,
    /// True when generation reached the token budget without EOS.
    pub truncated: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    pub bbox: BoundingBox,
    pub detection_confidence: Option<f32>,
    pub recognition: Recognition,
}
/// Adapter for a separately supplied page detector. Returned order is preserved.
pub trait TextDetector {
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>>;
}
/// Model and immutable weights. `&mut self` leaves room for reusable buffers;
/// keep it behind a Mutex or on a dedicated worker for Tauri.
pub struct MangaOcr {
    model: model::Model,
    vocab: Vec<String>,
    device: Device,
    backend: BackendInfo,
}
fn device(backend: Backend) -> Result<Device> {
    match backend {
        Backend::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "metal")]
        Backend::Metal => Ok(Device::new_metal(0)?),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Device::new_cuda(0)?),
        _ => Err(Error::InvalidInput(format!(
            "backend {backend:?} not compiled in"
        ))),
    }
}
fn select_backend(requested: Backend) -> Result<(Device, BackendInfo)> {
    let candidates = if requested == Backend::Auto {
        let mut c = Vec::new();
        if cfg!(all(target_os = "macos", feature = "metal")) {
            c.push(Backend::Metal);
        }
        if cfg!(feature = "cuda") {
            c.push(Backend::Cuda);
        }
        c.push(Backend::Cpu);
        c
    } else {
        vec![requested]
    };
    let mut reasons = Vec::new();
    for selected in candidates {
        match device(selected) {
            Ok(d) => {
                return Ok((
                    d,
                    BackendInfo {
                        requested,
                        selected,
                        fallback_reasons: reasons,
                    },
                ));
            }
            Err(e) if requested == Backend::Auto => reasons.push(e.to_string()),
            Err(e) => return Err(e),
        }
    }
    unreachable!("CPU always available")
}
fn validate_model(path: &Path) -> Result<()> {
    let c: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
    for (pointer, expected) in [
        ("/encoder/hidden_size", 768),
        ("/encoder/intermediate_size", 3072),
        ("/encoder/num_hidden_layers", 12),
        ("/encoder/num_attention_heads", 12),
        ("/encoder/patch_size", 16),
        ("/encoder/image_size", 224),
        ("/encoder/num_channels", 3),
        ("/decoder/hidden_size", 768),
        ("/decoder/intermediate_size", 3072),
        ("/decoder/num_hidden_layers", 2),
        ("/decoder/num_attention_heads", 12),
        ("/decoder/vocab_size", 6144),
        ("/decoder/max_position_embeddings", 512),
        ("/decoder/type_vocab_size", 2),
        ("/decoder_start_token_id", 2),
        ("/eos_token_id", 3),
        ("/pad_token_id", 0),
    ] {
        if c.pointer(pointer).and_then(|v| v.as_u64()) != Some(expected) {
            return Err(Error::ModelContract(format!(
                "{pointer} must equal {expected}"
            )));
        }
    }
    for (p, v) in [
        ("/encoder/model_type", "vit"),
        ("/decoder/model_type", "bert"),
        ("/encoder/hidden_act", "gelu"),
        ("/decoder/hidden_act", "gelu"),
        ("/decoder/position_embedding_type", "absolute"),
    ] {
        if c.pointer(p).and_then(|x| x.as_str()) != Some(v) {
            return Err(Error::ModelContract(format!("{p} must equal {v}")));
        }
    }
    for p in ["/encoder/layer_norm_eps", "/decoder/layer_norm_eps"] {
        if c.pointer(p).and_then(|v| v.as_f64()) != Some(1e-12) {
            return Err(Error::ModelContract(format!("{p} must equal 1e-12")));
        }
    }
    if c.pointer("/decoder/is_decoder").and_then(|v| v.as_bool()) != Some(true)
        || c.pointer("/decoder/add_cross_attention")
            .and_then(|v| v.as_bool())
            != Some(true)
    {
        return Err(Error::ModelContract(
            "expected causal cross-attention decoder".into(),
        ));
    }
    Ok(())
}
/// Decode PNG/JPEG/WebP in Rust. JPEG uses an accurate integer decoder to
/// minimize differences from upstream Pillow/libjpeg on sensitive crops.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.starts_with(&[0xff, 0xd8]) {
        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
        let pixels = decoder
            .decode()
            .map_err(|e| Error::InvalidInput(format!("JPEG decoding: {e}")))?;
        let info = decoder
            .info()
            .ok_or_else(|| Error::InvalidInput("missing JPEG metadata".into()))?;
        let (w, h) = (u32::from(info.width), u32::from(info.height));
        let rgb = match info.pixel_format {
            jpeg_decoder::PixelFormat::L8 => pixels.into_iter().flat_map(|v| [v; 3]).collect(),
            jpeg_decoder::PixelFormat::RGB24 => pixels,
            _ => return Ok(image::load_from_memory(bytes)?.to_rgb8()),
        };
        return RgbImage::from_raw(w, h, rgb)
            .ok_or_else(|| Error::InvalidInput("invalid JPEG dimensions".into()));
    }
    Ok(image::load_from_memory(bytes)?.to_rgb8())
}
impl MangaOcr {
    pub fn new(paths: ModelPaths) -> Result<Self> {
        Self::with_config(paths, Config::default())
    }
    pub fn with_config(paths: ModelPaths, config: Config) -> Result<Self> {
        validate_model(&paths.config)?;
        let vocab: Vec<String> = std::fs::read_to_string(&paths.vocab)?
            .lines()
            .map(str::to_owned)
            .collect();
        if vocab.len() != 6144 || vocab[..5] != ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] {
            return Err(Error::ModelContract(
                "expected upstream 6144-token Japanese character vocabulary".into(),
            ));
        }
        let (device, backend) = select_backend(config.backend)?;
        let vb = if paths
            .weights
            .extension()
            .is_some_and(|x| x == "safetensors")
        {
            // Read-only bytes avoid unsafe mmap and keep model loading native Rust.
            VarBuilder::from_buffered_safetensors(
                std::fs::read(&paths.weights)?,
                DType::F32,
                &device,
            )?
        } else {
            VarBuilder::from_pth(&paths.weights, DType::F32, &device)?
        };
        let model = model::Model::load(vb)?;
        Ok(Self {
            model,
            vocab,
            device,
            backend,
        })
    }
    pub fn backend_info(&self) -> &BackendInfo {
        &self.backend
    }
    pub fn ocr_file(
        &mut self,
        path: impl AsRef<Path>,
        options: &OcrOptions,
    ) -> Result<Recognition> {
        self.recognize(&decode_image(&std::fs::read(path)?)?, options)
    }
    pub fn recognize(&mut self, image: &RgbImage, options: &OcrOptions) -> Result<Recognition> {
        options.validate()?;
        preprocess::validate(image)?;
        if options.skip_uniform_images && image.pixels().all(|p| p == image.get_pixel(0, 0)) {
            return Ok(Recognition {
                text: String::new(),
                token_ids: Vec::new(),
                mean_log_probability: None,
                truncated: false,
            });
        }
        let pixels = preprocess::pixels(image, &self.device)?;
        let h = generation::generate(&self.model, &pixels, &self.device, options)?;
        let mean_log_probability = Some(h.score / (h.ids.len() - 1) as f64);
        let ids: Vec<_> = h.ids.into_iter().skip(1).filter(|&x| x != 3).collect();
        Ok(Recognition {
            text: post_process(&text::decode(&ids, &self.vocab)),
            token_ids: ids,
            mean_log_probability,
            truncated: !h.ended,
        })
    }
    /// Validate every box before inference. Regions retain the caller's reading order.
    pub fn recognize_regions(
        &mut self,
        image: &RgbImage,
        regions: &[Detection],
        options: &OcrOptions,
    ) -> Result<Vec<TextRegion>> {
        options.validate()?;
        preprocess::validate(image)?;
        for r in regions {
            let [l, t, right, bottom] = r.bbox;
            if l >= right
                || t >= bottom
                || right > image.width()
                || bottom > image.height()
                || r.confidence
                    .is_some_and(|x| !x.is_finite() || !(0. ..=1.).contains(&x))
            {
                return Err(Error::InvalidInput(format!("invalid region: {:?}", r.bbox)));
            }
        }
        regions
            .iter()
            .map(|r| {
                let [l, t, right, bottom] = r.bbox;
                let crop = image::imageops::crop_imm(image, l, t, right - l, bottom - t).to_image();
                Ok(TextRegion {
                    bbox: r.bbox,
                    detection_confidence: r.confidence,
                    recognition: self.recognize(&crop, options)?,
                })
            })
            .collect()
    }
    pub fn ocr_with_detector(
        &mut self,
        image: &RgbImage,
        detector: &mut impl TextDetector,
        options: &OcrOptions,
    ) -> Result<Vec<TextRegion>> {
        options.validate()?;
        preprocess::validate(image)?;
        let regions = detector.detect(image)?;
        self.recognize_regions(image, &regions, options)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engine_can_move_to_worker_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<MangaOcr>();
    }
    #[test]
    fn invalid_generation_options() {
        for o in [
            OcrOptions {
                beam_width: 0,
                ..Default::default()
            },
            OcrOptions {
                max_new_tokens: 512,
                ..Default::default()
            },
            OcrOptions {
                length_penalty: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(o.validate().is_err());
        }
    }
}

/// A detector seam for full-page recognition, independent of the detector model.
pub trait PageDetector {
    fn detect_page(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<Vec<DetectedGroup>>;
}
impl PageDetector for ComicTextDetector {
    fn detect_page(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<Vec<DetectedGroup>> {
        self.detect_groups(image, options)
    }
}

/// A recognized page block. Geometry is in the original capture's pixel space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageGroup {
    pub id: String,
    pub bbox: BoundingBox,
    pub orientation: Orientation,
    pub line_polygons: Vec<[[u32; 2]; 4]>,
    pub detection_confidence: f32,
    pub recognition: Recognition,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResult {
    pub groups: Vec<PageGroup>,
    pub full_text: String,
}
impl MangaOcr {
    pub fn recognize_page<D: PageDetector + ?Sized>(
        &mut self,
        image: &RgbImage,
        detector: &mut D,
        options: &OcrOptions,
        detection: &DetectionOptions,
    ) -> Result<PageResult> {
        options.validate()?;
        preprocess::validate(image)?;
        detection.validate()?;
        let regions = detector.detect_page(image, detection)?;
        if regions.len() > detection.max_regions {
            return Err(Error::InvalidInput(
                "page detector exceeded region limit".into(),
            ));
        }
        for r in &regions {
            let [l, t, right, bottom] = r.bbox;
            if l >= right
                || t >= bottom
                || right > image.width()
                || bottom > image.height()
                || !r.confidence.is_finite()
            {
                return Err(Error::InvalidInput(format!(
                    "invalid detected page region: {:?}",
                    r.bbox
                )));
            }
        }
        let mut groups = Vec::with_capacity(regions.len());
        for (i, r) in regions.into_iter().enumerate() {
            let [l, t, right, bottom] = r.bbox;
            let padding = ((right - l).min(bottom - t) as f32 * 0.04).round().max(2.) as u32;
            let x = l.saturating_sub(padding);
            let y = t.saturating_sub(padding);
            let end_x = right.saturating_add(padding).min(image.width());
            let end_y = bottom.saturating_add(padding).min(image.height());
            let crop = image::imageops::crop_imm(image, x, y, end_x - x, end_y - y).to_image();
            let recognition = self.recognize(&crop, options)?;
            if recognition.text.is_empty() {
                continue;
            }
            groups.push(PageGroup {
                id: format!("block-{i}"),
                bbox: r.bbox,
                orientation: r.orientation,
                line_polygons: r.line_polygons,
                detection_confidence: r.confidence,
                recognition,
            });
        }
        let full_text = groups
            .iter()
            .map(|g| g.recognition.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(PageResult { groups, full_text })
    }

    /// `recognize_page` for an image that may be a whole page or a snip of a few words.
    ///
    /// The detector needs the context of a page around text to call it text, and a snip
    /// drawn tight around one line (a sound effect, a single caption) can give it none: it
    /// finds no block. Manga OCR itself was trained on exactly such crops, so when the
    /// detector comes back empty the whole image is read as one block, and kept only if
    /// the model is sure of it. Real text reads at a mean token log-probability near zero
    /// (-0.003 to -0.11 on the upstream corpus and snips of whole bubbles); a page, or
    /// artwork, read as one block is invented text below -0.7.
    pub fn recognize_region<D: PageDetector + ?Sized>(
        &mut self,
        image: &RgbImage,
        detector: &mut D,
        options: &OcrOptions,
        detection: &DetectionOptions,
    ) -> Result<PageResult> {
        let page = self.recognize_page(image, detector, options, detection)?;
        if !page.groups.is_empty() {
            return Ok(page);
        }
        let recognition = self.recognize(image, options)?;
        let sure = recognition
            .mean_log_probability
            .is_some_and(|p| p >= WHOLE_IMAGE_MIN_LOG_PROBABILITY);
        if recognition.text.is_empty() || !sure || recognition.truncated {
            return Ok(page);
        }
        let (width, height) = image.dimensions();
        let group = PageGroup {
            id: "block-0".into(),
            bbox: [0, 0, width, height],
            orientation: if height > width {
                Orientation::Vertical
            } else {
                Orientation::Horizontal
            },
            line_polygons: Vec::new(),
            // Not a detection: the detector found nothing here.
            detection_confidence: 0.,
            recognition,
        };
        Ok(PageResult {
            full_text: group.recognition.text.clone(),
            groups: vec![group],
        })
    }
}

/// Below this mean token log-probability a whole-image read is invented text; see
/// `MangaOcr::recognize_region`.
pub const WHOLE_IMAGE_MIN_LOG_PROBABILITY: f64 = -0.3;
