//! Comic Text Detector page regions, executed natively with Candle.
use crate::blocks::{self, Candidate, Language};
use crate::detector_graph::DetectorGraph;
use crate::{Backend, BackendInfo, BoundingBox, Detection, Error, Result, TextDetector};
use candle_core::{Device, Tensor};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::path::Path;
const SIZE: usize = 1024;
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Orientation {
    Horizontal,
    Vertical,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedGroup {
    pub bbox: BoundingBox,
    /// The YOLO block's; zero for a block made of lines no YOLO block took.
    pub confidence: f32,
    pub orientation: Orientation,
    /// In reading order.
    pub line_polygons: Vec<[[u32; 2]; 4]>,
    /// The lines' mean thickness in page pixels: mokuro's `font_size`.
    pub font_size: f32,
    pub language: Language,
}
#[derive(Debug, Clone)]
pub struct DetectionOptions {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub line_threshold: f32,
    pub max_regions: usize,
}
impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.4,
            nms_threshold: 0.35,
            line_threshold: 0.3,
            max_regions: 128,
        }
    }
}
impl DetectionOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if [
            self.confidence_threshold,
            self.nms_threshold,
            self.line_threshold,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !(1..=1024).contains(&self.max_regions)
        {
            return Err(Error::InvalidInput(
                "detector thresholds must be in [0,1] and region limit in 1..=1024".into(),
            ));
        }
        Ok(())
    }
}
pub struct ComicTextDetector {
    graph: DetectorGraph,
    device: Device,
    backend: BackendInfo,
}
fn iou(a: BoundingBox, b: BoundingBox) -> f64 {
    let area = |r: BoundingBox| f64::from(r[2] - r[0]) * f64::from(r[3] - r[1]);
    let w = a[2].min(b[2]).saturating_sub(a[0].max(b[0]));
    let h = a[3].min(b[3]).saturating_sub(a[1].max(b[1]));
    let inter = f64::from(w) * f64::from(h);
    inter / (area(a) + area(b) - inter).max(1.)
}
/// Top-left letterbox with black padding and half-pixel bilinear sampling,
/// matching the detector's OpenCV preprocessing without OpenCV.
///
/// The reference stretches every image to fill the 1024 square. That suits the pages the
/// detector was trained on, which are shrunk to fit, but a snip of a single bubble is
/// blown up four- or eightfold: its glyphs come out over 100 pixels tall, far outside
/// anything the model learned, and it finds nothing or splits columns into fragments
/// ("めぐみさ", "みさんに"). With `upscale` off an image smaller than the square keeps its
/// own scale, so a snip looks like a corner of an ordinary page. Pages are unaffected.
fn pixels(image: &RgbImage, device: &Device, upscale: bool) -> Result<(Tensor, u32, u32)> {
    crate::preprocess::validate(image)?;
    let fit = SIZE as f64 / f64::from(image.width().max(image.height()));
    let ratio = if upscale { fit } else { fit.min(1.) };
    let w = (f64::from(image.width()) * ratio).round().max(1.) as u32;
    let h = (f64::from(image.height()) * ratio).round().max(1.) as u32;
    let mut data = vec![0f32; 3 * SIZE * SIZE];
    for y in 0..h {
        for x in 0..w {
            let sx = ((f64::from(x) + 0.5) * f64::from(image.width()) / f64::from(w) - 0.5).max(0.);
            let sy =
                ((f64::from(y) + 0.5) * f64::from(image.height()) / f64::from(h) - 0.5).max(0.);
            let x0 = (sx.floor() as u32).min(image.width() - 1);
            let y0 = (sy.floor() as u32).min(image.height() - 1);
            let x1 = (x0 + 1).min(image.width() - 1);
            let y1 = (y0 + 1).min(image.height() - 1);
            let ax = sx - sx.floor();
            let ay = sy - sy.floor();
            for c in 0..3 {
                let a = f64::from(image.get_pixel(x0, y0)[c]) * (1. - ax)
                    + f64::from(image.get_pixel(x1, y0)[c]) * ax;
                let b = f64::from(image.get_pixel(x0, y1)[c]) * (1. - ax)
                    + f64::from(image.get_pixel(x1, y1)[c]) * ax;
                data[c * SIZE * SIZE + y as usize * SIZE + x as usize] =
                    (a * (1. - ay) + b * ay).round() as f32 / 255.;
            }
        }
    }
    Ok((Tensor::from_vec(data, (1, 3, SIZE, SIZE), device)?, w, h))
}
fn mapped(b: [f32; 4], image: &RgbImage, w: u32, h: u32) -> Option<BoundingBox> {
    if b.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let sx = image.width() as f32 / w as f32;
    let sy = image.height() as f32 / h as f32;
    let r = [
        (b[0] * sx).floor().clamp(0., image.width() as f32) as u32,
        (b[1] * sy).floor().clamp(0., image.height() as f32) as u32,
        (b[2] * sx).ceil().clamp(0., image.width() as f32) as u32,
        (b[3] * sy).ceil().clamp(0., image.height() as f32) as u32,
    ];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}
fn polygon(b: BoundingBox) -> [[u32; 2]; 4] {
    [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}
/// Share of a block that must lie inside a larger one for it to count as a duplicate.
const CONTAINED: f64 = 0.8;
fn area(r: BoundingBox) -> u64 {
    u64::from(r[2] - r[0]) * u64::from(r[3] - r[1])
}
fn intersection_fraction(a: BoundingBox, b: BoundingBox) -> f64 {
    let area = f64::from(a[2] - a[0]) * f64::from(a[3] - a[1]);
    f64::from(a[2].min(b[2]).saturating_sub(a[0].max(b[0])))
        * f64::from(a[3].min(b[3]).saturating_sub(a[1].max(b[1])))
        / area.max(1.)
}
/// Extract connected text-line regions from DBNet's shrink probability map.
/// Native axis-aligned polygons are deliberately retained as such; no fabricated
/// rotation or character positions are claimed.
fn lines(map: &[f32], threshold: f32, w: u32, h: u32) -> Vec<BoundingBox> {
    let mut seen = vec![false; SIZE * SIZE];
    let mut out = Vec::new();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let start = y * SIZE + x;
            if seen[start] || map[start] <= threshold {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![start];
            let mut bounds = [x, y, x + 1, y + 1];
            let mut count = 0;
            let mut score = 0.;
            while let Some(i) = stack.pop() {
                let px = i % SIZE;
                let py = i / SIZE;
                count += 1;
                score += map[i];
                bounds = [
                    bounds[0].min(px),
                    bounds[1].min(py),
                    bounds[2].max(px + 1),
                    bounds[3].max(py + 1),
                ];
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let nx = px as isize + dx;
                        let ny = py as isize + dy;
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                            continue;
                        }
                        let n = ny as usize * SIZE + nx as usize;
                        if !seen[n] && map[n] > threshold {
                            seen[n] = true;
                            stack.push(n);
                        }
                    }
                }
            }
            // As the reference's `boxes_from_bitmap`: the contour runs through the outer pixels'
            // centres, so the rectangle around it is a pixel short each way; it is grown by
            // DBNet's unclip distance (area × 1.5 / perimeter) and kept when its short side is two
            // pixels or more and the map is sure of it (mean above 0.6, `box_thresh`).
            let bw = (bounds[2] - bounds[0] - 1) as f32;
            let bh = (bounds[3] - bounds[1] - 1) as f32;
            if bw.min(bh) < 2. || score / count as f32 <= 0.6 {
                continue;
            }
            let d = 1.5 * bw * bh / (2. * (bw + bh));
            out.push([
                (bounds[0] as f32 - d).round().max(0.) as u32,
                (bounds[1] as f32 - d).round().max(0.) as u32,
                ((bounds[2] - 1) as f32 + d).round().min(w as f32) as u32,
                ((bounds[3] - 1) as f32 + d).round().min(h as f32) as u32,
            ]);
        }
    }
    out
}
impl ComicTextDetector {
    pub fn new(path: impl AsRef<Path>, backend: Backend) -> Result<Self> {
        let (device, backend) = crate::select_backend(backend)?;
        let graph = DetectorGraph::load(path.as_ref(), &device)?;
        Ok(Self {
            graph,
            device,
            backend,
        })
    }
    pub fn backend_info(&self) -> &BackendInfo {
        &self.backend
    }
    pub fn detect_groups(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<Vec<DetectedGroup>> {
        Ok(self.detect_page_with_mask(image, options)?.0)
    }

    /// The page's blocks, grouped as mokuro groups them (see `blocks`), and the detector's
    /// text mask, which mokuro cuts long lines with.
    pub fn detect_page_with_mask(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<(Vec<DetectedGroup>, TextMask)> {
        options.validate()?;
        let (input, w, h) = pixels(image, &self.device, false)?;
        let output = self.graph.forward(input)?;
        if output["blk"].dims() != [1, 64512, 7]
            || output["det"].dims() != [1, 2, SIZE, SIZE]
            || output["seg"].dims() != [1, 1, SIZE, SIZE]
        {
            return Err(Error::ModelContract(
                "unexpected detector output shapes".into(),
            ));
        }
        let predictions = output["blk"].squeeze(0)?.to_vec2::<f32>()?;
        let mut boxes: Vec<_> = predictions
            .iter()
            .filter_map(|p| {
                let score = p[4] * p[5].max(p[6]);
                if !score.is_finite() || score < options.confidence_threshold {
                    return None;
                }
                // The head's classes are the reference's LANG_LIST: English, then Japanese.
                let language = if p[6] > p[5] {
                    Language::Ja
                } else {
                    Language::Eng
                };
                mapped(
                    [
                        p[0] - p[2] / 2.,
                        p[1] - p[3] / 2.,
                        p[0] + p[2] / 2.,
                        p[1] + p[3] / 2.,
                    ],
                    image,
                    w,
                    h,
                )
                .map(|b| (b, score, language))
            })
            .collect();
        boxes.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut accepted: Vec<(BoundingBox, f32, Language)> = Vec::new();
        for candidate in boxes {
            if accepted
                .iter()
                .any(|g| iou(g.0, candidate.0) > f64::from(options.nms_threshold))
            {
                continue;
            }
            accepted.push(candidate);
            if accepted.len() == options.max_regions {
                break;
            }
        }
        // A block lying almost wholly inside a larger one is the same text found twice
        // (non-maximum suppression compares overlap with the union, so it keeps a small
        // box inside a big one). The reference keeps it, and with no line of its own it is
        // read again from its box, repeating the line; it is dropped here.
        let contained: Vec<bool> = (0..accepted.len())
            .map(|i| {
                let a = accepted[i].0;
                accepted.iter().enumerate().any(|(j, other)| {
                    let b = other.0;
                    let larger = area(b) > area(a) || (area(b) == area(a) && j < i);
                    j != i && larger && intersection_fraction(a, b) >= CONTAINED
                })
            })
            .collect();
        let mut keep = contained.iter().map(|c| !c);
        accepted.retain(|_| keep.next().unwrap_or(true));
        let candidates: Vec<_> = accepted
            .iter()
            .map(|&(bbox, confidence, language)| Candidate {
                bbox: bbox.map(f64::from),
                confidence,
                language,
            })
            .collect();
        let map = output["det"]
            .narrow(1, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let lines: Vec<_> = lines(&map, options.line_threshold, w, h)
            .into_iter()
            .filter_map(|line| {
                // Scaled to the page and truncated, as the reference casts them.
                let (sx, sy) = (
                    f64::from(image.width()) / f64::from(w),
                    f64::from(image.height()) / f64::from(h),
                );
                let r = [
                    (f64::from(line[0]) * sx).trunc(),
                    (f64::from(line[1]) * sy).trunc(),
                    (f64::from(line[2]) * sx).trunc(),
                    (f64::from(line[3]) * sy).trunc(),
                ];
                (r[0] < r[2] && r[1] < r[3]).then_some(r)
            })
            .collect();
        let mask = TextMask {
            values: output["seg"].flatten_all()?.to_vec1::<f32>()?,
            scale_x: f64::from(w) / f64::from(image.width()),
            scale_y: f64::from(h) / f64::from(image.height()),
        };
        let (im_w, im_h) = (f64::from(image.width()), f64::from(image.height()));
        let groups = blocks::group(&candidates, &lines, im_w, im_h, |r| mask.mean(r))
            .into_iter()
            .take(options.max_regions)
            .filter_map(|b| {
                let clamp = |r: blocks::Rect| -> Option<BoundingBox> {
                    let r = [
                        r[0].clamp(0., im_w) as u32,
                        r[1].clamp(0., im_h) as u32,
                        r[2].clamp(0., im_w) as u32,
                        r[3].clamp(0., im_h) as u32,
                    ];
                    (r[0] < r[2] && r[1] < r[3]).then_some(r)
                };
                let line_polygons: Vec<_> = b
                    .lines
                    .iter()
                    .filter_map(|&l| clamp(l))
                    .map(polygon)
                    .collect();
                Some(DetectedGroup {
                    bbox: clamp(b.bbox)?,
                    confidence: b.confidence,
                    orientation: if b.vertical {
                        Orientation::Vertical
                    } else {
                        Orientation::Horizontal
                    },
                    line_polygons,
                    font_size: b.font_size as f32,
                    language: b.language,
                })
            })
            .filter(|g| !g.line_polygons.is_empty())
            .collect();
        Ok((groups, mask))
    }
}

/// The detector's text mask (0 to 1), read in page pixels.
pub struct TextMask {
    values: Vec<f32>,
    scale_x: f64,
    scale_y: f64,
}
impl TextMask {
    /// At a page pixel; outside the page, nothing.
    pub fn at(&self, x: f64, y: f64) -> f32 {
        let (mx, my) = ((x * self.scale_x).floor(), (y * self.scale_y).floor());
        if mx < 0. || my < 0. || mx >= SIZE as f64 || my >= SIZE as f64 {
            return 0.;
        }
        self.values[my as usize * SIZE + mx as usize]
    }
    /// Mean over a rectangle of page pixels.
    pub(crate) fn mean(&self, r: blocks::Rect) -> f64 {
        let x0 = (r[0] * self.scale_x).floor().max(0.) as usize;
        let y0 = (r[1] * self.scale_y).floor().max(0.) as usize;
        let x1 = ((r[2] * self.scale_x).ceil() as usize)
            .min(SIZE)
            .max(x0 + 1);
        let y1 = ((r[3] * self.scale_y).ceil() as usize)
            .min(SIZE)
            .max(y0 + 1);
        let mut sum = 0.;
        for y in y0..y1.min(SIZE) {
            for x in x0..x1 {
                sum += f64::from(self.values[y * SIZE + x]);
            }
        }
        sum / ((x1 - x0) * (y1 - y0)) as f64
    }
}
impl TextDetector for ComicTextDetector {
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>> {
        Ok(self
            .detect_groups(image, &DetectionOptions::default())?
            .into_iter()
            .map(|g| Detection {
                bbox: g.bbox,
                confidence: Some(g.confidence),
            })
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires downloaded detector weights; use release mode"]
    fn detector_graph_matches_reference() {
        let image = RgbImage::from_fn(257, 389, |x, y| {
            image::Rgb([
                (x * 17 + y * 3) as u8,
                (x * 5 + y * 11) as u8,
                (x + y * 7) as u8,
            ])
        });
        let backend = if std::env::var("MANGAOCR_TEST_BACKEND").as_deref() == Ok("metal") {
            Backend::Metal
        } else {
            Backend::Cpu
        };
        let engine = ComicTextDetector::new("models/comic-text-detector.onnx", backend).unwrap();
        // The reference was produced with the stretching letterbox.
        let (input, _, _) = pixels(&image, &engine.device, true).unwrap();
        let dump = std::env::var("MANGAOCR_DETECTOR_DUMP").ok();
        let write = |name: &str, t: &Tensor| {
            if let Some(dir) = &dump {
                std::fs::create_dir_all(dir).unwrap();
                let values = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/{name}.f32"), bytes).unwrap();
            }
        };
        write("images", &input);
        let output = engine.graph.forward(input).unwrap();
        for name in ["blk", "seg", "det"] {
            write(name, &output[name]);
        }
        let reference = std::fs::read_to_string("tests/fixtures/detector-reference.json").unwrap();
        let reference: serde_json::Value = serde_json::from_str(&reference).unwrap();
        for name in ["blk", "seg", "det"] {
            let values = output[name]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let samples = reference[name].as_array().unwrap();
            assert!(
                samples.len() >= 200,
                "reference must exercise the real graph"
            );
            for sample in samples {
                let index = sample[0].as_u64().unwrap() as usize;
                let expected = sample[1].as_f64().unwrap() as f32;
                let tolerance = if name == "blk" {
                    0.01 + expected.abs() * 0.0001
                } else {
                    0.001
                };
                assert!(
                    (values[index] - expected).abs() < tolerance,
                    "{name}[{index}] {} != {expected}",
                    values[index]
                );
            }
        }
    }
    #[test]
    fn small_images_keep_their_scale_and_pages_are_shrunk_to_fit() {
        let device = Device::Cpu;
        let snip = RgbImage::from_pixel(80, 220, image::Rgb([255, 255, 255]));
        let (_, w, h) = pixels(&snip, &device, false).unwrap();
        assert_eq!((w, h), (80, 220));
        let (_, w, h) = pixels(&snip, &device, true).unwrap();
        assert_eq!((w, h), (372, 1024));
        let page = RgbImage::from_pixel(1654, 1170, image::Rgb([255, 255, 255]));
        let (_, w, h) = pixels(&page, &device, false).unwrap();
        assert_eq!((w, h), (1024, 724));
    }
    #[test]
    fn overlap_is_scale_independent() {
        assert_eq!(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.);
        assert_eq!(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.);
    }
    #[test]
    fn line_components_keep_columns_separate() {
        let mut map = vec![0.; SIZE * SIZE];
        for y in 10..60 {
            for x in [10, 11, 12, 30, 31, 32] {
                map[y * SIZE + x] = 0.9;
            }
        }
        let out = lines(&map, 0.3, 100, 100);
        assert_eq!(out.len(), 2);
        assert!(out[0][3] - out[0][1] > out[0][2] - out[0][0]);
    }
}
