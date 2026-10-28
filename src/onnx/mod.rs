//! Manga OCR on ONNX Runtime: the detector's own ONNX file, and the recognizer as graphs built
//! in code from the same weights (`graphs`), on the CPU or DirectML.
//!
//! The recognizer reads a page's line crops in groups: one encoder pass per group, then every
//! line's beams step through the decoder together, so a page costs a few dozen decoder runs
//! rather than one per line per token. ONNX Runtime must be initialised by the caller (with
//! `load-dynamic`, `ort::init_from(path).commit()`).
mod graphs;
mod proto;

use crate::detector::{self, DetectedGroup, DetectionOptions, SIZE, TextMask};
use crate::generation::{Beams, Hypothesis};
use crate::{
    Error, LineReader, ModelPaths, OcrOptions, PageDetector, PageResult, Recognition, Result,
    finish, preprocess,
};
use image::RgbImage;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynValue, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Where the ONNX graphs run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    Cpu,
    DirectMl,
}

/// The line crops one encoder pass and one batched decode take.
const GROUP: usize = 32;

fn session(model: &[u8], provider: Provider) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_op_spinning(false)?;
    match provider {
        // Subnormals cost x86 cores a microcode assist each (see the Candle detector's
        // FlushDenormals); ONNX Runtime flushes them on its own threads when asked.
        Provider::Cpu => {
            builder = builder.with_config_entry("session.set_denormal_as_zero", "1")?;
        }
        Provider::DirectMl => {
            #[cfg(feature = "directml")]
            {
                builder = builder
                    .with_parallel_execution(false)?
                    .with_memory_pattern(false)?
                    .with_execution_providers([
                        ort::execution_providers::DirectMLExecutionProvider::default()
                            .build()
                            .error_on_failure(),
                    ])?;
            }
            #[cfg(not(feature = "directml"))]
            return Err(Error::InvalidInput("DirectML is not compiled in".into()));
        }
    }
    Ok(builder.commit_from_memory(model)?)
}

/// The comic text detector's ONNX file, run by ONNX Runtime.
pub struct OnnxDetector {
    session: Session,
}
impl OnnxDetector {
    pub fn new(path: impl AsRef<Path>, provider: Provider) -> Result<Self> {
        Ok(Self {
            session: session(&std::fs::read(path)?, provider)?,
        })
    }
}
impl PageDetector for OnnxDetector {
    fn detect_page(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<Vec<DetectedGroup>> {
        Ok(self.detect_page_with_mask(image, options)?.0)
    }
    fn detect_page_with_mask(
        &mut self,
        image: &RgbImage,
        options: &DetectionOptions,
    ) -> Result<(Vec<DetectedGroup>, Option<TextMask>)> {
        options.validate()?;
        let (data, w, h) = detector::pixel_data(image, false)?;
        let input = Tensor::from_array(([1usize, 3, SIZE, SIZE], data))?;
        let out = self.session.run(ort::inputs!["images" => input])?;
        let (_, blk) = out["blk"].try_extract_tensor::<f32>()?;
        let (_, det) = out["det"].try_extract_tensor::<f32>()?;
        let (_, seg) = out["seg"].try_extract_tensor::<f32>()?;
        if det.len() < SIZE * SIZE {
            return Err(Error::ModelContract(
                "unexpected detector output shapes".into(),
            ));
        }
        let (groups, mask) =
            detector::page(image, w, h, blk, &det[..SIZE * SIZE], seg.to_vec(), options)?;
        Ok((groups, Some(mask)))
    }
}

/// The recognizer's encoder and decoder-step graphs.
pub struct OnnxReader {
    encoder: Session,
    step: Session,
    vocab: Vec<String>,
    beams: usize,
}
impl OnnxReader {
    /// Builds the graphs for `beams`-wide search from the checkpoint at `paths`.
    pub fn new(paths: &ModelPaths, provider: Provider, beams: usize) -> Result<Self> {
        crate::validate_model(&paths.config)?;
        let vocab = crate::load_vocab(&paths.vocab)?;
        let weights = graphs::Weights::load(&paths.weights)?;
        let encoder = session(&graphs::encoder(&weights)?, provider)?;
        let step = session(&graphs::step(&weights, beams)?, provider)?;
        Ok(Self {
            encoder,
            step,
            vocab,
            beams,
        })
    }
    /// Every line's search, its beams stepping through the decoder with all the others'.
    fn decode(
        &mut self,
        cross: &[DynValue],
        lines: usize,
        options: &OcrOptions,
    ) -> Result<Vec<Hypothesis>> {
        let w = self.beams;
        let rows = lines * w;
        let mut beams: Vec<_> = (0..lines).map(|_| Beams::new(w)).collect();
        let mut done = vec![false; lines];
        // One masked slot of zeros stands for the empty cache.
        let mut past = (0..2 * graphs::LAYERS)
            .map(|_| {
                Ok(Tensor::from_array(([rows, 12, 1, 64], vec![0f32; rows * 12 * 64]))?.into_dyn())
            })
            .collect::<Result<Vec<DynValue>>>()?;
        let mut parent: Vec<i64> = (0..rows as i64).collect();
        let mut mask = vec![-1e9f32];
        for step in 0..options.max_new_tokens {
            mask.push(0.);
            let mut ids = Vec::with_capacity(rows);
            for (j, b) in beams.iter().enumerate() {
                if done[j] {
                    ids.extend(std::iter::repeat_n(3i64, w));
                } else {
                    ids.extend(b.tokens().into_iter().map(i64::from));
                }
            }
            let mut out = self.step.run(ort::inputs![
                "ids" => Tensor::from_array(([rows, 1], ids))?,
                "pos" => Tensor::from_array(([1usize], vec![step as i64]))?,
                "parent" => Tensor::from_array(([rows], parent))?,
                "mask" => Tensor::from_array(([mask.len()], mask.clone()))?,
                "pk0" => &past[0], "pv0" => &past[1], "pk1" => &past[2], "pv1" => &past[3],
                "ck0" => &cross[0], "cv0" => &cross[1], "ck1" => &cross[2], "cv1" => &cross[3],
            ])?;
            let (_, logp) = out["logp"].try_extract_tensor::<f32>()?;
            let width = w * 6144;
            let mut next = Vec::with_capacity(rows);
            for j in 0..lines {
                if !done[j] {
                    match beams[j].advance(&logp[j * width..(j + 1) * width], options)? {
                        Some(parents) => {
                            next.extend(parents.into_iter().map(|p| (j * w) as i64 + i64::from(p)));
                            continue;
                        }
                        None => done[j] = true,
                    }
                }
                // A finished line keeps its rows, stepping on EOS, until every line is done.
                next.extend((0..w).map(|k| (j * w + k) as i64));
            }
            parent = next;
            past = ["nk0", "nv0", "nk1", "nv1"]
                .into_iter()
                .map(|k| {
                    out.remove(k)
                        .ok_or_else(|| Error::ModelContract(format!("missing step output {k}")))
                })
                .collect::<Result<_>>()?;
            if done.iter().all(|&d| d) {
                break;
            }
        }
        beams.into_iter().map(|b| b.best(options)).collect()
    }
}
impl LineReader for OnnxReader {
    fn read_lines(&mut self, crops: &[RgbImage], options: &OcrOptions) -> Result<Vec<Recognition>> {
        options.validate()?;
        if options.beam_width != self.beams {
            return Err(Error::InvalidInput(format!(
                "the decoder was built for {} beams",
                self.beams
            )));
        }
        for crop in crops {
            preprocess::validate(crop)?;
        }
        let mut out = vec![None; crops.len()];
        let mut pending = Vec::with_capacity(crops.len());
        for (i, crop) in crops.iter().enumerate() {
            if options.skip_uniform_images && crop.pixels().all(|p| p == crop.get_pixel(0, 0)) {
                out[i] = Some(Recognition {
                    text: String::new(),
                    token_ids: Vec::new(),
                    mean_log_probability: None,
                    truncated: false,
                });
            } else {
                pending.push(i);
            }
        }
        for group in pending.chunks(GROUP) {
            let mut pixels = Vec::with_capacity(group.len() * 3 * 224 * 224);
            for &i in group {
                pixels.extend(preprocess::pixel_data(&crops[i])?);
            }
            let input = Tensor::from_array(([group.len(), 3, 224, 224], pixels))?;
            let mut encoded = self.encoder.run(ort::inputs!["pixels" => input])?;
            let cross = ["ck0", "cv0", "ck1", "cv1"]
                .into_iter()
                .map(|k| {
                    encoded
                        .remove(k)
                        .ok_or_else(|| Error::ModelContract(format!("missing encoder output {k}")))
                })
                .collect::<Result<Vec<_>>>()?;
            drop(encoded);
            for (&i, h) in group.iter().zip(self.decode(&cross, group.len(), options)?) {
                out[i] = Some(finish(h, &self.vocab));
            }
        }
        Ok(out.into_iter().flatten().collect())
    }
}

/// The detector and recognizer together.
pub struct OnnxPage {
    pub detector: OnnxDetector,
    pub reader: OnnxReader,
    provider: Provider,
}
impl OnnxPage {
    /// `detector` is the comic text detector's ONNX file; `paths` the recognizer's checkpoint.
    pub fn new(
        paths: &ModelPaths,
        detector: impl AsRef<Path>,
        provider: Provider,
        beams: usize,
    ) -> Result<Self> {
        Ok(Self {
            detector: OnnxDetector::new(detector, provider)?,
            reader: OnnxReader::new(paths, provider, beams)?,
            provider,
        })
    }
    pub fn provider(&self) -> Provider {
        self.provider
    }
    /// As [`crate::MangaOcr::recognize_region`].
    pub fn recognize_region(
        &mut self,
        image: &RgbImage,
        options: &OcrOptions,
        detection: &DetectionOptions,
    ) -> Result<PageResult> {
        crate::read_region(
            &mut self.reader,
            image,
            &mut self.detector,
            options,
            detection,
        )
    }
    /// As [`crate::MangaOcr::recognize_page`].
    pub fn recognize_page(
        &mut self,
        image: &RgbImage,
        options: &OcrOptions,
        detection: &DetectionOptions,
    ) -> Result<PageResult> {
        crate::read_page(
            &mut self.reader,
            image,
            &mut self.detector,
            options,
            detection,
        )
    }
}
