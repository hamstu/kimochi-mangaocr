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
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynValue, Tensor};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Where the ONNX graphs run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    Cpu,
    DirectMl,
}

/// The line crops one encoder pass takes.
const ENCODE_BATCH: usize = 32;
/// The lines one decode steps together. A page's lines go through in one: the decode takes as
/// many steps as its longest line, so splitting a page multiplies the steps.
const DECODE_LINES: usize = 128;

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

fn host_memory() -> Result<MemoryInfo> {
    Ok(MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
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
    provider: Provider,
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
            provider,
        })
    }
    /// Where the decoder's cache stays between steps: on the GPU for DirectML.
    fn cache_memory(&self) -> Result<MemoryInfo> {
        Ok(match self.provider {
            Provider::DirectMl => MemoryInfo::new(
                AllocationDevice::DIRECTML,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
            Provider::Cpu => host_memory()?,
        })
    }
    /// Every line's search, its beams stepping through the decoder with all the others'.
    ///
    /// `cross` holds each line's encoder keys and values (`CROSS` order, lines in a row). Lines
    /// leave the batch as they finish, a few at a time: their cache rows drop out through the
    /// step's own reordering, and the remaining lines' keys and values are bound again.
    fn decode(
        &mut self,
        cross: &[Vec<f32>; 4],
        lines: usize,
        options: &OcrOptions,
    ) -> Result<Vec<Hypothesis>> {
        let w = self.beams;
        let mut beams: Vec<_> = (0..lines).map(|_| Beams::new(w)).collect();
        let mut done = vec![false; lines];
        // The lines in the batch, in row order: line `active[i]` owns rows `i·w..(i+1)·w`.
        let mut active: Vec<usize> = (0..lines).collect();
        // The encoder's keys and values are copied to the device when bound and read by every
        // step; the cache stays where the step leaves it. Only token ids, parents and the
        // log-probabilities cross between host and device each step.
        let mut binding = self.step.create_binding()?;
        bind_cross(&mut binding, cross, &active)?;
        let cache = self.cache_memory()?;
        let host = host_memory()?;
        let rows = lines * w;
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
            let rows = active.len() * w;
            let mut ids = Vec::with_capacity(rows);
            for &j in &active {
                if done[j] {
                    ids.extend(std::iter::repeat_n(3i64, w));
                } else {
                    ids.extend(beams[j].tokens().into_iter().map(i64::from));
                }
            }
            binding.bind_input("ids", &Tensor::from_array(([rows, 1], ids))?)?;
            binding.bind_input("pos", &Tensor::from_array(([1usize], vec![step as i64]))?)?;
            binding.bind_input("parent", &Tensor::from_array(([rows], parent))?)?;
            binding.bind_input("mask", &Tensor::from_array(([mask.len()], mask.clone()))?)?;
            for (name, value) in ["pk0", "pv0", "pk1", "pv1"].into_iter().zip(&past) {
                binding.bind_input(name, value)?;
            }
            // Rebound each step: a bound output keeps the last run's tensor, and the cache grows.
            for name in ["nk0", "nv0", "nk1", "nv1"] {
                binding.bind_output_to_device(name, &cache)?;
            }
            binding.bind_output_to_device("logp", &host)?;
            let mut out = self.step.run_binding(&binding)?;
            let (_, logp) = out["logp"].try_extract_tensor::<f32>()?;
            let width = w * 6144;
            // Each line's search is independent of the others'. `row_of`: the stepping lines'
            // places in the batch.
            let row_of: std::collections::HashMap<usize, usize> = active
                .iter()
                .enumerate()
                .filter(|&(_, &j)| !done[j])
                .map(|(i, &j)| (j, i))
                .collect();
            let advanced: Vec<(usize, Option<Vec<u32>>)> = beams
                .par_iter_mut()
                .enumerate()
                .filter_map(|(j, b)| row_of.get(&j).map(|&i| (j, i, b)))
                .map(|(j, i, b)| Ok((j, b.advance(&logp[i * width..(i + 1) * width], options)?)))
                .collect::<Result<_>>()?;
            let mut parents: std::collections::HashMap<usize, Vec<u32>> = Default::default();
            for (j, p) in advanced {
                match p {
                    Some(p) => {
                        parents.insert(j, p);
                    }
                    None => done[j] = true,
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
            past = ["nk0", "nv0", "nk1", "nv1"]
                .into_iter()
                .map(|k| {
                    out.remove(k)
                        .ok_or_else(|| Error::ModelContract(format!("missing step output {k}")))
                })
                .collect::<Result<_>>()?;
            drop(out);
            // Drop finished lines once a quarter of the batch has finished: each drop binds the
            // remaining lines' keys and values again, so it is done a few times, not per line.
            let live = active.iter().filter(|&&j| !done[j]).count();
            let keep: Vec<usize> = if live * 4 <= active.len() * 3 {
                active.iter().copied().filter(|&j| !done[j]).collect()
            } else {
                active.clone()
            };
            let mut next = Vec::with_capacity(keep.len() * w);
            for &j in &keep {
                // Rows index the cache this step wrote, in the old row order.
                let row = active
                    .iter()
                    .position(|&a| a == j)
                    .expect("kept line is active");
                let base = (row * w) as i64;
                match parents.get(&j) {
                    Some(p) => next.extend(p.iter().map(|&p| base + i64::from(p))),
                    // A finished line not yet dropped keeps its rows, stepping on EOS.
                    None => next.extend((0..w as i64).map(|k| base + k)),
                }
            }
            if keep.len() != active.len() {
                bind_cross(&mut binding, cross, &keep)?;
                active = keep;
            }
            parent = next;
        }
        beams.into_iter().map(|b| b.best(options)).collect()
    }
}

/// The encoder outputs the step reads: name and per-line shape, in `cross` order.
const CROSS: [(&str, [usize; 3]); 4] = [
    ("ck0", [12, 64, 197]),
    ("cv0", [12, 197, 64]),
    ("ck1", [12, 64, 197]),
    ("cv1", [12, 197, 64]),
];

/// Binds the keys and values of `lines` (indices into `cross`'s lines) as the step's inputs.
fn bind_cross(
    binding: &mut ort::io_binding::IoBinding,
    cross: &[Vec<f32>; 4],
    lines: &[usize],
) -> Result<()> {
    for (all, (name, [a, b, c])) in cross.iter().zip(CROSS) {
        let per = a * b * c;
        let mut data = Vec::with_capacity(lines.len() * per);
        for &j in lines {
            data.extend_from_slice(&all[j * per..(j + 1) * per]);
        }
        binding.bind_input(name, &Tensor::from_array(([lines.len(), a, b, c], data))?)?;
    }
    Ok(())
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
        for group in pending.chunks(DECODE_LINES) {
            // Encoded in batches, their keys and values gathered for one decode.
            let mut cross: [Vec<f32>; 4] = Default::default();
            for batch in group.chunks(ENCODE_BATCH) {
                let mut pixels = Vec::with_capacity(batch.len() * 3 * 224 * 224);
                for &i in batch {
                    pixels.extend(preprocess::pixel_data(&crops[i])?);
                }
                let input = Tensor::from_array(([batch.len(), 3, 224, 224], pixels))?;
                let encoded = self.encoder.run(ort::inputs!["pixels" => input])?;
                for (all, (name, _)) in cross.iter_mut().zip(CROSS) {
                    all.extend_from_slice(encoded[name].try_extract_tensor::<f32>()?.1);
                }
            }
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
