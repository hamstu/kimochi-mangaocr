//! The recognizer as two ONNX graphs, built from the same weights as the Candle model
//! (`model.rs`), op for op.
//!
//! - `encoder`: pixels `[n, 3, 224, 224]` → each image's cross-attention keys, transposed,
//!   `ck{i}` `[n, 12, 64, 197]`, and values `cv{i}` `[n, 12, 197, 64]`, for both decoder
//!   layers. The ViT's hidden states are needed for nothing else.
//! - `step`: one decoder step for `l` images × `beams` beams at once (rows image-major).
//!   Inputs: `ids` `[r, 1]`, `pos` `[1]`, `parent` `[r]` (the row of the cache each row
//!   continues), `mask` `[t + 1]` (added to self-attention scores), the self-attention cache
//!   `pk{i}`/`pv{i}` `[p, 12, t, 64]` and the encoder's `ck{i}`/`cv{i}`. Outputs: `logp`
//!   `[r, 6144]` and the grown cache `nk{i}`/`nv{i}` `[r, 12, t + 1, 64]`.
//!
//! The cache starts as one masked slot of zeros: a zero-length tensor is not something every
//! execution provider takes.
use super::proto::{Builder, Dim, FLOAT, INT64, float, int, ints};
use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

const H: usize = 768;
const HEADS: i64 = 12;
const DIM: i64 = 64;
pub const LAYERS: usize = 2;

/// The checkpoint's tensors by name, as f32 on the CPU.
pub struct Weights(HashMap<String, Tensor>);
impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let tensors: Vec<(String, Tensor)> = if path.extension().is_some_and(|x| x == "safetensors")
        {
            candle_core::safetensors::load(path, &Device::Cpu)?
                .into_iter()
                .collect()
        } else {
            candle_core::pickle::read_all(path)?
        };
        Ok(Self(
            tensors
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
                .collect::<Result<_>>()?,
        ))
    }
    fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.0
            .get(name)
            .ok_or_else(|| Error::ModelContract(format!("missing weight {name}")))
    }
    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    fn data(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = self.tensor(name)?;
        Ok((t.dims().to_vec(), t.flatten_all()?.to_vec1()?))
    }
    /// A Linear's weight transposed to `[in, out]` (several side by side), and its biases.
    fn linear(&self, prefixes: &[&str]) -> Result<(Vec<usize>, Vec<f32>, Vec<f32>)> {
        let ts = prefixes
            .iter()
            .map(|p| Ok(self.tensor(&format!("{p}.weight"))?.t()?.contiguous()?))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<_> = ts.iter().collect();
        let w = Tensor::cat(&refs, 1)?;
        let bias = prefixes
            .iter()
            .map(|p| self.tensor(&format!("{p}.bias")))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            w.dims().to_vec(),
            w.flatten_all()?.to_vec1()?,
            Tensor::cat(&bias, 0)?.to_vec1()?,
        ))
    }
}

/// Shared node patterns.
struct Net<'a> {
    g: Builder,
    w: &'a Weights,
}
impl Net<'_> {
    /// `x · Wᵀ + b` for one or several Linears fused side by side.
    fn linear(&mut self, x: &str, prefixes: &[&str]) -> Result<String> {
        let (dims, w, b) = self.w.linear(prefixes)?;
        let name = format!("{}.fused", prefixes[0]);
        let wn = self.g.weight(&format!("{name}.weight"), &dims, &w);
        let bn = self.g.weight(&format!("{name}.bias"), &[b.len()], &b);
        let y = self.g.op("MatMul", &[x, &wn], vec![]);
        Ok(self.g.op("Add", &[&y, &bn], vec![]))
    }
    fn norm(&mut self, x: &str, prefix: &str) -> Result<String> {
        let (_, w) = self.w.data(&format!("{prefix}.weight"))?;
        let (_, b) = self.w.data(&format!("{prefix}.bias"))?;
        let wn = self.g.weight(&format!("{prefix}.weight"), &[H], &w);
        let bn = self.g.weight(&format!("{prefix}.bias"), &[H], &b);
        Ok(self.g.op(
            "LayerNormalization",
            &[x, &wn, &bn],
            vec![int("axis", -1), float("epsilon", 1e-12)],
        ))
    }
    /// GELU with erf, as `gelu_erf`: `x · ½ · (1 + erf(x / √2))`.
    fn gelu(&mut self, x: &str) -> String {
        let r = self.g.scalar(std::f32::consts::FRAC_1_SQRT_2);
        let one = self.g.scalar(1.);
        let half = self.g.scalar(0.5);
        let s = self.g.op("Mul", &[x, &r], vec![]);
        let e = self.g.op("Erf", &[&s], vec![]);
        let e = self.g.op("Add", &[&e, &one], vec![]);
        let y = self.g.op("Mul", &[x, &e], vec![]);
        self.g.op("Mul", &[&y, &half], vec![])
    }
    fn feed_forward(&mut self, x: &str, prefix: &str) -> Result<String> {
        let h = self.linear(x, &[&format!("{prefix}.intermediate.dense")])?;
        let h = self.gelu(&h);
        self.linear(&h, &[&format!("{prefix}.output.dense")])
    }
    fn reshape(&mut self, x: &str, shape: &[i64]) -> String {
        let s = self.g.list(shape);
        self.g.op("Reshape", &[x, &s], vec![])
    }
    fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        self.g.op("Transpose", &[x], vec![ints("perm", perm)])
    }
    fn pick(&mut self, x: &str, index: i64) -> String {
        let i = self.g.ints(&[], &[index]);
        self.g.op("Gather", &[x, &i], vec![int("axis", 0)])
    }
    /// Fused query/key/value projections as heads: `q`, `k`, `v`, each `[b, 12, s, 64]`.
    fn qkv(&mut self, x: &str, prefix: &str) -> Result<[String; 3]> {
        let qkv = self.linear(
            x,
            &[
                &format!("{prefix}.query"),
                &format!("{prefix}.key"),
                &format!("{prefix}.value"),
            ],
        )?;
        let qkv = self.reshape(&qkv, &[0, 0, 3, HEADS, DIM]);
        let qkv = self.transpose(&qkv, &[2, 0, 3, 1, 4]);
        Ok([self.pick(&qkv, 0), self.pick(&qkv, 1), self.pick(&qkv, 2)])
    }
    /// softmax(q · kt / 8 + mask) · v, with `kt` already `[.., 64, s]`.
    fn attend(&mut self, q: &str, kt: &str, v: &str, mask: Option<&str>) -> String {
        let scale = self.g.scalar(1. / (DIM as f32).sqrt());
        let s = self.g.op("MatMul", &[q, kt], vec![]);
        let mut s = self.g.op("Mul", &[&s, &scale], vec![]);
        if let Some(mask) = mask {
            s = self.g.op("Add", &[&s, mask], vec![]);
        }
        let p = self.g.op("Softmax", &[&s], vec![int("axis", -1)]);
        self.g.op("MatMul", &[&p, v], vec![])
    }
    /// Heads `[b, 12, s, 64]` back to `[b, s, 768]`.
    fn merge(&mut self, x: &str) -> String {
        let x = self.transpose(x, &[0, 2, 1, 3]);
        self.reshape(&x, &[0, 0, H as i64])
    }
}

pub fn encoder(w: &Weights) -> Result<Vec<u8>> {
    let mut n = Net {
        g: Builder::default(),
        w,
    };
    let pixels = n.g.input(
        "pixels",
        FLOAT,
        &[
            Dim::Free("n"),
            Dim::Fixed(3),
            Dim::Fixed(224),
            Dim::Fixed(224),
        ],
    );
    let p = "encoder.embeddings";
    let (dims, data) = w.data(&format!("{p}.patch_embeddings.projection.weight"))?;
    let pw = n.g.weight("patch.weight", &dims, &data);
    let (_, data) = w.data(&format!("{p}.patch_embeddings.projection.bias"))?;
    let pb = n.g.weight("patch.bias", &[H], &data);
    let x = n.g.op(
        "Conv",
        &[&pixels, &pw, &pb],
        vec![ints("kernel_shape", &[16, 16]), ints("strides", &[16, 16])],
    );
    let x = n.reshape(&x, &[0, H as i64, 196]);
    let x = n.transpose(&x, &[0, 2, 1]);
    // The CLS token, one per image.
    let shape = n.g.op("Shape", &[&pixels], vec![]);
    let start = n.g.list(&[0]);
    let end = n.g.list(&[1]);
    let batch = n.g.op("Slice", &[&shape, &start, &end], vec![]);
    let rest = n.g.list(&[1, H as i64]);
    let cls_shape = n.g.op("Concat", &[&batch, &rest], vec![int("axis", 0)]);
    let (_, data) = w.data(&format!("{p}.cls_token"))?;
    let cls = n.g.weight("cls", &[1, 1, H], &data);
    let cls = n.g.op("Expand", &[&cls, &cls_shape], vec![]);
    let x = n.g.op("Concat", &[&cls, &x], vec![int("axis", 1)]);
    let (_, data) = w.data(&format!("{p}.position_embeddings"))?;
    let positions = n.g.weight("positions", &[1, 197, H], &data);
    let mut x = n.g.op("Add", &[&x, &positions], vec![]);
    for i in 0..12 {
        let l = format!("encoder.encoder.layer.{i}");
        let h = n.norm(&x, &format!("{l}.layernorm_before"))?;
        let [q, k, v] = n.qkv(&h, &format!("{l}.attention.attention"))?;
        let kt = n.transpose(&k, &[0, 1, 3, 2]);
        let a = n.attend(&q, &kt, &v, None);
        let a = n.merge(&a);
        let a = n.linear(&a, &[&format!("{l}.attention.output.dense")])?;
        x = n.g.op("Add", &[&x, &a], vec![]);
        let h = n.norm(&x, &format!("{l}.layernorm_after"))?;
        let f = n.feed_forward(&h, &l)?;
        x = n.g.op("Add", &[&x, &f], vec![]);
    }
    let hidden = n.norm(&x, "encoder.layernorm")?;
    for i in 0..LAYERS {
        let c = format!("decoder.bert.encoder.layer.{i}.crossattention.self");
        let k = n.linear(&hidden, &[&format!("{c}.key")])?;
        let k = n.reshape(&k, &[0, 197, HEADS, DIM]);
        let kt = n.transpose(&k, &[0, 2, 3, 1]);
        let v = n.linear(&hidden, &[&format!("{c}.value")])?;
        let v = n.reshape(&v, &[0, 197, HEADS, DIM]);
        let v = n.transpose(&v, &[0, 2, 1, 3]);
        let free = Dim::Free("n");
        n.g.output(
            &kt,
            &format!("ck{i}"),
            FLOAT,
            &[free, Dim::Fixed(HEADS), Dim::Fixed(DIM), Dim::Fixed(197)],
        );
        n.g.output(
            &v,
            &format!("cv{i}"),
            FLOAT,
            &[
                Dim::Free("n"),
                Dim::Fixed(HEADS),
                Dim::Fixed(197),
                Dim::Fixed(DIM),
            ],
        );
    }
    Ok(n.g.finish("mangaocr-encoder"))
}

pub fn step(w: &Weights, beams: usize) -> Result<Vec<u8>> {
    let mut n = Net {
        g: Builder::default(),
        w,
    };
    let cache = || {
        [
            Dim::Free("p"),
            Dim::Fixed(HEADS),
            Dim::Free("t"),
            Dim::Fixed(DIM),
        ]
    };
    let ids = n.g.input("ids", INT64, &[Dim::Free("r"), Dim::Fixed(1)]);
    let pos = n.g.input("pos", INT64, &[Dim::Fixed(1)]);
    let parent = n.g.input("parent", INT64, &[Dim::Free("r")]);
    let mask = n.g.input("mask", FLOAT, &[Dim::Free("t1")]);
    let mut past = Vec::new();
    let mut cross = Vec::new();
    for i in 0..LAYERS {
        past.push((
            n.g.input(&format!("pk{i}"), FLOAT, &cache()),
            n.g.input(&format!("pv{i}"), FLOAT, &cache()),
        ));
        cross.push((
            n.g.input(
                &format!("ck{i}"),
                FLOAT,
                &[
                    Dim::Free("l"),
                    Dim::Fixed(HEADS),
                    Dim::Fixed(DIM),
                    Dim::Fixed(197),
                ],
            ),
            n.g.input(
                &format!("cv{i}"),
                FLOAT,
                &[
                    Dim::Free("l"),
                    Dim::Fixed(HEADS),
                    Dim::Fixed(197),
                    Dim::Fixed(DIM),
                ],
            ),
        ));
    }
    let e = "decoder.bert.embeddings";
    let (dims, data) = w.data(&format!("{e}.word_embeddings.weight"))?;
    let words = n.g.weight("words", &dims, &data);
    let (dims, data) = w.data(&format!("{e}.position_embeddings.weight"))?;
    let positions = n.g.weight("decoder.positions", &dims, &data);
    let (_, data) = w.data(&format!("{e}.token_type_embeddings.weight"))?;
    let token_type = n.g.weight("token_type", &[H], &data[..H]);
    let x = n.g.op("Gather", &[&words, &ids], vec![int("axis", 0)]);
    let p = n.g.op("Gather", &[&positions, &pos], vec![int("axis", 0)]);
    let x = n.g.op("Add", &[&x, &p], vec![]);
    let x = n.g.op("Add", &[&x, &token_type], vec![]);
    let mut x = n.norm(&x, &format!("{e}.LayerNorm"))?;
    for i in 0..LAYERS {
        let l = format!("decoder.bert.encoder.layer.{i}");
        // Self-attention over the cache, reordered to this step's beams, and this token.
        let [q, k, v] = n.qkv(&x, &format!("{l}.attention.self"))?;
        let pk =
            n.g.op("Gather", &[&past[i].0, &parent], vec![int("axis", 0)]);
        let pv =
            n.g.op("Gather", &[&past[i].1, &parent], vec![int("axis", 0)]);
        let k = n.g.op("Concat", &[&pk, &k], vec![int("axis", 2)]);
        let v = n.g.op("Concat", &[&pv, &v], vec![int("axis", 2)]);
        let grown = || {
            [
                Dim::Free("r"),
                Dim::Fixed(HEADS),
                Dim::Free("t1"),
                Dim::Fixed(DIM),
            ]
        };
        n.g.output(&k, &format!("nk{i}"), FLOAT, &grown());
        n.g.output(&v, &format!("nv{i}"), FLOAT, &grown());
        let kt = n.transpose(&k, &[0, 1, 3, 2]);
        let a = n.attend(&q, &kt, &v, Some(&mask));
        let a = n.merge(&a);
        let a = n.linear(&a, &[&format!("{l}.attention.output.dense")])?;
        let a = n.g.op("Add", &[&x, &a], vec![]);
        x = n.norm(&a, &format!("{l}.attention.output.LayerNorm"))?;
        // Cross-attention: each image's beams are the query rows of one product per head.
        let c = format!("{l}.crossattention");
        let q = n.linear(&x, &[&format!("{c}.self.query")])?;
        let q = n.reshape(&q, &[-1, beams as i64, HEADS, DIM]);
        let q = n.transpose(&q, &[0, 2, 1, 3]);
        let a = n.attend(&q, &cross[i].0, &cross[i].1, None);
        let a = n.transpose(&a, &[0, 2, 1, 3]);
        let a = n.reshape(&a, &[-1, 1, H as i64]);
        let a = n.linear(&a, &[&format!("{c}.output.dense")])?;
        let a = n.g.op("Add", &[&x, &a], vec![]);
        x = n.norm(&a, &format!("{c}.output.LayerNorm"))?;
        let f = n.feed_forward(&x, &l)?;
        let f = n.g.op("Add", &[&x, &f], vec![]);
        x = n.norm(&f, &format!("{l}.output.LayerNorm"))?;
    }
    let head = "decoder.cls.predictions";
    let h = n.linear(&x, &[&format!("{head}.transform.dense")])?;
    let h = n.gelu(&h);
    let h = n.norm(&h, &format!("{head}.transform.LayerNorm"))?;
    // HF's tied decoder weight may be omitted; it is then the word embeddings.
    let weight = if w.has(&format!("{head}.decoder.weight")) {
        w.tensor(&format!("{head}.decoder.weight"))?
    } else {
        w.tensor(&format!("{e}.word_embeddings.weight"))?
    };
    let weight = weight.t()?.contiguous()?;
    let wn = n.g.weight(
        "lm.weight",
        weight.dims(),
        &weight.flatten_all()?.to_vec1::<f32>()?,
    );
    let bias = if w.has(&format!("{head}.decoder.bias")) {
        format!("{head}.decoder.bias")
    } else {
        format!("{head}.bias")
    };
    let (_, data) = w.data(&bias)?;
    let bn = n.g.weight("lm.bias", &[data.len()], &data);
    let logits = n.g.op("MatMul", &[&h, &wn], vec![]);
    let logits = n.g.op("Add", &[&logits, &bn], vec![]);
    let logits = n.reshape(&logits, &[-1, 6144]);
    let logp = n.g.op("LogSoftmax", &[&logits], vec![int("axis", -1)]);
    n.g.output(&logp, "logp", FLOAT, &[Dim::Free("r"), Dim::Fixed(6144)]);
    Ok(n.g.finish("mangaocr-step"))
}
