//! Fixed upstream ViT-base / two-layer causal BERT architecture.
use candle_core::{D, DType, Module, Storage, Tensor};
use candle_nn::{Conv2d, Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
type R<T> = candle_core::Result<T>;
const H: usize = 768;
const HEADS: usize = 12;
const DIM: usize = H / HEADS;

#[derive(Clone)]
pub(crate) struct Kv {
    k: Tensor,
    v: Tensor,
}
impl Kv {
    pub fn select(&self, indices: &Tensor) -> R<Self> {
        Ok(Self {
            k: self.k.index_select(indices, 0)?,
            v: self.v.index_select(indices, 0)?,
        })
    }
}
/// Cross-attention keys, transposed, and values of one image: every beam reads the same.
pub(crate) struct Shared {
    kt: Tensor,
    v: Tensor,
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
}
impl Attention {
    fn load(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            q: linear(H, H, vb.pp("query"))?,
            k: linear(H, H, vb.pp("key"))?,
            v: linear(H, H, vb.pp("value"))?,
        })
    }
    fn split(t: Tensor) -> R<Tensor> {
        let (b, s, _) = t.dims3()?;
        t.reshape((b, s, HEADS, DIM))?.transpose(1, 2)?.contiguous()
    }
    fn kv(&self, x: &Tensor) -> R<Kv> {
        Ok(Kv {
            k: Self::split(self.k.forward(x)?)?,
            v: Self::split(self.v.forward(x)?)?,
        })
    }
    fn shared(&self, x: &Tensor) -> R<Shared> {
        let kv = self.kv(x)?;
        Ok(Shared {
            kt: kv.k.transpose(2, 3)?.contiguous()?,
            v: kv.v,
        })
    }
    /// One query token per beam against one image: the beams stand in for the query rows of a
    /// single product per head, instead of a one-row product per beam and head.
    fn attend_shared(&self, x: &Tensor, kv: &Shared) -> R<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((b * s, HEADS, DIM))?
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?;
        let scores = (q.matmul(&kv.kt)? / (DIM as f64).sqrt())?;
        candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&kv.v)?
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((b, s, H))
    }
    fn attend(&self, x: &Tensor, kv: &Kv) -> R<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = Self::split(self.q.forward(x)?)?;
        let scores = (q.matmul(&kv.k.transpose(2, 3)?.contiguous()?)? / (DIM as f64).sqrt())?;
        candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&kv.v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, H))
    }
}
struct FeedForward {
    input: Linear,
    output: Linear,
}
impl FeedForward {
    fn load(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            input: linear(H, 3072, vb.pp("intermediate.dense"))?,
            output: linear(3072, H, vb.pp("output.dense"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        self.output.forward(&gelu(&self.input.forward(x)?)?)
    }
}
struct VitLayer {
    attn: Attention,
    projection: Linear,
    before: LayerNorm,
    after: LayerNorm,
    ff: FeedForward,
}
impl VitLayer {
    fn load(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("attention.attention"))?,
            projection: linear(H, H, vb.pp("attention.output.dense"))?,
            before: layer_norm(H, 1e-12, vb.pp("layernorm_before"))?,
            after: layer_norm(H, 1e-12, vb.pp("layernorm_after"))?,
            ff: FeedForward::load(vb)?,
        })
    }
    fn forward(&self, x: &Tensor) -> R<Tensor> {
        let n = self.before.forward(x)?;
        let x = (x + self
            .projection
            .forward(&self.attn.attend(&n, &self.attn.kv(&n)?)?)?)?;
        &x + self.ff.forward(&self.after.forward(&x)?)?
    }
}
struct BertAttention {
    attn: Attention,
    projection: Linear,
    norm: LayerNorm,
}
impl BertAttention {
    fn load(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("self"))?,
            projection: linear(H, H, vb.pp("output.dense"))?,
            norm: layer_norm(H, 1e-12, vb.pp("output.LayerNorm"))?,
        })
    }
    fn forward(&self, x: &Tensor, kv: &Kv) -> R<Tensor> {
        self.norm
            .forward(&(x + self.projection.forward(&self.attn.attend(x, kv)?)?)?)
    }
    fn forward_shared(&self, x: &Tensor, kv: &Shared) -> R<Tensor> {
        self.norm
            .forward(&(x + self.projection.forward(&self.attn.attend_shared(x, kv)?)?)?)
    }
}
struct BertLayer {
    self_attn: BertAttention,
    cross: BertAttention,
    ff: FeedForward,
    norm: LayerNorm,
}
impl BertLayer {
    fn load(vb: VarBuilder) -> R<Self> {
        Ok(Self {
            self_attn: BertAttention::load(vb.pp("attention"))?,
            cross: BertAttention::load(vb.pp("crossattention"))?,
            norm: layer_norm(H, 1e-12, vb.pp("output.LayerNorm"))?,
            ff: FeedForward::load(vb)?,
        })
    }
    fn step(&self, x: &Tensor, cache: &mut Option<Kv>, cross: &Shared) -> R<Tensor> {
        let next = self.self_attn.attn.kv(x)?;
        let kv = match cache.as_ref() {
            None => next,
            Some(old) => Kv {
                k: Tensor::cat(&[&old.k, &next.k], 2)?,
                v: Tensor::cat(&[&old.v, &next.v], 2)?,
            },
        };
        let x = self.self_attn.forward(x, &kv)?;
        *cache = Some(kv);
        let x = self.cross.forward_shared(&x, cross)?;
        self.norm.forward(&(&x + self.ff.forward(&x)?)?)
    }
}
pub(crate) struct Model {
    patches: Conv2d,
    cls: Tensor,
    positions: Tensor,
    vit: Vec<VitLayer>,
    vit_norm: LayerNorm,
    words: Embedding,
    token_types: Embedding,
    decoder_positions: Embedding,
    embedding_norm: LayerNorm,
    bert: Vec<BertLayer>,
    lm: Linear,
    lm_norm: LayerNorm,
    lm_output: Linear,
}
pub(crate) struct DecodeState {
    caches: Vec<Option<Kv>>,
    cross: Vec<Shared>,
    pub position: usize,
}
impl DecodeState {
    pub fn reorder(&mut self, indices: &Tensor) -> R<()> {
        self.caches = self
            .caches
            .iter()
            .map(|x| x.as_ref().map(|v| v.select(indices)).transpose())
            .collect::<R<_>>()?;
        // Every beam belongs to the same image: the cross-attention KV is held once for all
        // of them and never needs reordering.
        Ok(())
    }
}
impl Model {
    pub fn load(vb: VarBuilder) -> R<Self> {
        let enc = vb.pp("encoder");
        let dec = vb.pp("decoder.bert");
        let head = vb.pp("decoder.cls.predictions");
        let words = embedding(6144, H, dec.pp("embeddings.word_embeddings"))?;
        // HF's tied decoder weight may be omitted in safetensors.
        let weight = if head.contains_tensor("decoder.weight") {
            head.get((6144, H), "decoder.weight")?
        } else {
            words.embeddings().clone()
        };
        let bias = if head.contains_tensor("decoder.bias") {
            head.get(6144, "decoder.bias")?
        } else {
            head.get(6144, "bias")?
        };
        Ok(Self {
            patches: candle_nn::conv2d(
                3,
                H,
                16,
                candle_nn::Conv2dConfig {
                    stride: 16,
                    ..Default::default()
                },
                enc.pp("embeddings.patch_embeddings.projection"),
            )?,
            cls: enc.get((1, 1, H), "embeddings.cls_token")?,
            positions: enc.get((1, 197, H), "embeddings.position_embeddings")?,
            vit: (0..12)
                .map(|i| VitLayer::load(enc.pp(format!("encoder.layer.{i}"))))
                .collect::<R<_>>()?,
            vit_norm: layer_norm(H, 1e-12, enc.pp("layernorm"))?,
            words,
            token_types: embedding(2, H, dec.pp("embeddings.token_type_embeddings"))?,
            decoder_positions: embedding(512, H, dec.pp("embeddings.position_embeddings"))?,
            embedding_norm: layer_norm(H, 1e-12, dec.pp("embeddings.LayerNorm"))?,
            bert: (0..2)
                .map(|i| BertLayer::load(dec.pp(format!("encoder.layer.{i}"))))
                .collect::<R<_>>()?,
            lm: linear(H, H, head.pp("transform.dense"))?,
            lm_norm: layer_norm(H, 1e-12, head.pp("transform.LayerNorm"))?,
            lm_output: Linear::new(weight, Some(bias)),
        })
    }
    /// The ViT's last hidden states, `[n, 197, 768]`, for a batch of images `[n, 3, 224, 224]`.
    pub fn encode(&self, pixels: &Tensor) -> R<Tensor> {
        let patches = self
            .patches
            .forward(pixels)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let cls = self
            .cls
            .broadcast_as((patches.dim(0)?, 1, H))?
            .contiguous()?;
        let mut x = Tensor::cat(&[&cls, &patches], 1)?.broadcast_add(&self.positions)?;
        for layer in &self.vit {
            x = layer.forward(&x)?;
        }
        self.vit_norm.forward(&x)
    }
    /// A fresh decode of one image's hidden states, `[1, 197, 768]`.
    pub fn start(&self, encoded: &Tensor) -> R<DecodeState> {
        let cross = self
            .bert
            .iter()
            .map(|l| l.cross.attn.shared(encoded))
            .collect::<R<_>>()?;
        Ok(DecodeState {
            caches: vec![None; self.bert.len()],
            cross,
            position: 0,
        })
    }
    pub fn step(&self, ids: &Tensor, state: &mut DecodeState) -> R<Tensor> {
        let shape = ids.shape();
        let pos = Tensor::full(state.position as u32, shape, ids.device())?;
        let types = Tensor::zeros(shape, candle_core::DType::U32, ids.device())?;
        let mut x = self.embedding_norm.forward(
            &((&self.words.forward(ids)? + self.decoder_positions.forward(&pos)?)?
                + self.token_types.forward(&types)?)?,
        )?;
        for (i, l) in self.bert.iter().enumerate() {
            x = l.step(&x, &mut state.caches[i], &state.cross[i])?;
        }
        state.position += 1;
        candle_nn::ops::log_softmax(
            &self
                .lm_output
                .forward(&self.lm_norm.forward(&gelu(&self.lm.forward(&x)?)?)?)?
                .squeeze(1)?
                .to_dtype(candle_core::DType::F32)?,
            D::Minus1,
        )
    }
}

/// `gelu_erf`, spread over threads on the CPU. Candle maps unary ops on one thread, and the erf
/// of every feed-forward activation became the serial part of a batched encode. The same
/// formula and erf as Candle's, so the values are identical.
fn gelu(x: &Tensor) -> R<Tensor> {
    use rayon::prelude::*;
    if !x.device().is_cpu() || x.dtype() != DType::F32 {
        return x.gelu_erf();
    }
    let x = x.contiguous()?;
    let (storage, layout) = x.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        unreachable!("CPU tensors")
    };
    let src = &storage.as_slice::<f32>()?[layout.start_offset()..][..x.elem_count()];
    let mut out = vec![0f32; src.len()];
    out.par_chunks_mut(1 << 14)
        .zip(src.par_chunks(1 << 14))
        .for_each(|(dst, src)| {
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.)
                    * 0.5
                    * v;
            }
        });
    Tensor::from_vec(out, x.shape(), x.device())
}
