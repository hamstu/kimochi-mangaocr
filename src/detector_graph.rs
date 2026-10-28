//! Native execution of the pinned detector's ONNX tensor graph. ONNX is a file
//! format here, not an inference runtime. Weights are uploaded once; dead tensors
//! are released after their last consumer to bound page-inference memory.
use crate::{Error, Result};
use candle_core::{Device, Storage, Tensor};
use prost::Message;
use rayon::prelude::*;
use std::collections::HashMap;
#[derive(Clone, PartialEq, Message)]
struct Model {
    #[prost(message, optional, tag = "7")]
    graph: Option<Graph>,
}
#[derive(Clone, PartialEq, Message)]
struct Graph {
    #[prost(message, repeated, tag = "1")]
    nodes: Vec<Node>,
    #[prost(message, repeated, tag = "5")]
    initializers: Vec<Weight>,
}
#[derive(Clone, PartialEq, Message)]
struct Node {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op: String,
    #[prost(message, repeated, tag = "5")]
    attrs: Vec<Attr>,
}
#[derive(Clone, PartialEq, Message)]
struct Attr {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(float, tag = "2")]
    float: f32,
    #[prost(int64, tag = "3")]
    int: i64,
    #[prost(bytes = "vec", tag = "4")]
    bytes: Vec<u8>,
    #[prost(int64, repeated, tag = "8")]
    ints: Vec<i64>,
}
#[derive(Clone, PartialEq, Message)]
struct Weight {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    dtype: i32,
    #[prost(float, repeated, tag = "4")]
    floats: Vec<f32>,
    #[prost(int64, repeated, tag = "7")]
    ints: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw: Vec<u8>,
}
impl Node {
    fn int(&self, key: &str, default: i64) -> i64 {
        self.attrs
            .iter()
            .find(|a| a.name == key)
            .map_or(default, |a| a.int)
    }
    fn float(&self, key: &str, default: f32) -> f32 {
        self.attrs
            .iter()
            .find(|a| a.name == key)
            .map_or(default, |a| a.float)
    }
    fn ints(&self, key: &str, default: &[i64]) -> Vec<i64> {
        self.attrs
            .iter()
            .find(|a| a.name == key)
            .map_or_else(|| default.to_vec(), |a| a.ints.clone())
    }
    fn pair(&self, key: &str, default: i64) -> Result<usize> {
        let v = self.ints(key, &[default, default]);
        if v.len() != 2 || v[0] != v[1] || v[0] < 0 {
            return Err(Error::ModelContract(format!(
                "nonuniform {key} in {}",
                self.name
            )));
        }
        Ok(v[0] as usize)
    }
    fn padding(&self) -> Result<usize> {
        let p = self.ints("pads", &[0; 4]);
        if p.len() != 4 || p.iter().any(|&x| x != p[0] || x < 0) {
            return Err(Error::ModelContract(format!(
                "asymmetric padding in {}",
                self.name
            )));
        }
        Ok(p[0] as usize)
    }
}
pub(crate) struct DetectorGraph {
    nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    uses: HashMap<String, usize>,
}
impl DetectorGraph {
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self> {
        let model = Model::decode(std::fs::read(path)?.as_slice())
            .map_err(|e| Error::ModelContract(format!("ONNX: {e}")))?;
        let graph = model
            .graph
            .ok_or_else(|| Error::ModelContract("ONNX graph missing".into()))?;
        let mut weights = HashMap::new();
        for w in graph.initializers {
            if w.dims.iter().any(|&x| x < 0) {
                return Err(Error::ModelContract(
                    "negative initializer dimensions".into(),
                ));
            }
            let dims: Vec<_> = w.dims.iter().map(|&x| x as usize).collect();
            let t = match w.dtype {
                1 => {
                    let data = if w.raw.is_empty() {
                        w.floats
                    } else {
                        w.raw
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect()
                    };
                    Tensor::from_vec(
                        data,
                        dims.as_slice(),
                        if dims.contains(&0) {
                            &Device::Cpu
                        } else {
                            device
                        },
                    )?
                }
                7 => {
                    let data = if w.raw.is_empty() {
                        w.ints
                    } else {
                        w.raw
                            .chunks_exact(8)
                            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                            .collect()
                    };
                    // Shape parameters stay on CPU, avoiding unsupported GPU integer ops.
                    Tensor::from_vec(data, dims.as_slice(), &Device::Cpu)?
                }
                _ => {
                    return Err(Error::ModelContract(format!(
                        "unsupported initializer type {}",
                        w.dtype
                    )));
                }
            };
            weights.insert(w.name, t);
        }
        let supported = [
            "Conv",
            "ConvTranspose",
            "BatchNormalization",
            "Sigmoid",
            "Relu",
            "LeakyRelu",
            "Add",
            "Sub",
            "Mul",
            "Pow",
            "Concat",
            "Slice",
            "Reshape",
            "Transpose",
            "Resize",
            "MaxPool",
            "AveragePool",
        ];
        let mut uses = HashMap::new();
        for n in &graph.nodes {
            if !supported.contains(&n.op.as_str()) || n.output.len() != 1 {
                return Err(Error::ModelContract(format!(
                    "unsupported detector node {}: {}",
                    n.name, n.op
                )));
            }
            for i in &n.input {
                *uses.entry(i.clone()).or_insert(0) += 1;
            }
        }
        Ok(Self {
            nodes: graph.nodes,
            weights,
            uses,
        })
    }
    pub fn forward(&self, input: Tensor) -> Result<HashMap<String, Tensor>> {
        let _flush = matches!(input.device(), Device::Cpu).then(FlushDenormals::new);
        let mut values = self.weights.clone();
        values.insert("images".into(), input);
        let mut uses = self.uses.clone();
        for n in &self.nodes {
            let get = |i: usize| -> Result<&Tensor> {
                values
                    .get(n.input.get(i).ok_or_else(|| {
                        Error::ModelContract(format!("missing input in {}", n.name))
                    })?)
                    .ok_or_else(|| Error::ModelContract(format!("missing tensor in {}", n.name)))
            };
            let x = get(0)?;
            let output = match n.op.as_str() {
                "Conv" | "ConvTranspose" => {
                    let p = n.padding()?;
                    let s = n.pair("strides", 1)?;
                    let d = n.pair("dilations", 1)?;
                    let y = if n.op == "Conv"
                        && matches!(x.device(), Device::Cpu)
                        && n.int("group", 1) == 1
                    {
                        conv_cpu(x, get(1)?, p, s, d)?
                    } else if n.op == "Conv" {
                        let (_, c, h, w) = x.dims4()?;
                        // Candle 0.9.2's CPU tiled convolution mistakes contiguous
                        // NCHW for NHWC when C == H == W. Force distinct spatial
                        // strides for that shape so its layout conversion runs.
                        let input = if matches!(x.device(), Device::Cpu) && c == h && h == w {
                            x.transpose(2, 3)?.contiguous()?.transpose(2, 3)?
                        } else {
                            x.contiguous()?
                        };
                        input.conv2d(get(1)?, p, s, d, n.int("group", 1) as usize)?
                    } else {
                        if n.int("group", 1) != 1 {
                            return Err(Error::ModelContract(
                                "grouped transpose convolution".into(),
                            ));
                        }
                        if matches!(x.device(), Device::Cpu) {
                            conv_transpose_cpu(x, get(1)?, p, s, d)?
                        } else {
                            x.contiguous()?.conv_transpose2d(get(1)?, p, 0, s, d)?
                        }
                    };
                    if n.input.len() > 2 {
                        y.broadcast_add(&get(2)?.reshape((1, (), 1, 1))?)?
                    } else {
                        y
                    }
                }
                "BatchNormalization" => {
                    let shape = (1, (), 1, 1);
                    let mean = get(3)?.reshape(shape)?;
                    let var = get(4)?.reshape(shape)?;
                    let scale = get(1)?.reshape(shape)?;
                    let bias = get(2)?.reshape(shape)?;
                    x.broadcast_sub(&mean)?
                        .broadcast_div(&(var + f64::from(n.float("epsilon", 1e-5)))?.sqrt()?)?
                        .broadcast_mul(&scale)?
                        .broadcast_add(&bias)?
                }
                "Sigmoid" => candle_nn::ops::sigmoid(x)?,
                "Relu" => x.relu()?,
                "LeakyRelu" => x.maximum(&(x * f64::from(n.float("alpha", 0.01)))?)?,
                "Add" => x.broadcast_add(get(1)?)?,
                "Sub" => x.broadcast_sub(get(1)?)?,
                "Mul" => x.broadcast_mul(get(1)?)?,
                "Pow" => x.powf(get(1)?.flatten_all()?.to_vec1::<f32>()?[0] as f64)?,
                "Concat" => {
                    Tensor::cat(&n.input.iter().map(|s| &values[s]).collect::<Vec<_>>(), {
                        let a = n.int("axis", 0);
                        if a < 0 {
                            (x.rank() as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })?
                }
                "Transpose" => x.permute(
                    n.ints("perm", &[])
                        .iter()
                        .map(|&x| x as usize)
                        .collect::<Vec<_>>(),
                )?,
                "Reshape" => {
                    let mut dims = get(1)?.to_vec1::<i64>()?;
                    for (i, d) in dims.iter_mut().enumerate() {
                        if *d == 0 {
                            *d = x.dims()[i] as i64;
                        }
                    }
                    let known: usize = dims
                        .iter()
                        .filter(|&&d| d > 0)
                        .map(|&d| d as usize)
                        .product();
                    let dims: Vec<_> = dims
                        .iter()
                        .map(|&d| {
                            if d == -1 {
                                x.elem_count() / known
                            } else {
                                d as usize
                            }
                        })
                        .collect();
                    x.contiguous()?.reshape(dims)?
                }
                "Slice" => {
                    let starts = get(1)?.to_vec1::<i64>()?;
                    let ends = get(2)?.to_vec1::<i64>()?;
                    let axes = get(3)?.to_vec1::<i64>()?;
                    let steps = get(4)?.to_vec1::<i64>()?;
                    let mut y = x.clone();
                    for (((start, end), axis), step) in
                        starts.into_iter().zip(ends).zip(axes).zip(steps)
                    {
                        if start < 0 || end < 0 || axis < 0 || step <= 0 {
                            return Err(Error::ModelContract("unsupported detector slice".into()));
                        }
                        let axis = axis as usize;
                        let end = (end as usize).min(y.dim(axis)?);
                        let start = (start as usize).min(end);
                        y = y.narrow(axis, start, end - start)?;
                        if step > 1 {
                            let ids: Vec<_> = (0..end - start)
                                .step_by(step as usize)
                                .map(|x| x as u32)
                                .collect();
                            y = y.index_select(&Tensor::new(ids, y.device())?, axis)?;
                        }
                    }
                    y.contiguous()?
                }
                "Resize" => {
                    let scales = get(2)?.to_vec1::<f32>()?;
                    if scales.len() != 4 || scales[0] != 1. || scales[1] != 1. {
                        return Err(Error::ModelContract("unsupported detector resize".into()));
                    }
                    x.contiguous()?.upsample_nearest2d(
                        (x.dim(2)? as f32 * scales[2]) as usize,
                        (x.dim(3)? as f32 * scales[3]) as usize,
                    )?
                }
                "MaxPool" | "AveragePool" => {
                    let k = n.pair("kernel_shape", 1)?;
                    let s = n.pair("strides", 1)?;
                    let p = n.padding()?;
                    let mut y = x.clone();
                    if p > 0 {
                        // SPP max pooling must pad with -infinity, not zero.
                        let (b, c, h, w) = y.dims4()?;
                        let v = if n.op == "MaxPool" {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        };
                        let horizontal = Tensor::full(v, (b, c, h, p), y.device())?;
                        y = Tensor::cat(&[&horizontal, &y, &horizontal], 3)?;
                        let vertical = Tensor::full(v, (b, c, p, w + 2 * p), y.device())?;
                        y = Tensor::cat(&[&vertical, &y, &vertical], 2)?;
                    }
                    if n.op == "MaxPool" {
                        y.contiguous()?.max_pool2d_with_stride(k, s)?
                    } else {
                        y.contiguous()?.avg_pool2d_with_stride(k, s)?
                    }
                }
                _ => unreachable!("validated graph operators"),
            };
            values.insert(n.output[0].clone(), output);
            for i in &n.input {
                if let Some(left) = uses.get_mut(i) {
                    *left -= 1;
                    if *left == 0
                        && !self.weights.contains_key(i)
                        && !["blk", "seg", "det"].contains(&i.as_str())
                    {
                        values.remove(i);
                    }
                }
            }
        }
        ["blk", "seg", "det"]
            .into_iter()
            .map(|s| {
                values
                    .remove(s)
                    .map(|t| (s.to_owned(), t))
                    .ok_or_else(|| Error::ModelContract(format!("missing detector output {s}")))
            })
            .collect()
    }
}

/// The most floats one band of a CPU convolution's column matrix holds (64 MB).
const BAND_FLOATS: usize = 16 << 20;

/// `ConvTranspose` on the CPU as matrix products and a scatter of their columns (col2im).
///
/// Candle 0.9.2's CPU transposed convolution takes one dot product per output pixel through
/// its own `vec_dot`, which is vectorized only when the whole binary is compiled with AVX2 (or
/// NEON): on a stock x86-64 build it is a scalar loop, and the detector's upsampling layers,
/// half its work even with NEON, dominated a Windows scan that took over a minute. Here the arithmetic is one product of the
/// weights, `[c_out·k·k, c_in]`, with a band of input rows, `[c_in, rows·w]`, through gemm,
/// which picks AVX2/FMA or NEON at run time; each column then adds into the k×k output pixels
/// it belongs to. Bands keep that column matrix bounded on the 1024-square canvas.
fn conv_transpose_cpu(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (batch, c_in, in_h, in_w) = x.dims4()?;
    let (k_in, c_out, k_h, k_w) = kernel.dims4()?;
    if k_in != c_in {
        return Err(Error::ModelContract(
            "transpose convolution channel mismatch".into(),
        ));
    }
    let span = |input: usize, k: usize| {
        ((input - 1) * stride + dilation * (k - 1) + 1)
            .checked_sub(2 * padding)
            .ok_or_else(|| Error::ModelContract("transpose convolution padding".into()))
    };
    let (out_h, out_w) = (span(in_h, k_h)?, span(in_w, k_w)?);
    let taps = k_h * k_w;
    let weights = kernel.reshape((c_in, c_out * taps))?.t()?.contiguous()?;
    let rows = (BAND_FLOATS / (c_out * taps * in_w)).clamp(1, in_h);
    let plane = out_h * out_w;
    let mut out = vec![0f32; batch * c_out * plane];
    for b in 0..batch {
        let image = x.get(b)?;
        let out = &mut out[b * c_out * plane..(b + 1) * c_out * plane];
        for y0 in (0..in_h).step_by(rows) {
            let n = rows.min(in_h - y0);
            let band = image
                .narrow(1, y0, n)?
                .contiguous()?
                .reshape((c_in, n * in_w))?;
            let cols = weights.matmul(&band)?;
            let (storage, layout) = cols.storage_and_layout();
            let Storage::Cpu(storage) = &*storage else {
                unreachable!("CPU tensors")
            };
            let cols = &storage.as_slice::<f32>()?[layout.start_offset()..];
            let width = n * in_w;
            out.par_chunks_mut(plane)
                .enumerate()
                .for_each(|(channel, plane)| {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let tap = (channel * taps + ky * k_w + kx) * width;
                            for iy in 0..n {
                                let Some(oy) = ((y0 + iy) * stride + ky * dilation)
                                    .checked_sub(padding)
                                    .filter(|&oy| oy < out_h)
                                else {
                                    continue;
                                };
                                let row = &cols[tap + iy * in_w..tap + (iy + 1) * in_w];
                                let dst = &mut plane[oy * out_w..(oy + 1) * out_w];
                                for (ix, v) in row.iter().enumerate() {
                                    if let Some(ox) = (ix * stride + kx * dilation)
                                        .checked_sub(padding)
                                        .filter(|&ox| ox < out_w)
                                    {
                                        dst[ox] += v;
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }
    Ok(Tensor::from_vec(
        out,
        (batch, c_out, out_h, out_w),
        x.device(),
    )?)
}

/// `Conv` on the CPU as one matrix product per band of output rows (im2col).
///
/// Candle 0.9.2's CPU convolution gathers its column matrix one element at a time in tiles of
/// 512 pixels, at a fraction of gemm's speed. Here each column-matrix row, one input channel
/// at one kernel tap, is filled a whole output row at a time from one input row, and the
/// product of the weights, `[c_out, c_in·k·k]`, with a band of it, `[c_in·k·k, rows·w]`, lands
/// directly in those rows of every output channel.
fn conv_cpu(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (batch, c_in, in_h, in_w) = x.dims4()?;
    let (c_out, k_in, k_h, k_w) = kernel.dims4()?;
    if k_in != c_in {
        return Err(Error::ModelContract("convolution channel mismatch".into()));
    }
    let span = |input: usize, k: usize| {
        (input + 2 * padding)
            .checked_sub(dilation * (k - 1) + 1)
            .map(|v| v / stride + 1)
            .ok_or_else(|| Error::ModelContract("convolution kernel exceeds input".into()))
    };
    let (out_h, out_w) = (span(in_h, k_h)?, span(in_w, k_w)?);
    let depth = c_in * k_h * k_w;
    let weights = kernel.reshape((c_out, depth))?.contiguous()?;
    let x = x.contiguous()?;
    let (storage, layout) = x.storage_and_layout();
    let Storage::Cpu(storage) = &*storage else {
        unreachable!("CPU tensors")
    };
    let input = &storage.as_slice::<f32>()?[layout.start_offset()..];
    let rows = (BAND_FLOATS / (depth * out_w)).clamp(1, out_h);
    let plane = out_h * out_w;
    let mut out = vec![0f32; batch * c_out * plane];
    for b in 0..batch {
        let input = &input[b * c_in * in_h * in_w..(b + 1) * c_in * in_h * in_w];
        let out = &mut out[b * c_out * plane..(b + 1) * c_out * plane];
        for y0 in (0..out_h).step_by(rows) {
            let n = rows.min(out_h - y0);
            let width = n * out_w;
            let mut cols = vec![0f32; depth * width];
            cols.par_chunks_mut(width).enumerate().for_each(|(r, dst)| {
                let (c, ky, kx) = (r / (k_h * k_w), r / k_w % k_h, r % k_w);
                let src = &input[c * in_h * in_w..(c + 1) * in_h * in_w];
                for oy in 0..n {
                    let Some(iy) = ((y0 + oy) * stride + ky * dilation)
                        .checked_sub(padding)
                        .filter(|&iy| iy < in_h)
                    else {
                        continue;
                    };
                    let src = &src[iy * in_w..(iy + 1) * in_w];
                    let dst = &mut dst[oy * out_w..(oy + 1) * out_w];
                    for (ox, v) in dst.iter_mut().enumerate() {
                        if let Some(ix) = (ox * stride + kx * dilation)
                            .checked_sub(padding)
                            .filter(|&ix| ix < in_w)
                        {
                            *v = src[ix];
                        }
                    }
                }
            });
            let product = weights.matmul(&Tensor::from_vec(cols, (depth, width), x.device())?)?;
            let (storage, layout) = product.storage_and_layout();
            let Storage::Cpu(storage) = &*storage else {
                unreachable!("CPU tensors")
            };
            let product = &storage.as_slice::<f32>()?[layout.start_offset()..];
            out.par_chunks_mut(plane)
                .zip(product.par_chunks(width))
                .for_each(|(plane, row)| {
                    plane[y0 * out_w..y0 * out_w + width].copy_from_slice(row);
                });
        }
    }
    Ok(Tensor::from_vec(
        out,
        (batch, c_out, out_h, out_w),
        x.device(),
    )?)
}

/// Subnormal floats flushed to zero (x86's FTZ and DAZ) on this thread and every thread of
/// rayon's global pool, Candle's and gemm's workers, for the guard's lifetime; each thread's
/// own setting comes back on drop.
///
/// The detector reads a snip on a 1024 square that is mostly black padding, and its
/// activations there decay through LeakyReLU, SiLU and sigmoid tails into subnormals. x86
/// cores take a microcode assist on every subnormal operand, around a hundred times the
/// cost of a normal one: on an i7-7700K a 690×240 snip's detection took 27.6 s, and 2.9 s
/// flushed. Apple's cores handle subnormals at full speed, which is why no Mac showed it.
/// Values below 1.2e-38 cannot move a detection.
struct FlushDenormals {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    saved: (u32, Vec<u32>),
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod mxcsr {
    #[cfg(target_arch = "x86")]
    use std::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64 as arch;
    /// Flush-to-zero (bit 15) and denormals-are-zero (bit 6).
    const FLUSH: u32 = 0x8040;
    /// Sets FTZ and DAZ on the calling thread, answering its previous MXCSR.
    pub fn flush() -> u32 {
        // SSE is in the baseline of every x86-64 target. `_mm_setcsr` is deprecated only in
        // favour of inline assembly, for code that relies on the compiler's ordering of
        // floating-point operations around it; nothing here does.
        #[allow(deprecated)]
        unsafe {
            let previous = arch::_mm_getcsr();
            arch::_mm_setcsr(previous | FLUSH);
            previous
        }
    }
    pub fn restore(previous: u32) {
        #[allow(deprecated)]
        unsafe {
            arch::_mm_setcsr(previous)
        }
    }
}
impl FlushDenormals {
    fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        return Self {
            saved: (mxcsr::flush(), rayon::broadcast(|_| mxcsr::flush())),
        };
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Self {}
    }
}
impl Drop for FlushDenormals {
    fn drop(&mut self) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let (own, pool) = &self.saved;
            rayon::broadcast(|context| mxcsr::restore(pool[context.index()]));
            mxcsr::restore(*own);
        }
    }
}
