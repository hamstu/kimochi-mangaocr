//! Native execution of the pinned detector's ONNX tensor graph. ONNX is a file
//! format here, not an inference runtime. Weights are uploaded once; dead tensors
//! are released after their last consumer to bound page-inference memory.
use crate::{Error, Result};
use candle_core::{Device, Tensor};
use prost::Message;
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
                    let y = if n.op == "Conv" {
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
                        x.contiguous()?.conv_transpose2d(get(1)?, p, 0, s, d)?
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
