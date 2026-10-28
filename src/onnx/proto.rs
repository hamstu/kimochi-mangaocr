//! The parts of ONNX's protobuf schema (onnx.proto) needed to write a model, and a small
//! builder for graphs.
use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetId>,
}
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetId {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    /// AttributeType: FLOAT 1, INT 2, INTS 7.
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    /// FLOAT 1, INT64 7.
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TensorTypeProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorTypeProto {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}
#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}
#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
}

pub const FLOAT: i32 = 1;
pub const INT64: i32 = 7;

/// A graph input's or output's dimension: fixed, or named and free.
pub enum Dim {
    Fixed(i64),
    Free(&'static str),
}

pub fn int(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        i,
        r#type: 2,
        ..Default::default()
    }
}
pub fn ints(name: &str, ints: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        ints: ints.to_vec(),
        r#type: 7,
        ..Default::default()
    }
}
pub fn float(name: &str, f: f32) -> AttributeProto {
    AttributeProto {
        name: name.into(),
        f,
        r#type: 1,
        ..Default::default()
    }
}

/// Nodes, weights and the graph's interface, written out as one model.
#[derive(Default)]
pub struct Builder {
    graph: GraphProto,
    next: usize,
}
impl Builder {
    fn fresh(&mut self, hint: &str) -> String {
        self.next += 1;
        format!("{hint}.{}", self.next)
    }
    /// A float initializer under `name`.
    pub fn weight(&mut self, name: &str, dims: &[usize], data: &[f32]) -> String {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "{name}");
        self.graph.initializer.push(TensorProto {
            dims: dims.iter().map(|&d| d as i64).collect(),
            data_type: FLOAT,
            name: name.into(),
            raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        name.into()
    }
    /// An int64 constant, as an initializer.
    pub fn ints(&mut self, dims: &[usize], data: &[i64]) -> String {
        let name = self.fresh("const");
        self.graph.initializer.push(TensorProto {
            dims: dims.iter().map(|&d| d as i64).collect(),
            data_type: INT64,
            name: name.clone(),
            raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        name
    }
    /// A 1-D shape or index list.
    pub fn list(&mut self, data: &[i64]) -> String {
        self.ints(&[data.len()], data)
    }
    /// A float scalar.
    pub fn scalar(&mut self, v: f32) -> String {
        let name = self.fresh("scalar");
        self.weight(&name, &[], &[v])
    }
    pub fn op(&mut self, op: &str, inputs: &[&str], attrs: Vec<AttributeProto>) -> String {
        let out = self.fresh(op);
        self.graph.node.push(NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: vec![out.clone()],
            name: out.clone(),
            op_type: op.into(),
            attribute: attrs,
        });
        out
    }
    pub fn input(&mut self, name: &str, elem: i32, dims: &[Dim]) -> String {
        self.graph.input.push(value_info(name, elem, dims));
        name.into()
    }
    /// Names `value` as the graph output `name`.
    pub fn output(&mut self, value: &str, name: &str, elem: i32, dims: &[Dim]) {
        self.graph.node.push(NodeProto {
            input: vec![value.into()],
            output: vec![name.into()],
            name: format!("{name}.out"),
            op_type: "Identity".into(),
            attribute: Vec::new(),
        });
        self.graph.output.push(value_info(name, elem, dims));
    }
    /// The serialized model, opset 17.
    pub fn finish(self, name: &str) -> Vec<u8> {
        let mut graph = self.graph;
        graph.name = name.into();
        ModelProto {
            ir_version: 8,
            producer_name: "mangaocr".into(),
            graph: Some(graph),
            opset_import: vec![OperatorSetId {
                domain: String::new(),
                version: 17,
            }],
        }
        .encode_to_vec()
    }
}
fn value_info(name: &str, elem: i32, dims: &[Dim]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.into(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: elem,
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|d| match d {
                            Dim::Fixed(v) => Dimension {
                                dim_value: Some(*v),
                                dim_param: None,
                            },
                            Dim::Free(p) => Dimension {
                                dim_value: None,
                                dim_param: Some((*p).into()),
                            },
                        })
                        .collect(),
                }),
            }),
        }),
    }
}
