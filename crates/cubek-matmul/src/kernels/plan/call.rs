use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::Product;
use crate::kernels::tune::key::{MatmulTuneKey, create_key};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::metadata::Metadata;
use cubek_std::Hardware;
use cubek_std::plan::BoxedPlan;
use std::sync::Arc;

/// A plan of the matmul family: what a product's launch tries, on one [`MatmulCall`].
pub type MatmulPlan = BoxedPlan<MatmulCall, TensorHandle>;

/// One product a plan runs: its two inputs, each as it is stored, the element it is written in,
/// and the device it runs on as the plan prepares for it.
#[derive(Clone)]
pub struct MatmulCall {
    pub hw: Hardware,
    pub client: Client,
    /// The device's identity, which an autotune keys its measurements by.
    pub tune_id: CubeTuneId,
    pub lhs: MatmulCallInput,
    pub rhs: MatmulCallInput,
    /// The element the product is written in.
    pub out: ElemType,
    /// The buffer the product is written into, where the caller allocated it at the product's
    /// shape; a launch allocates one otherwise.
    pub destination: Option<TensorHandle>,
}

impl MatmulCall {
    /// The key an autotune measures the product under: its extents, the element it is written in,
    /// and each input as it lies.
    pub fn key(&self) -> MatmulTuneKey {
        create_key(
            self.out,
            (&self.lhs.meta, self.lhs.tensor.kind()),
            (&self.rhs.meta, self.rhs.tensor.kind()),
        )
    }

    /// The kind of product its inputs make: what a candidate listed for the other kind declines.
    pub(crate) fn product(&self) -> Product {
        Product::of([self.lhs.tensor.kind(), self.rhs.tensor.kind()])
    }

    /// What a tuning round measures for `key`: the product as large as the key's bucket stands
    /// for, its inputs grown with zeros to the bucket's rows, reduction and width
    /// ([`MatmulTuneKey::bucketed`]), so the winner is elected on the bucket's largest product
    /// whatever order its calls come in. The call itself where nothing falls short of the bucket,
    /// or where an input that does cannot be laid out alike: a packed or a strided one.
    pub fn representative(key: &MatmulTuneKey, call: &Self) -> Self {
        let shape = key.shape();
        // `[.., m, k] · [.., k, n]`: what each input's last two axes grow to.
        let lhs = Self::grown(&call.client, &call.lhs, [shape.m, shape.k]);
        let rhs = Self::grown(&call.client, &call.rhs, [shape.k, shape.n]);
        match (lhs, rhs) {
            (Some(lhs), Some(rhs)) if lhs.is_some() || rhs.is_some() => Self {
                lhs: lhs.map_or_else(|| call.lhs.clone(), MatmulCallInput::plain),
                rhs: rhs.map_or_else(|| call.rhs.clone(), MatmulCallInput::plain),
                // A grown input makes a product of another shape than the caller's buffer, so
                // the round's launches allocate their own.
                destination: None,
                ..call.clone()
            },
            _ => call.clone(),
        }
    }

    /// `input` grown with zeros to `extents` along its last two axes, each where it falls short:
    /// `Some(None)` where neither does, `None` where one does and the input cannot be grown.
    fn grown(
        client: &Client,
        input: &MatmulCallInput,
        extents: [usize; 2],
    ) -> Option<Option<PlainTensor>> {
        let shape = input
            .meta
            .logical_shape()
            .expect("a tensor's tiling describes its own rank");
        let rank = shape.rank();
        let short: Vec<(usize, usize)> = (rank.checked_sub(2)?..rank)
            .zip(extents)
            .filter(|&(axis, extent)| extent > shape[axis])
            .collect();
        if short.is_empty() {
            return Some(None);
        }
        let StoredTensor::Plain(plain) = &input.tensor else {
            return None;
        };
        let mut grown = plain.clone();
        for (axis, extent) in short {
            grown = grown.zeros_grown(client, axis, extent)?;
        }
        Some(Some(grown))
    }
}

/// One input of a [`MatmulCall`]: the tensor as it is stored, how it lies as its caller states it,
/// and, where it is packed, how the caller's framework dequantizes it.
///
/// The framework's dequantize reads every scheme, and the kernels here do not: a plan reaches for
/// it where no kernel reads the scheme, and where the decode declines an input. It is the caller's
/// own tensor that the framework reads, so the caller binds it here and the plan never sees it.
#[derive(Clone)]
pub struct MatmulCallInput {
    pub tensor: StoredTensor,
    /// Its logical shape and strides, which a tuning key reads.
    pub meta: Metadata,
    pub framework_dequantize: Option<FrameworkDequantize>,
}

/// The framework's dequantize of one packed input, into the element the product is written in.
pub type FrameworkDequantize = Arc<dyn Fn() -> Result<PlainTensor, Refusal> + Send + Sync>;

impl MatmulCallInput {
    /// A float input, which no dequantize is asked about.
    pub fn plain(tensor: PlainTensor) -> Self {
        Self {
            meta: tensor.meta(),
            tensor: StoredTensor::Plain(tensor),
            framework_dequantize: None,
        }
    }
}
