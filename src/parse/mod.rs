/// Safetensors header parsing, tensor classification, and quantization scheme detection.
pub mod safetensors;

pub use safetensors::{Dtype, QuantScheme, SafetensorsHeader, TensorEntry, TensorRole};
