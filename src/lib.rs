// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(warnings)]

pub mod error;
pub mod parse;

pub use error::{AnamnesisError, Result};
pub use parse::{Dtype, QuantScheme, SafetensorsHeader, TensorEntry, TensorRole};
