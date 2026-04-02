// SPDX-License-Identifier: MIT OR Apache-2.0

/// Errors produced by anamnesis operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AnamnesisError {
    /// A format decoding failure (malformed header, invalid tensor metadata).
    #[error("parse error: {reason}")]
    Parse {
        /// Human-readable description of what went wrong.
        reason: String,
    },

    /// A recognized but unimplemented format or feature.
    #[error("unsupported format `{format}`: {detail}")]
    Unsupported {
        /// The format name (e.g., `"GPTQ"`, `"safetensors"`).
        format: String,
        /// What specifically is not supported.
        detail: String,
    },

    /// A file system error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl From<safetensors::SafeTensorError> for AnamnesisError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Parse {
            reason: format!("failed to parse safetensors header: {e}"),
        }
    }
}

#[cfg(any(feature = "npz", feature = "pth"))]
impl From<zip::result::ZipError> for AnamnesisError {
    fn from(e: zip::result::ZipError) -> Self {
        Self::Parse {
            reason: format!("failed to read ZIP archive: {e}"),
        }
    }
}

/// A convenience alias for `Result<T, AnamnesisError>`.
pub type Result<T> = std::result::Result<T, AnamnesisError>;
