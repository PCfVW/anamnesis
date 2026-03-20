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

/// A convenience alias for `Result<T, AnamnesisError>`.
pub type Result<T> = std::result::Result<T, AnamnesisError>;
