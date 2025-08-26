//! Error types for the state reader module

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, StateReaderError>;

/// State reader error types
#[derive(Error, Debug)]
pub enum StateReaderError {
    /// Network connection error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    
    /// JSON-RPC error
    #[error("RPC error: {0}")]
    Rpc(String),
    
    /// WebSocket error
    #[error("WebSocket error: {0}")]
    WebSocket(String),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Invalid response format
    #[error("Invalid response format: {0}")]
    InvalidResponse(String),
    
    /// Chain not supported
    #[error("Chain not supported: {0}")]
    UnsupportedChain(String),
    
    /// Block not found
    #[error("Block not found: {0}")]
    BlockNotFound(u64),
    
    /// Transaction not found
    #[error("Transaction not found: {0}")]
    TransactionNotFound(String),
    
    /// Address invalid
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    /// Timeout error
    #[error("Request timeout")]
    Timeout,
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// Redis error
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    /// Subscription closed
    #[error("Subscription closed")]
    SubscriptionClosed,
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Parse error
    #[error("Parse error: {0}")]
    Parse(String),
    
    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),
    
    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    /// Service unavailable
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
}

impl StateReaderError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            StateReaderError::Network(_) => true,
            StateReaderError::Timeout => true,
            StateReaderError::RateLimitExceeded => true,
            StateReaderError::ServiceUnavailable(_) => true,
            StateReaderError::Rpc(msg) => {
                // Some RPC errors are retryable
                msg.contains("timeout") || 
                msg.contains("rate limit") || 
                msg.contains("service unavailable")
            }
            _ => false,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            StateReaderError::Network(_) => "network",
            StateReaderError::Rpc(_) => "rpc",
            StateReaderError::WebSocket(_) => "websocket",
            StateReaderError::Serialization(_) => "serialization",
            StateReaderError::InvalidResponse(_) => "invalid_response",
            StateReaderError::UnsupportedChain(_) => "unsupported_chain",
            StateReaderError::BlockNotFound(_) => "block_not_found",
            StateReaderError::TransactionNotFound(_) => "transaction_not_found",
            StateReaderError::InvalidAddress(_) => "invalid_address",
            StateReaderError::RateLimitExceeded => "rate_limit",
            StateReaderError::Timeout => "timeout",
            StateReaderError::Configuration(_) => "configuration",
            StateReaderError::Database(_) => "database",
            StateReaderError::Redis(_) => "redis",
            StateReaderError::SubscriptionClosed => "subscription_closed",
            StateReaderError::Internal(_) => "internal",
            StateReaderError::Io(_) => "io",
            StateReaderError::Parse(_) => "parse",
            StateReaderError::Authentication(_) => "authentication",
            StateReaderError::PermissionDenied(_) => "permission_denied",
            StateReaderError::ServiceUnavailable(_) => "service_unavailable",
        }
    }
}
