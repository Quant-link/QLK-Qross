//! Error types for performance optimization engine

use thiserror::Error;

/// Result type for optimization operations
pub type Result<T> = std::result::Result<T, OptimizationError>;

/// Performance optimization errors
#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Fee optimization failed: {0}")]
    FeeOptimizationFailed(String),
    
    #[error("Batch processing failed: {0}")]
    BatchProcessingFailed(String),
    
    #[error("Priority queue error: {0}")]
    PriorityQueueError(String),
    
    #[error("Gas prediction failed: {0}")]
    GasPredictionFailed(String),
    
    #[error("Cross-chain optimization failed: {0}")]
    CrossChainOptimizationFailed(String),
    
    #[error("AMM optimization failed: {0}")]
    AMMOptimizationFailed(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Timeout error: operation timed out after {duration:?}")]
    TimeoutError { duration: std::time::Duration },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Invalid transaction: {reason}")]
    InvalidTransaction { reason: String },
    
    #[error("Insufficient liquidity for optimization")]
    InsufficientLiquidity,
    
    #[error("Spam detected: transaction rejected")]
    SpamDetected,
    
    #[error("Model prediction failed: {model}")]
    ModelPredictionFailed { model: String },
    
    #[error("Optimization cache error: {0}")]
    CacheError(String),
    
    #[error("Metrics collection failed: {0}")]
    MetricsError(String),
    
    #[error("Algorithm adaptation failed: {0}")]
    AdaptationFailed(String),
    
    #[error("Cost modeling error: {0}")]
    CostModelingError(String),
    
    #[error("Performance monitoring error: {0}")]
    MonitoringError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
    
    #[error("External service error: {service} - {error}")]
    ExternalServiceError { service: String, error: String },
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Consensus error: {0}")]
    ConsensusError(#[from] qross_consensus::ConsensusError),
    
    #[error("ZK verification error: {0}")]
    ZKVerificationError(#[from] qross_zk_verification::VerificationError),
    
    #[error("Network error: {0}")]
    P2PNetworkError(#[from] qross_p2p_network::NetworkError),
    
    #[error("Liquidity management error: {0}")]
    LiquidityError(#[from] qross_liquidity_management::LiquidityError),
    
    #[error("Security error: {0}")]
    SecurityError(#[from] qross_security_risk_management::SecurityError),
}

impl OptimizationError {
    /// Create a fee optimization error
    pub fn fee_optimization_failed(msg: impl Into<String>) -> Self {
        Self::FeeOptimizationFailed(msg.into())
    }
    
    /// Create a batch processing error
    pub fn batch_processing_failed(msg: impl Into<String>) -> Self {
        Self::BatchProcessingFailed(msg.into())
    }
    
    /// Create a priority queue error
    pub fn priority_queue_error(msg: impl Into<String>) -> Self {
        Self::PriorityQueueError(msg.into())
    }
    
    /// Create a gas prediction error
    pub fn gas_prediction_failed(msg: impl Into<String>) -> Self {
        Self::GasPredictionFailed(msg.into())
    }
    
    /// Create a cross-chain optimization error
    pub fn cross_chain_optimization_failed(msg: impl Into<String>) -> Self {
        Self::CrossChainOptimizationFailed(msg.into())
    }
    
    /// Create an AMM optimization error
    pub fn amm_optimization_failed(msg: impl Into<String>) -> Self {
        Self::AMMOptimizationFailed(msg.into())
    }
    
    /// Create a network error
    pub fn network_error(msg: impl Into<String>) -> Self {
        Self::NetworkError(msg.into())
    }
    
    /// Create a configuration error
    pub fn configuration_error(msg: impl Into<String>) -> Self {
        Self::ConfigurationError(msg.into())
    }
    
    /// Create a validation error
    pub fn validation_error(msg: impl Into<String>) -> Self {
        Self::ValidationError(msg.into())
    }
    
    /// Create a timeout error
    pub fn timeout_error(duration: std::time::Duration) -> Self {
        Self::TimeoutError { duration }
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>) -> Self {
        Self::ResourceExhausted { resource: resource.into() }
    }
    
    /// Create an invalid transaction error
    pub fn invalid_transaction(reason: impl Into<String>) -> Self {
        Self::InvalidTransaction { reason: reason.into() }
    }
    
    /// Create a model prediction error
    pub fn model_prediction_failed(model: impl Into<String>) -> Self {
        Self::ModelPredictionFailed { model: model.into() }
    }
    
    /// Create a cache error
    pub fn cache_error(msg: impl Into<String>) -> Self {
        Self::CacheError(msg.into())
    }
    
    /// Create a metrics error
    pub fn metrics_error(msg: impl Into<String>) -> Self {
        Self::MetricsError(msg.into())
    }
    
    /// Create an adaptation error
    pub fn adaptation_failed(msg: impl Into<String>) -> Self {
        Self::AdaptationFailed(msg.into())
    }
    
    /// Create a cost modeling error
    pub fn cost_modeling_error(msg: impl Into<String>) -> Self {
        Self::CostModelingError(msg.into())
    }
    
    /// Create a monitoring error
    pub fn monitoring_error(msg: impl Into<String>) -> Self {
        Self::MonitoringError(msg.into())
    }
    
    /// Create an internal error
    pub fn internal_error(msg: impl Into<String>) -> Self {
        Self::InternalError(msg.into())
    }
    
    /// Create an external service error
    pub fn external_service_error(service: impl Into<String>, error: impl Into<String>) -> Self {
        Self::ExternalServiceError {
            service: service.into(),
            error: error.into(),
        }
    }
    
    /// Create a parse error
    pub fn parse_error(msg: impl Into<String>) -> Self {
        Self::ParseError(msg.into())
    }
    
    /// Create a database error
    pub fn database_error(msg: impl Into<String>) -> Self {
        Self::DatabaseError(msg.into())
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError(_) |
            Self::TimeoutError { .. } |
            Self::ExternalServiceError { .. } |
            Self::ResourceExhausted { .. } |
            Self::DatabaseError(_)
        )
    }
    
    /// Check if error is critical
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            Self::InternalError(_) |
            Self::ConfigurationError(_) |
            Self::SecurityError(_) |
            Self::ConsensusError(_)
        )
    }
    
    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::FeeOptimizationFailed(_) |
            Self::BatchProcessingFailed(_) |
            Self::PriorityQueueError(_) |
            Self::GasPredictionFailed(_) |
            Self::CrossChainOptimizationFailed(_) |
            Self::AMMOptimizationFailed(_) => ErrorCategory::Optimization,
            
            Self::NetworkError(_) |
            Self::P2PNetworkError(_) |
            Self::ExternalServiceError { .. } => ErrorCategory::Network,
            
            Self::ConfigurationError(_) |
            Self::ValidationError(_) |
            Self::InvalidTransaction { .. } => ErrorCategory::Configuration,
            
            Self::TimeoutError { .. } |
            Self::ResourceExhausted { .. } => ErrorCategory::Resource,
            
            Self::SecurityError(_) |
            Self::SpamDetected => ErrorCategory::Security,
            
            Self::InternalError(_) |
            Self::SerializationError(_) |
            Self::IoError(_) |
            Self::ParseError(_) |
            Self::DatabaseError(_) => ErrorCategory::System,
            
            _ => ErrorCategory::Other,
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    Optimization,
    Network,
    Configuration,
    Resource,
    Security,
    System,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = OptimizationError::fee_optimization_failed("test error");
        assert!(matches!(error, OptimizationError::FeeOptimizationFailed(_)));
    }
    
    #[test]
    fn test_error_retryable() {
        let network_error = OptimizationError::network_error("connection failed");
        assert!(network_error.is_retryable());
        
        let config_error = OptimizationError::configuration_error("invalid config");
        assert!(!config_error.is_retryable());
    }
    
    #[test]
    fn test_error_critical() {
        let internal_error = OptimizationError::internal_error("system failure");
        assert!(internal_error.is_critical());
        
        let timeout_error = OptimizationError::timeout_error(std::time::Duration::from_secs(30));
        assert!(!timeout_error.is_critical());
    }
    
    #[test]
    fn test_error_category() {
        let fee_error = OptimizationError::fee_optimization_failed("fee calculation failed");
        assert_eq!(fee_error.category(), ErrorCategory::Optimization);
        
        let network_error = OptimizationError::network_error("connection lost");
        assert_eq!(network_error.category(), ErrorCategory::Network);
        
        let security_error = OptimizationError::SpamDetected;
        assert_eq!(security_error.category(), ErrorCategory::Security);
    }
}
