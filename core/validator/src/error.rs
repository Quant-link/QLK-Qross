//! Error types for the validator selection module

use thiserror::Error;
use crate::ValidatorId;

/// Result type alias
pub type Result<T> = std::result::Result<T, ValidatorError>;

/// Validator selection error types
#[derive(Error, Debug)]
pub enum ValidatorError {
    /// Insufficient validators available
    #[error("Insufficient validators: available {available}, required {required}")]
    InsufficientValidators { available: usize, required: usize },
    
    /// Validator not found
    #[error("Validator not found: {0}")]
    ValidatorNotFound(ValidatorId),
    
    /// Invalid validator configuration
    #[error("Invalid validator configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Stake concentration too high
    #[error("Stake concentration too high")]
    StakeConcentrationTooHigh,
    
    /// Average reputation too low
    #[error("Average reputation too low")]
    AverageReputationTooLow,
    
    /// Selection strategy error
    #[error("Selection strategy error: {0}")]
    SelectionStrategyError(String),
    
    /// Reputation calculation error
    #[error("Reputation calculation error: {0}")]
    ReputationCalculationError(String),
    
    /// Performance analysis error
    #[error("Performance analysis error: {0}")]
    PerformanceAnalysisError(String),
    
    /// Invalid performance data
    #[error("Invalid performance data: {0}")]
    InvalidPerformanceData(String),
    
    /// Invalid reputation update
    #[error("Invalid reputation update: {0}")]
    InvalidReputationUpdate(String),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// Redis error
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Network error
    #[error("Network error: {0}")]
    Network(String),
    
    /// Timeout error
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Concurrent modification error
    #[error("Concurrent modification detected for validator: {0}")]
    ConcurrentModification(ValidatorId),
    
    /// Invalid epoch
    #[error("Invalid epoch: {0}")]
    InvalidEpoch(u64),
    
    /// Selection already exists
    #[error("Selection already exists for epoch: {0}")]
    SelectionAlreadyExists(u64),
    
    /// Validator already active
    #[error("Validator already active: {0}")]
    ValidatorAlreadyActive(ValidatorId),
    
    /// Validator not active
    #[error("Validator not active: {0}")]
    ValidatorNotActive(ValidatorId),
    
    /// Insufficient stake
    #[error("Insufficient stake: validator {validator}, required {required}, available {available}")]
    InsufficientStake {
        validator: ValidatorId,
        required: u128,
        available: u128,
    },
    
    /// Invalid stake amount
    #[error("Invalid stake amount: {0}")]
    InvalidStakeAmount(u128),
    
    /// Validator in grace period
    #[error("Validator in grace period: {0}")]
    ValidatorInGracePeriod(ValidatorId),
    
    /// Validator recently slashed
    #[error("Validator recently slashed: {0}")]
    ValidatorRecentlySlashed(ValidatorId),
    
    /// Maximum consecutive selections exceeded
    #[error("Maximum consecutive selections exceeded for validator: {0}")]
    MaxConsecutiveSelectionsExceeded(ValidatorId),
    
    /// Invalid selection parameters
    #[error("Invalid selection parameters: {0}")]
    InvalidSelectionParameters(String),
    
    /// Selection validation failed
    #[error("Selection validation failed: {0}")]
    SelectionValidationFailed(String),
    
    /// Reputation score out of range
    #[error("Reputation score out of range: {0}")]
    ReputationScoreOutOfRange(u8),
    
    /// Performance score out of range
    #[error("Performance score out of range: {0}")]
    PerformanceScoreOutOfRange(f64),
    
    /// Historical data insufficient
    #[error("Insufficient historical data for validator: {0}")]
    InsufficientHistoricalData(ValidatorId),
    
    /// Data provider error
    #[error("Data provider error: {0}")]
    DataProviderError(String),
    
    /// Metrics error
    #[error("Metrics error: {0}")]
    MetricsError(String),
}

impl ValidatorError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            ValidatorError::Database(_) => true,
            ValidatorError::Redis(_) => true,
            ValidatorError::Network(_) => true,
            ValidatorError::Timeout(_) => true,
            ValidatorError::ConcurrentModification(_) => true,
            ValidatorError::DataProviderError(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ValidatorError::InsufficientValidators { .. } => ErrorSeverity::Critical,
            ValidatorError::ValidatorNotFound(_) => ErrorSeverity::Medium,
            ValidatorError::InvalidConfiguration(_) => ErrorSeverity::High,
            ValidatorError::StakeConcentrationTooHigh => ErrorSeverity::High,
            ValidatorError::AverageReputationTooLow => ErrorSeverity::Medium,
            ValidatorError::SelectionStrategyError(_) => ErrorSeverity::High,
            ValidatorError::ReputationCalculationError(_) => ErrorSeverity::Medium,
            ValidatorError::PerformanceAnalysisError(_) => ErrorSeverity::Medium,
            ValidatorError::InvalidPerformanceData(_) => ErrorSeverity::Low,
            ValidatorError::InvalidReputationUpdate(_) => ErrorSeverity::Low,
            ValidatorError::Database(_) => ErrorSeverity::High,
            ValidatorError::Redis(_) => ErrorSeverity::Medium,
            ValidatorError::Serialization(_) => ErrorSeverity::Medium,
            ValidatorError::Network(_) => ErrorSeverity::Medium,
            ValidatorError::Timeout(_) => ErrorSeverity::Low,
            ValidatorError::Internal(_) => ErrorSeverity::High,
            ValidatorError::ConcurrentModification(_) => ErrorSeverity::Medium,
            ValidatorError::InvalidEpoch(_) => ErrorSeverity::Low,
            ValidatorError::SelectionAlreadyExists(_) => ErrorSeverity::Low,
            ValidatorError::ValidatorAlreadyActive(_) => ErrorSeverity::Low,
            ValidatorError::ValidatorNotActive(_) => ErrorSeverity::Medium,
            ValidatorError::InsufficientStake { .. } => ErrorSeverity::Medium,
            ValidatorError::InvalidStakeAmount(_) => ErrorSeverity::Low,
            ValidatorError::ValidatorInGracePeriod(_) => ErrorSeverity::Low,
            ValidatorError::ValidatorRecentlySlashed(_) => ErrorSeverity::Medium,
            ValidatorError::MaxConsecutiveSelectionsExceeded(_) => ErrorSeverity::Low,
            ValidatorError::InvalidSelectionParameters(_) => ErrorSeverity::Medium,
            ValidatorError::SelectionValidationFailed(_) => ErrorSeverity::High,
            ValidatorError::ReputationScoreOutOfRange(_) => ErrorSeverity::Low,
            ValidatorError::PerformanceScoreOutOfRange(_) => ErrorSeverity::Low,
            ValidatorError::InsufficientHistoricalData(_) => ErrorSeverity::Low,
            ValidatorError::DataProviderError(_) => ErrorSeverity::Medium,
            ValidatorError::MetricsError(_) => ErrorSeverity::Low,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            ValidatorError::InsufficientValidators { .. } => "insufficient_validators",
            ValidatorError::ValidatorNotFound(_) => "validator_not_found",
            ValidatorError::InvalidConfiguration(_) => "invalid_config",
            ValidatorError::StakeConcentrationTooHigh => "stake_concentration",
            ValidatorError::AverageReputationTooLow => "reputation",
            ValidatorError::SelectionStrategyError(_) => "selection_strategy",
            ValidatorError::ReputationCalculationError(_) => "reputation",
            ValidatorError::PerformanceAnalysisError(_) => "performance",
            ValidatorError::InvalidPerformanceData(_) => "performance",
            ValidatorError::InvalidReputationUpdate(_) => "reputation",
            ValidatorError::Database(_) => "database",
            ValidatorError::Redis(_) => "redis",
            ValidatorError::Serialization(_) => "serialization",
            ValidatorError::Network(_) => "network",
            ValidatorError::Timeout(_) => "timeout",
            ValidatorError::Internal(_) => "internal",
            ValidatorError::ConcurrentModification(_) => "concurrency",
            ValidatorError::InvalidEpoch(_) => "epoch",
            ValidatorError::SelectionAlreadyExists(_) => "selection",
            ValidatorError::ValidatorAlreadyActive(_) => "validator_state",
            ValidatorError::ValidatorNotActive(_) => "validator_state",
            ValidatorError::InsufficientStake { .. } => "stake",
            ValidatorError::InvalidStakeAmount(_) => "stake",
            ValidatorError::ValidatorInGracePeriod(_) => "grace_period",
            ValidatorError::ValidatorRecentlySlashed(_) => "slashing",
            ValidatorError::MaxConsecutiveSelectionsExceeded(_) => "consecutive_selections",
            ValidatorError::InvalidSelectionParameters(_) => "selection_params",
            ValidatorError::SelectionValidationFailed(_) => "selection_validation",
            ValidatorError::ReputationScoreOutOfRange(_) => "reputation",
            ValidatorError::PerformanceScoreOutOfRange(_) => "performance",
            ValidatorError::InsufficientHistoricalData(_) => "historical_data",
            ValidatorError::DataProviderError(_) => "data_provider",
            ValidatorError::MetricsError(_) => "metrics",
        }
    }
    
    /// Check if error should trigger emergency procedures
    pub fn should_trigger_emergency(&self) -> bool {
        matches!(
            self,
            ValidatorError::InsufficientValidators { .. } |
            ValidatorError::StakeConcentrationTooHigh |
            ValidatorError::SelectionValidationFailed(_)
        )
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "low"),
            ErrorSeverity::Medium => write!(f, "medium"),
            ErrorSeverity::High => write!(f, "high"),
            ErrorSeverity::Critical => write!(f, "critical"),
        }
    }
}
