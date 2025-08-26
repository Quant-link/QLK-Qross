//! Error types for the slashing module

use thiserror::Error;
use crate::{ValidatorId, InvestigationId, EvidenceId};

/// Result type alias
pub type Result<T> = std::result::Result<T, SlashingError>;

/// Slashing error types
#[derive(Error, Debug)]
pub enum SlashingError {
    /// Investigation not found
    #[error("Investigation not found: {0}")]
    InvestigationNotFound(ValidatorId),
    
    /// Evidence not found
    #[error("Evidence not found: {0}")]
    EvidenceNotFound(EvidenceId),
    
    /// Invalid evidence
    #[error("Invalid evidence: {0}")]
    InvalidEvidence(String),
    
    /// Evidence verification failed
    #[error("Evidence verification failed: {0}")]
    EvidenceVerificationFailed(String),
    
    /// Insufficient stake for slashing
    #[error("Insufficient stake for slashing: validator {validator}, required {required}, available {available}")]
    InsufficientStake {
        validator: ValidatorId,
        required: u128,
        available: u128,
    },
    
    /// Validator not found
    #[error("Validator not found: {0}")]
    ValidatorNotFound(ValidatorId),
    
    /// Invalid penalty calculation
    #[error("Invalid penalty calculation: {0}")]
    InvalidPenaltyCalculation(String),
    
    /// Slashing cooldown active
    #[error("Slashing cooldown active for validator: {0}")]
    SlashingCooldownActive(ValidatorId),
    
    /// Maximum slashing limit exceeded
    #[error("Maximum slashing limit exceeded for validator: {0}")]
    MaxSlashingExceeded(ValidatorId),
    
    /// Emergency halt active
    #[error("Emergency halt active - slashing suspended")]
    EmergencyHaltActive,
    
    /// Network partition detected
    #[error("Network partition detected - slashing suspended")]
    NetworkPartitionDetected,
    
    /// Redistribution failed
    #[error("Redistribution failed: {0}")]
    RedistributionFailed(String),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// Redis error
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    Cryptographic(String),
    
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
    
    /// Appeal not found
    #[error("Appeal not found: {0}")]
    AppealNotFound(uuid::Uuid),
    
    /// Invalid appeal
    #[error("Invalid appeal: {0}")]
    InvalidAppeal(String),
    
    /// Appeal deadline exceeded
    #[error("Appeal deadline exceeded")]
    AppealDeadlineExceeded,
    
    /// Unauthorized operation
    #[error("Unauthorized operation: {0}")]
    Unauthorized(String),
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded for validator: {0}")]
    RateLimitExceeded(ValidatorId),
    
    /// Invalid signature
    #[error("Invalid signature")]
    InvalidSignature,
    
    /// Duplicate evidence
    #[error("Duplicate evidence for validator: {0}")]
    DuplicateEvidence(ValidatorId),
    
    /// Evidence too old
    #[error("Evidence too old: {0}")]
    EvidenceTooOld(String),
    
    /// Evidence too recent
    #[error("Evidence too recent: {0}")]
    EvidenceTooRecent(String),
    
    /// Validator in grace period
    #[error("Validator in grace period: {0}")]
    ValidatorInGracePeriod(ValidatorId),
}

impl SlashingError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            SlashingError::Database(_) => true,
            SlashingError::Redis(_) => true,
            SlashingError::Network(_) => true,
            SlashingError::Timeout(_) => true,
            SlashingError::ConcurrentModification(_) => true,
            SlashingError::RateLimitExceeded(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            SlashingError::InvestigationNotFound(_) => ErrorSeverity::Medium,
            SlashingError::EvidenceNotFound(_) => ErrorSeverity::Medium,
            SlashingError::InvalidEvidence(_) => ErrorSeverity::High,
            SlashingError::EvidenceVerificationFailed(_) => ErrorSeverity::High,
            SlashingError::InsufficientStake { .. } => ErrorSeverity::High,
            SlashingError::ValidatorNotFound(_) => ErrorSeverity::Medium,
            SlashingError::InvalidPenaltyCalculation(_) => ErrorSeverity::High,
            SlashingError::SlashingCooldownActive(_) => ErrorSeverity::Low,
            SlashingError::MaxSlashingExceeded(_) => ErrorSeverity::High,
            SlashingError::EmergencyHaltActive => ErrorSeverity::Critical,
            SlashingError::NetworkPartitionDetected => ErrorSeverity::Critical,
            SlashingError::RedistributionFailed(_) => ErrorSeverity::High,
            SlashingError::InvalidConfiguration(_) => ErrorSeverity::Critical,
            SlashingError::Database(_) => ErrorSeverity::High,
            SlashingError::Redis(_) => ErrorSeverity::Medium,
            SlashingError::Serialization(_) => ErrorSeverity::Medium,
            SlashingError::Cryptographic(_) => ErrorSeverity::High,
            SlashingError::Network(_) => ErrorSeverity::Medium,
            SlashingError::Timeout(_) => ErrorSeverity::Low,
            SlashingError::Internal(_) => ErrorSeverity::High,
            SlashingError::ConcurrentModification(_) => ErrorSeverity::Medium,
            SlashingError::AppealNotFound(_) => ErrorSeverity::Low,
            SlashingError::InvalidAppeal(_) => ErrorSeverity::Medium,
            SlashingError::AppealDeadlineExceeded => ErrorSeverity::Low,
            SlashingError::Unauthorized(_) => ErrorSeverity::High,
            SlashingError::RateLimitExceeded(_) => ErrorSeverity::Low,
            SlashingError::InvalidSignature => ErrorSeverity::High,
            SlashingError::DuplicateEvidence(_) => ErrorSeverity::Low,
            SlashingError::EvidenceTooOld(_) => ErrorSeverity::Low,
            SlashingError::EvidenceTooRecent(_) => ErrorSeverity::Low,
            SlashingError::ValidatorInGracePeriod(_) => ErrorSeverity::Low,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            SlashingError::InvestigationNotFound(_) => "investigation",
            SlashingError::EvidenceNotFound(_) => "evidence",
            SlashingError::InvalidEvidence(_) => "evidence",
            SlashingError::EvidenceVerificationFailed(_) => "evidence",
            SlashingError::InsufficientStake { .. } => "stake",
            SlashingError::ValidatorNotFound(_) => "validator",
            SlashingError::InvalidPenaltyCalculation(_) => "penalty",
            SlashingError::SlashingCooldownActive(_) => "cooldown",
            SlashingError::MaxSlashingExceeded(_) => "limits",
            SlashingError::EmergencyHaltActive => "emergency",
            SlashingError::NetworkPartitionDetected => "network",
            SlashingError::RedistributionFailed(_) => "redistribution",
            SlashingError::InvalidConfiguration(_) => "config",
            SlashingError::Database(_) => "database",
            SlashingError::Redis(_) => "redis",
            SlashingError::Serialization(_) => "serialization",
            SlashingError::Cryptographic(_) => "crypto",
            SlashingError::Network(_) => "network",
            SlashingError::Timeout(_) => "timeout",
            SlashingError::Internal(_) => "internal",
            SlashingError::ConcurrentModification(_) => "concurrency",
            SlashingError::AppealNotFound(_) => "appeal",
            SlashingError::InvalidAppeal(_) => "appeal",
            SlashingError::AppealDeadlineExceeded => "appeal",
            SlashingError::Unauthorized(_) => "auth",
            SlashingError::RateLimitExceeded(_) => "rate_limit",
            SlashingError::InvalidSignature => "signature",
            SlashingError::DuplicateEvidence(_) => "evidence",
            SlashingError::EvidenceTooOld(_) => "evidence",
            SlashingError::EvidenceTooRecent(_) => "evidence",
            SlashingError::ValidatorInGracePeriod(_) => "grace_period",
        }
    }
    
    /// Check if error should trigger emergency halt
    pub fn should_trigger_emergency_halt(&self) -> bool {
        matches!(
            self,
            SlashingError::NetworkPartitionDetected |
            SlashingError::InvalidConfiguration(_) |
            SlashingError::RedistributionFailed(_)
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
