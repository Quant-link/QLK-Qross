//! Error types for trusted setup ceremony coordination

use thiserror::Error;
use crate::{CeremonyId, ParameterSetId};

/// Result type alias
pub type Result<T> = std::result::Result<T, CeremonyError>;

/// Trusted setup ceremony error types
#[derive(Error, Debug)]
pub enum CeremonyError {
    /// Ceremony not found
    #[error("Ceremony not found: {0}")]
    CeremonyNotFound(CeremonyId),
    
    /// Parameter set not found
    #[error("Parameter set not found: {0}")]
    ParameterSetNotFound(ParameterSetId),
    
    /// Emergency halt is active
    #[error("Emergency halt is active, ceremony operations suspended")]
    EmergencyHaltActive,
    
    /// Insufficient participants for ceremony
    #[error("Insufficient participants: required {required}, available {available}")]
    InsufficientParticipants { required: usize, available: usize },
    
    /// Insufficient participation in ceremony round
    #[error("Insufficient participation: required {required}, actual {actual}")]
    InsufficientParticipation { required: usize, actual: usize },
    
    /// Insufficient signatures for beacon
    #[error("Insufficient signatures: required {required}, collected {collected}")]
    InsufficientSignatures { required: usize, collected: usize },
    
    /// Insufficient randomness quality
    #[error("Insufficient randomness quality: score {score}, threshold {threshold}")]
    InsufficientRandomnessQuality { score: f64, threshold: f64 },
    
    /// Validator selection failed
    #[error("Validator selection failed: {0}")]
    ValidatorSelectionFailed(String),
    
    /// Invalid validator for ceremony
    #[error("Invalid validator: {0}")]
    InvalidValidator(String),
    
    /// Validator performance below threshold
    #[error("Validator performance below threshold: {validator_id}, score: {score}, threshold: {threshold}")]
    ValidatorPerformanceBelowThreshold {
        validator_id: String,
        score: f64,
        threshold: f64,
    },
    
    /// Contribution timeout
    #[error("Contribution collection timeout")]
    ContributionTimeout,
    
    /// Invalid contribution
    #[error("Invalid contribution from validator {validator_id}: {reason}")]
    InvalidContribution { validator_id: String, reason: String },
    
    /// Contribution verification failed
    #[error("Contribution verification failed: {0}")]
    ContributionVerificationFailed(String),
    
    /// Parameter generation failed
    #[error("Parameter generation failed: {0}")]
    ParameterGenerationFailed(String),
    
    /// Parameter verification failed
    #[error("Parameter verification failed: {0}")]
    ParameterVerificationFailed(String),
    
    /// Ceremony timeout
    #[error("Ceremony execution timeout")]
    CeremonyTimeout,
    
    /// Round timeout
    #[error("Round execution timeout")]
    RoundTimeout,
    
    /// Beacon generation failed
    #[error("Random beacon generation failed: {0}")]
    BeaconGenerationFailed(String),
    
    /// Beacon verification failed
    #[error("Random beacon verification failed: {0}")]
    BeaconVerificationFailed(String),
    
    /// Signature verification failed
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    
    /// Invalid ceremony state
    #[error("Invalid ceremony state: {0}")]
    InvalidCeremonyState(String),
    
    /// Ceremony restart failed
    #[error("Ceremony restart failed: {0}")]
    CeremonyRestartFailed(String),
    
    /// Maximum restart attempts exceeded
    #[error("Maximum restart attempts exceeded: {attempts}")]
    MaxRestartAttemptsExceeded { attempts: usize },
    
    /// Consensus rejection
    #[error("Consensus rejected ceremony parameters")]
    ConsensusRejection,
    
    /// Network partition detected
    #[error("Network partition detected, ceremony suspended")]
    NetworkPartition,
    
    /// Validator network integration error
    #[error("Validator network integration error: {0}")]
    ValidatorNetworkIntegration(String),
    
    /// Consensus integration error
    #[error("Consensus integration error: {0}")]
    ConsensusIntegration(String),
    
    /// Cryptographic operation failed
    #[error("Cryptographic operation failed: {0}")]
    CryptographicOperation(String),
    
    /// Entropy collection failed
    #[error("Entropy collection failed: {0}")]
    EntropyCollectionFailed(String),
    
    /// Quality assessment failed
    #[error("Quality assessment failed: {0}")]
    QualityAssessmentFailed(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Resource exhaustion
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    /// Computation timeout
    #[error("Computation timeout: {0}")]
    ComputationTimeout(String),
    
    /// Memory limit exceeded
    #[error("Memory limit exceeded: used {used}, limit {limit}")]
    MemoryLimitExceeded { used: usize, limit: usize },
    
    /// Invalid parameter format
    #[error("Invalid parameter format: {0}")]
    InvalidParameterFormat(String),
    
    /// Parameter integrity check failed
    #[error("Parameter integrity check failed")]
    ParameterIntegrityCheckFailed,
    
    /// Contribution chain validation failed
    #[error("Contribution chain validation failed")]
    ContributionChainValidationFailed,
    
    /// Concurrent modification detected
    #[error("Concurrent modification detected")]
    ConcurrentModification,
    
    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),
    
    /// Metrics error
    #[error("Metrics error: {0}")]
    Metrics(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl CeremonyError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            CeremonyError::ContributionTimeout => true,
            CeremonyError::CeremonyTimeout => true,
            CeremonyError::RoundTimeout => true,
            CeremonyError::NetworkPartition => true,
            CeremonyError::ResourceExhaustion(_) => true,
            CeremonyError::ComputationTimeout(_) => true,
            CeremonyError::ConcurrentModification => true,
            CeremonyError::ValidatorNetworkIntegration(_) => true,
            CeremonyError::ConsensusIntegration(_) => true,
            CeremonyError::Io(_) => true,
            _ => false,
        }
    }
    
    /// Check if the error is critical and should trigger emergency halt
    pub fn is_critical(&self) -> bool {
        match self {
            CeremonyError::EmergencyHaltActive => true,
            CeremonyError::NetworkPartition => true,
            CeremonyError::ParameterIntegrityCheckFailed => true,
            CeremonyError::ContributionChainValidationFailed => true,
            CeremonyError::ConsensusRejection => true,
            CeremonyError::MaxRestartAttemptsExceeded { .. } => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            CeremonyError::CeremonyNotFound(_) => ErrorSeverity::Medium,
            CeremonyError::ParameterSetNotFound(_) => ErrorSeverity::Medium,
            CeremonyError::EmergencyHaltActive => ErrorSeverity::Critical,
            CeremonyError::InsufficientParticipants { .. } => ErrorSeverity::High,
            CeremonyError::InsufficientParticipation { .. } => ErrorSeverity::Medium,
            CeremonyError::InsufficientSignatures { .. } => ErrorSeverity::High,
            CeremonyError::InsufficientRandomnessQuality { .. } => ErrorSeverity::High,
            CeremonyError::ValidatorSelectionFailed(_) => ErrorSeverity::Medium,
            CeremonyError::InvalidValidator(_) => ErrorSeverity::Medium,
            CeremonyError::ValidatorPerformanceBelowThreshold { .. } => ErrorSeverity::Low,
            CeremonyError::ContributionTimeout => ErrorSeverity::Medium,
            CeremonyError::InvalidContribution { .. } => ErrorSeverity::High,
            CeremonyError::ContributionVerificationFailed(_) => ErrorSeverity::High,
            CeremonyError::ParameterGenerationFailed(_) => ErrorSeverity::High,
            CeremonyError::ParameterVerificationFailed(_) => ErrorSeverity::High,
            CeremonyError::CeremonyTimeout => ErrorSeverity::Medium,
            CeremonyError::RoundTimeout => ErrorSeverity::Low,
            CeremonyError::BeaconGenerationFailed(_) => ErrorSeverity::High,
            CeremonyError::BeaconVerificationFailed(_) => ErrorSeverity::High,
            CeremonyError::SignatureVerificationFailed => ErrorSeverity::High,
            CeremonyError::InvalidCeremonyState(_) => ErrorSeverity::Medium,
            CeremonyError::CeremonyRestartFailed(_) => ErrorSeverity::Medium,
            CeremonyError::MaxRestartAttemptsExceeded { .. } => ErrorSeverity::Critical,
            CeremonyError::ConsensusRejection => ErrorSeverity::Critical,
            CeremonyError::NetworkPartition => ErrorSeverity::Critical,
            CeremonyError::ValidatorNetworkIntegration(_) => ErrorSeverity::Medium,
            CeremonyError::ConsensusIntegration(_) => ErrorSeverity::High,
            CeremonyError::CryptographicOperation(_) => ErrorSeverity::High,
            CeremonyError::EntropyCollectionFailed(_) => ErrorSeverity::Medium,
            CeremonyError::QualityAssessmentFailed(_) => ErrorSeverity::Medium,
            CeremonyError::Configuration(_) => ErrorSeverity::High,
            CeremonyError::Serialization(_) => ErrorSeverity::Medium,
            CeremonyError::Io(_) => ErrorSeverity::Medium,
            CeremonyError::ResourceExhaustion(_) => ErrorSeverity::Medium,
            CeremonyError::ComputationTimeout(_) => ErrorSeverity::Low,
            CeremonyError::MemoryLimitExceeded { .. } => ErrorSeverity::Medium,
            CeremonyError::InvalidParameterFormat(_) => ErrorSeverity::Medium,
            CeremonyError::ParameterIntegrityCheckFailed => ErrorSeverity::Critical,
            CeremonyError::ContributionChainValidationFailed => ErrorSeverity::Critical,
            CeremonyError::ConcurrentModification => ErrorSeverity::Medium,
            CeremonyError::Cache(_) => ErrorSeverity::Low,
            CeremonyError::Metrics(_) => ErrorSeverity::Low,
            CeremonyError::Internal(_) => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            CeremonyError::CeremonyNotFound(_) => "ceremony_not_found",
            CeremonyError::ParameterSetNotFound(_) => "parameter_set_not_found",
            CeremonyError::EmergencyHaltActive => "emergency_halt",
            CeremonyError::InsufficientParticipants { .. } => "insufficient_participants",
            CeremonyError::InsufficientParticipation { .. } => "insufficient_participation",
            CeremonyError::InsufficientSignatures { .. } => "insufficient_signatures",
            CeremonyError::InsufficientRandomnessQuality { .. } => "insufficient_randomness_quality",
            CeremonyError::ValidatorSelectionFailed(_) => "validator_selection_failed",
            CeremonyError::InvalidValidator(_) => "invalid_validator",
            CeremonyError::ValidatorPerformanceBelowThreshold { .. } => "validator_performance_below_threshold",
            CeremonyError::ContributionTimeout => "contribution_timeout",
            CeremonyError::InvalidContribution { .. } => "invalid_contribution",
            CeremonyError::ContributionVerificationFailed(_) => "contribution_verification_failed",
            CeremonyError::ParameterGenerationFailed(_) => "parameter_generation_failed",
            CeremonyError::ParameterVerificationFailed(_) => "parameter_verification_failed",
            CeremonyError::CeremonyTimeout => "ceremony_timeout",
            CeremonyError::RoundTimeout => "round_timeout",
            CeremonyError::BeaconGenerationFailed(_) => "beacon_generation_failed",
            CeremonyError::BeaconVerificationFailed(_) => "beacon_verification_failed",
            CeremonyError::SignatureVerificationFailed => "signature_verification_failed",
            CeremonyError::InvalidCeremonyState(_) => "invalid_ceremony_state",
            CeremonyError::CeremonyRestartFailed(_) => "ceremony_restart_failed",
            CeremonyError::MaxRestartAttemptsExceeded { .. } => "max_restart_attempts_exceeded",
            CeremonyError::ConsensusRejection => "consensus_rejection",
            CeremonyError::NetworkPartition => "network_partition",
            CeremonyError::ValidatorNetworkIntegration(_) => "validator_network_integration",
            CeremonyError::ConsensusIntegration(_) => "consensus_integration",
            CeremonyError::CryptographicOperation(_) => "cryptographic_operation",
            CeremonyError::EntropyCollectionFailed(_) => "entropy_collection_failed",
            CeremonyError::QualityAssessmentFailed(_) => "quality_assessment_failed",
            CeremonyError::Configuration(_) => "configuration",
            CeremonyError::Serialization(_) => "serialization",
            CeremonyError::Io(_) => "io",
            CeremonyError::ResourceExhaustion(_) => "resource_exhaustion",
            CeremonyError::ComputationTimeout(_) => "computation_timeout",
            CeremonyError::MemoryLimitExceeded { .. } => "memory_limit_exceeded",
            CeremonyError::InvalidParameterFormat(_) => "invalid_parameter_format",
            CeremonyError::ParameterIntegrityCheckFailed => "parameter_integrity_check_failed",
            CeremonyError::ContributionChainValidationFailed => "contribution_chain_validation_failed",
            CeremonyError::ConcurrentModification => "concurrent_modification",
            CeremonyError::Cache(_) => "cache",
            CeremonyError::Metrics(_) => "metrics",
            CeremonyError::Internal(_) => "internal",
        }
    }
    
    /// Check if error requires validator slashing
    pub fn requires_slashing(&self) -> bool {
        matches!(
            self,
            CeremonyError::InvalidContribution { .. } |
            CeremonyError::SignatureVerificationFailed |
            CeremonyError::InvalidValidator(_)
        )
    }
    
    /// Check if error affects ceremony integrity
    pub fn affects_ceremony_integrity(&self) -> bool {
        matches!(
            self,
            CeremonyError::ParameterIntegrityCheckFailed |
            CeremonyError::ContributionChainValidationFailed |
            CeremonyError::InsufficientRandomnessQuality { .. } |
            CeremonyError::ParameterVerificationFailed(_)
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
