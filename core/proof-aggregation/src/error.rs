//! Error types for proof aggregation protocol

use thiserror::Error;
use crate::{AggregationId, ProofId, StateTransitionId, DependencyId};

/// Result type alias
pub type Result<T> = std::result::Result<T, AggregationError>;

/// Proof aggregation error types
#[derive(Error, Debug)]
pub enum AggregationError {
    /// Aggregation not found
    #[error("Aggregation not found: {0}")]
    AggregationNotFound(AggregationId),
    
    /// Invalid proof submitted for aggregation
    #[error("Invalid proof: {0}")]
    InvalidProof(ProofId),
    
    /// Invalid state transition
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(StateTransitionId),
    
    /// Empty proof set submitted
    #[error("Empty proof set submitted for aggregation")]
    EmptyProofSet,
    
    /// Too many proofs in aggregation
    #[error("Too many proofs: submitted {submitted}, max allowed {max_allowed}")]
    TooManyProofs { submitted: usize, max_allowed: usize },
    
    /// Cyclic dependency detected
    #[error("Cyclic dependency detected in proof set")]
    CyclicDependency,
    
    /// Unsatisfied dependency
    #[error("Unsatisfied dependency: {0}")]
    UnsatisfiedDependency(DependencyId),
    
    /// Dependency timeout
    #[error("Dependency resolution timeout")]
    DependencyTimeout,
    
    /// Dependency depth exceeded
    #[error("Dependency depth exceeded: {actual} > {max_allowed}")]
    DependencyDepthExceeded { actual: usize, max_allowed: usize },
    
    /// Chain state not found
    #[error("Chain state not found: {0}")]
    ChainStateNotFound(String),
    
    /// Emergency halt active
    #[error("Emergency halt is active, aggregation suspended")]
    EmergencyHaltActive,
    
    /// Insufficient validators for aggregation
    #[error("Insufficient validators: required {required}, available {available}")]
    InsufficientValidators { required: usize, available: usize },
    
    /// Batch verification failed
    #[error("Batch verification failed")]
    BatchVerificationFailed,
    
    /// Batch processing timeout
    #[error("Batch processing timeout")]
    BatchTimeout,
    
    /// Finality submission failed
    #[error("Finality submission failed: {0}")]
    FinalitySubmissionFailed(String),
    
    /// Finality timeout
    #[error("Finality determination timeout")]
    FinalityTimeout,
    
    /// Consensus integration error
    #[error("Consensus integration error: {0}")]
    ConsensusIntegration(String),
    
    /// State synchronization error
    #[error("State synchronization error: {0}")]
    StateSynchronization(String),
    
    /// Network partition detected
    #[error("Network partition detected, aggregation suspended")]
    NetworkPartition,
    
    /// Validator allocation failed
    #[error("Validator allocation failed: {0}")]
    ValidatorAllocation(String),
    
    /// Resource exhaustion
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// ZK circuit error
    #[error("ZK circuit error: {0}")]
    ZkCircuit(#[from] qross_zk_circuits::CircuitError),
    
    /// Consensus error
    #[error("Consensus error: {0}")]
    Consensus(String),
    
    /// Timeout error
    #[error("Operation timeout: {0}")]
    Timeout(String),
    
    /// Concurrent modification error
    #[error("Concurrent modification detected")]
    ConcurrentModification,
    
    /// Invalid aggregation state
    #[error("Invalid aggregation state: {0}")]
    InvalidState(String),
    
    /// Proof composition failed
    #[error("Proof composition failed: {0}")]
    ProofComposition(String),
    
    /// Signature verification failed
    #[error("Signature verification failed")]
    SignatureVerification,
    
    /// Insufficient signatures
    #[error("Insufficient signatures: required {required}, collected {collected}")]
    InsufficientSignatures { required: usize, collected: usize },
    
    /// Invalid validator
    #[error("Invalid validator: {0}")]
    InvalidValidator(String),
    
    /// Performance degradation
    #[error("Performance degradation detected: {0}")]
    PerformanceDegradation(String),
    
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

impl AggregationError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            AggregationError::DependencyTimeout => true,
            AggregationError::BatchTimeout => true,
            AggregationError::FinalityTimeout => true,
            AggregationError::ResourceExhaustion(_) => true,
            AggregationError::NetworkPartition => true,
            AggregationError::ConcurrentModification => true,
            AggregationError::Timeout(_) => true,
            AggregationError::Io(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            AggregationError::AggregationNotFound(_) => ErrorSeverity::Medium,
            AggregationError::InvalidProof(_) => ErrorSeverity::High,
            AggregationError::InvalidStateTransition(_) => ErrorSeverity::High,
            AggregationError::EmptyProofSet => ErrorSeverity::Low,
            AggregationError::TooManyProofs { .. } => ErrorSeverity::Medium,
            AggregationError::CyclicDependency => ErrorSeverity::High,
            AggregationError::UnsatisfiedDependency(_) => ErrorSeverity::Medium,
            AggregationError::DependencyTimeout => ErrorSeverity::Medium,
            AggregationError::DependencyDepthExceeded { .. } => ErrorSeverity::Medium,
            AggregationError::ChainStateNotFound(_) => ErrorSeverity::Medium,
            AggregationError::EmergencyHaltActive => ErrorSeverity::Critical,
            AggregationError::InsufficientValidators { .. } => ErrorSeverity::High,
            AggregationError::BatchVerificationFailed => ErrorSeverity::High,
            AggregationError::BatchTimeout => ErrorSeverity::Medium,
            AggregationError::FinalitySubmissionFailed(_) => ErrorSeverity::High,
            AggregationError::FinalityTimeout => ErrorSeverity::Medium,
            AggregationError::ConsensusIntegration(_) => ErrorSeverity::High,
            AggregationError::StateSynchronization(_) => ErrorSeverity::Medium,
            AggregationError::NetworkPartition => ErrorSeverity::Critical,
            AggregationError::ValidatorAllocation(_) => ErrorSeverity::Medium,
            AggregationError::ResourceExhaustion(_) => ErrorSeverity::Medium,
            AggregationError::Configuration(_) => ErrorSeverity::High,
            AggregationError::Serialization(_) => ErrorSeverity::Medium,
            AggregationError::Io(_) => ErrorSeverity::Medium,
            AggregationError::ZkCircuit(_) => ErrorSeverity::High,
            AggregationError::Consensus(_) => ErrorSeverity::High,
            AggregationError::Timeout(_) => ErrorSeverity::Low,
            AggregationError::ConcurrentModification => ErrorSeverity::Medium,
            AggregationError::InvalidState(_) => ErrorSeverity::Medium,
            AggregationError::ProofComposition(_) => ErrorSeverity::High,
            AggregationError::SignatureVerification => ErrorSeverity::High,
            AggregationError::InsufficientSignatures { .. } => ErrorSeverity::High,
            AggregationError::InvalidValidator(_) => ErrorSeverity::Medium,
            AggregationError::PerformanceDegradation(_) => ErrorSeverity::Medium,
            AggregationError::Cache(_) => ErrorSeverity::Low,
            AggregationError::Metrics(_) => ErrorSeverity::Low,
            AggregationError::Internal(_) => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            AggregationError::AggregationNotFound(_) => "aggregation_not_found",
            AggregationError::InvalidProof(_) => "invalid_proof",
            AggregationError::InvalidStateTransition(_) => "invalid_state_transition",
            AggregationError::EmptyProofSet => "empty_proof_set",
            AggregationError::TooManyProofs { .. } => "too_many_proofs",
            AggregationError::CyclicDependency => "cyclic_dependency",
            AggregationError::UnsatisfiedDependency(_) => "unsatisfied_dependency",
            AggregationError::DependencyTimeout => "dependency_timeout",
            AggregationError::DependencyDepthExceeded { .. } => "dependency_depth_exceeded",
            AggregationError::ChainStateNotFound(_) => "chain_state_not_found",
            AggregationError::EmergencyHaltActive => "emergency_halt",
            AggregationError::InsufficientValidators { .. } => "insufficient_validators",
            AggregationError::BatchVerificationFailed => "batch_verification_failed",
            AggregationError::BatchTimeout => "batch_timeout",
            AggregationError::FinalitySubmissionFailed(_) => "finality_submission_failed",
            AggregationError::FinalityTimeout => "finality_timeout",
            AggregationError::ConsensusIntegration(_) => "consensus_integration",
            AggregationError::StateSynchronization(_) => "state_synchronization",
            AggregationError::NetworkPartition => "network_partition",
            AggregationError::ValidatorAllocation(_) => "validator_allocation",
            AggregationError::ResourceExhaustion(_) => "resource_exhaustion",
            AggregationError::Configuration(_) => "configuration",
            AggregationError::Serialization(_) => "serialization",
            AggregationError::Io(_) => "io",
            AggregationError::ZkCircuit(_) => "zk_circuit",
            AggregationError::Consensus(_) => "consensus",
            AggregationError::Timeout(_) => "timeout",
            AggregationError::ConcurrentModification => "concurrent_modification",
            AggregationError::InvalidState(_) => "invalid_state",
            AggregationError::ProofComposition(_) => "proof_composition",
            AggregationError::SignatureVerification => "signature_verification",
            AggregationError::InsufficientSignatures { .. } => "insufficient_signatures",
            AggregationError::InvalidValidator(_) => "invalid_validator",
            AggregationError::PerformanceDegradation(_) => "performance_degradation",
            AggregationError::Cache(_) => "cache",
            AggregationError::Metrics(_) => "metrics",
            AggregationError::Internal(_) => "internal",
        }
    }
    
    /// Check if error should trigger emergency halt
    pub fn should_trigger_emergency_halt(&self) -> bool {
        matches!(
            self,
            AggregationError::NetworkPartition |
            AggregationError::PerformanceDegradation(_) |
            AggregationError::InsufficientValidators { .. }
        )
    }
    
    /// Check if error affects consensus finality
    pub fn affects_finality(&self) -> bool {
        matches!(
            self,
            AggregationError::InvalidProof(_) |
            AggregationError::BatchVerificationFailed |
            AggregationError::ProofComposition(_) |
            AggregationError::SignatureVerification |
            AggregationError::InsufficientSignatures { .. }
        )
    }
    
    /// Check if error requires validator slashing
    pub fn requires_slashing(&self) -> bool {
        matches!(
            self,
            AggregationError::InvalidProof(_) |
            AggregationError::SignatureVerification |
            AggregationError::InvalidValidator(_)
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
