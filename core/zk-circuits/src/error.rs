//! Error types for the zk-STARK circuit library

use thiserror::Error;
use crate::CircuitId;

/// Result type alias
pub type Result<T> = std::result::Result<T, CircuitError>;

/// zk-STARK circuit error types
#[derive(Error, Debug)]
pub enum CircuitError {
    /// Circuit not found
    #[error("Circuit not found: {0}")]
    CircuitNotFound(CircuitId),
    
    /// Invalid circuit input
    #[error("Invalid circuit input: {0}")]
    InvalidInput(String),
    
    /// Trace generation failed
    #[error("Trace generation failed: {0}")]
    TraceGeneration(String),
    
    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGeneration(String),
    
    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    Verification(String),
    
    /// Circuit optimization failed
    #[error("Circuit optimization failed: {0}")]
    Optimization(String),
    
    /// Polynomial commitment error
    #[error("Polynomial commitment error: {0}")]
    PolynomialCommitment(String),
    
    /// Merkle tree error
    #[error("Merkle tree error: {0}")]
    MerkleTree(String),
    
    /// Recursive proof composition error
    #[error("Recursive proof composition error: {0}")]
    RecursiveComposition(String),
    
    /// Trusted setup error
    #[error("Trusted setup error: {0}")]
    TrustedSetup(String),
    
    /// Field arithmetic error
    #[error("Field arithmetic error: {0}")]
    FieldArithmetic(String),
    
    /// Constraint system error
    #[error("Constraint system error: {0}")]
    ConstraintSystem(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Resource exhaustion error
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    /// Timeout error
    #[error("Operation timeout: {0}")]
    Timeout(String),
    
    /// Concurrent access error
    #[error("Concurrent access error: {0}")]
    ConcurrentAccess(String),
    
    /// Invalid proof format
    #[error("Invalid proof format: {0}")]
    InvalidProofFormat(String),
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    /// Security violation
    #[error("Security violation: {0}")]
    SecurityViolation(String),
    
    /// Parameter mismatch
    #[error("Parameter mismatch: {0}")]
    ParameterMismatch(String),
    
    /// Degree bound exceeded
    #[error("Degree bound exceeded: expected {expected}, got {actual}")]
    DegreeBoundExceeded { expected: usize, actual: usize },
    
    /// Invalid field element
    #[error("Invalid field element: {0}")]
    InvalidFieldElement(String),
    
    /// Constraint violation
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    /// Batch processing error
    #[error("Batch processing error: {0}")]
    BatchProcessing(String),
    
    /// Worker pool error
    #[error("Worker pool error: {0}")]
    WorkerPool(String),
    
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

impl CircuitError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            CircuitError::Timeout(_) => true,
            CircuitError::ResourceExhaustion(_) => true,
            CircuitError::ConcurrentAccess(_) => true,
            CircuitError::WorkerPool(_) => true,
            CircuitError::Io(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            CircuitError::CircuitNotFound(_) => ErrorSeverity::Medium,
            CircuitError::InvalidInput(_) => ErrorSeverity::Medium,
            CircuitError::TraceGeneration(_) => ErrorSeverity::High,
            CircuitError::ProofGeneration(_) => ErrorSeverity::High,
            CircuitError::Verification(_) => ErrorSeverity::High,
            CircuitError::Optimization(_) => ErrorSeverity::Low,
            CircuitError::PolynomialCommitment(_) => ErrorSeverity::High,
            CircuitError::MerkleTree(_) => ErrorSeverity::Medium,
            CircuitError::RecursiveComposition(_) => ErrorSeverity::High,
            CircuitError::TrustedSetup(_) => ErrorSeverity::Critical,
            CircuitError::FieldArithmetic(_) => ErrorSeverity::High,
            CircuitError::ConstraintSystem(_) => ErrorSeverity::High,
            CircuitError::Serialization(_) => ErrorSeverity::Medium,
            CircuitError::Io(_) => ErrorSeverity::Medium,
            CircuitError::Configuration(_) => ErrorSeverity::High,
            CircuitError::ResourceExhaustion(_) => ErrorSeverity::Medium,
            CircuitError::Timeout(_) => ErrorSeverity::Low,
            CircuitError::ConcurrentAccess(_) => ErrorSeverity::Medium,
            CircuitError::InvalidProofFormat(_) => ErrorSeverity::Medium,
            CircuitError::UnsupportedOperation(_) => ErrorSeverity::Medium,
            CircuitError::SecurityViolation(_) => ErrorSeverity::Critical,
            CircuitError::ParameterMismatch(_) => ErrorSeverity::Medium,
            CircuitError::DegreeBoundExceeded { .. } => ErrorSeverity::Medium,
            CircuitError::InvalidFieldElement(_) => ErrorSeverity::Medium,
            CircuitError::ConstraintViolation(_) => ErrorSeverity::High,
            CircuitError::BatchProcessing(_) => ErrorSeverity::Medium,
            CircuitError::WorkerPool(_) => ErrorSeverity::Medium,
            CircuitError::Cache(_) => ErrorSeverity::Low,
            CircuitError::Metrics(_) => ErrorSeverity::Low,
            CircuitError::Internal(_) => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            CircuitError::CircuitNotFound(_) => "circuit_not_found",
            CircuitError::InvalidInput(_) => "invalid_input",
            CircuitError::TraceGeneration(_) => "trace_generation",
            CircuitError::ProofGeneration(_) => "proof_generation",
            CircuitError::Verification(_) => "verification",
            CircuitError::Optimization(_) => "optimization",
            CircuitError::PolynomialCommitment(_) => "polynomial_commitment",
            CircuitError::MerkleTree(_) => "merkle_tree",
            CircuitError::RecursiveComposition(_) => "recursive_composition",
            CircuitError::TrustedSetup(_) => "trusted_setup",
            CircuitError::FieldArithmetic(_) => "field_arithmetic",
            CircuitError::ConstraintSystem(_) => "constraint_system",
            CircuitError::Serialization(_) => "serialization",
            CircuitError::Io(_) => "io",
            CircuitError::Configuration(_) => "configuration",
            CircuitError::ResourceExhaustion(_) => "resource_exhaustion",
            CircuitError::Timeout(_) => "timeout",
            CircuitError::ConcurrentAccess(_) => "concurrent_access",
            CircuitError::InvalidProofFormat(_) => "invalid_proof_format",
            CircuitError::UnsupportedOperation(_) => "unsupported_operation",
            CircuitError::SecurityViolation(_) => "security_violation",
            CircuitError::ParameterMismatch(_) => "parameter_mismatch",
            CircuitError::DegreeBoundExceeded { .. } => "degree_bound_exceeded",
            CircuitError::InvalidFieldElement(_) => "invalid_field_element",
            CircuitError::ConstraintViolation(_) => "constraint_violation",
            CircuitError::BatchProcessing(_) => "batch_processing",
            CircuitError::WorkerPool(_) => "worker_pool",
            CircuitError::Cache(_) => "cache",
            CircuitError::Metrics(_) => "metrics",
            CircuitError::Internal(_) => "internal",
        }
    }
    
    /// Check if error should trigger security alert
    pub fn is_security_critical(&self) -> bool {
        matches!(
            self,
            CircuitError::SecurityViolation(_) |
            CircuitError::TrustedSetup(_) |
            CircuitError::Verification(_)
        )
    }
    
    /// Check if error affects proof validity
    pub fn affects_proof_validity(&self) -> bool {
        matches!(
            self,
            CircuitError::ProofGeneration(_) |
            CircuitError::Verification(_) |
            CircuitError::ConstraintViolation(_) |
            CircuitError::InvalidProofFormat(_) |
            CircuitError::FieldArithmetic(_) |
            CircuitError::ConstraintSystem(_)
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
