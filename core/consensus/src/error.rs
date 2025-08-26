//! Error types for the consensus module

use thiserror::Error;
use crate::ValidatorId;

/// Result type alias
pub type Result<T> = std::result::Result<T, ConsensusError>;

/// Consensus error types
#[derive(Error, Debug)]
pub enum ConsensusError {
    /// Invalid signature
    #[error("Invalid signature")]
    InvalidSignature,
    
    /// Missing signature
    #[error("Missing signature")]
    MissingSignature,
    
    /// Unknown validator
    #[error("Unknown validator: {0}")]
    UnknownValidator(ValidatorId),
    
    /// Insufficient stake
    #[error("Insufficient stake: required {required}, got {actual}")]
    InsufficientStake { required: u128, actual: u128 },
    
    /// Invalid proposal
    #[error("Invalid proposal: {0}")]
    InvalidProposal(String),
    
    /// Proposal timeout
    #[error("Proposal timeout: {0}")]
    ProposalTimeout(String),
    
    /// Duplicate vote
    #[error("Duplicate vote from validator: {0}")]
    DuplicateVote(ValidatorId),
    
    /// Invalid vote
    #[error("Invalid vote: {0}")]
    InvalidVote(String),
    
    /// Consensus not reached
    #[error("Consensus not reached")]
    ConsensusNotReached,
    
    /// Byzantine behavior detected
    #[error("Byzantine behavior detected from validator: {0}")]
    ByzantineBehavior(ValidatorId),
    
    /// Network error
    #[error("Network error: {0}")]
    Network(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    /// Redis error
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),
    
    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    Cryptographic(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// State machine error
    #[error("State machine error: {0}")]
    StateMachine(String),
    
    /// Timeout error
    #[error("Timeout error: {0}")]
    Timeout(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Validator set change error
    #[error("Validator set change error: {0}")]
    ValidatorSetChange(String),
    
    /// Slashing error
    #[error("Slashing error: {0}")]
    Slashing(String),
    
    /// Reputation error
    #[error("Reputation error: {0}")]
    Reputation(String),
    
    /// Cross-chain error
    #[error("Cross-chain error: {0}")]
    CrossChain(String),
    
    /// Proof verification error
    #[error("Proof verification error: {0}")]
    ProofVerification(String),
}

impl ConsensusError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            ConsensusError::Network(_) => true,
            ConsensusError::Timeout(_) => true,
            ConsensusError::Database(_) => true,
            ConsensusError::Redis(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ConsensusError::InvalidSignature => ErrorSeverity::High,
            ConsensusError::ByzantineBehavior(_) => ErrorSeverity::Critical,
            ConsensusError::UnknownValidator(_) => ErrorSeverity::Medium,
            ConsensusError::InsufficientStake { .. } => ErrorSeverity::Medium,
            ConsensusError::InvalidProposal(_) => ErrorSeverity::Medium,
            ConsensusError::ProposalTimeout(_) => ErrorSeverity::Low,
            ConsensusError::DuplicateVote(_) => ErrorSeverity::Medium,
            ConsensusError::InvalidVote(_) => ErrorSeverity::Medium,
            ConsensusError::ConsensusNotReached => ErrorSeverity::Medium,
            ConsensusError::Network(_) => ErrorSeverity::Low,
            ConsensusError::Serialization(_) => ErrorSeverity::Medium,
            ConsensusError::Database(_) => ErrorSeverity::High,
            ConsensusError::Redis(_) => ErrorSeverity::Medium,
            ConsensusError::Cryptographic(_) => ErrorSeverity::High,
            ConsensusError::Configuration(_) => ErrorSeverity::High,
            ConsensusError::StateMachine(_) => ErrorSeverity::High,
            ConsensusError::Timeout(_) => ErrorSeverity::Low,
            ConsensusError::Internal(_) => ErrorSeverity::High,
            ConsensusError::ValidatorSetChange(_) => ErrorSeverity::Medium,
            ConsensusError::Slashing(_) => ErrorSeverity::High,
            ConsensusError::Reputation(_) => ErrorSeverity::Low,
            ConsensusError::CrossChain(_) => ErrorSeverity::Medium,
            ConsensusError::ProofVerification(_) => ErrorSeverity::High,
            ConsensusError::MissingSignature => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            ConsensusError::InvalidSignature => "signature",
            ConsensusError::MissingSignature => "signature",
            ConsensusError::UnknownValidator(_) => "validator",
            ConsensusError::InsufficientStake { .. } => "stake",
            ConsensusError::InvalidProposal(_) => "proposal",
            ConsensusError::ProposalTimeout(_) => "timeout",
            ConsensusError::DuplicateVote(_) => "vote",
            ConsensusError::InvalidVote(_) => "vote",
            ConsensusError::ConsensusNotReached => "consensus",
            ConsensusError::ByzantineBehavior(_) => "byzantine",
            ConsensusError::Network(_) => "network",
            ConsensusError::Serialization(_) => "serialization",
            ConsensusError::Database(_) => "database",
            ConsensusError::Redis(_) => "redis",
            ConsensusError::Cryptographic(_) => "crypto",
            ConsensusError::Configuration(_) => "config",
            ConsensusError::StateMachine(_) => "state_machine",
            ConsensusError::Timeout(_) => "timeout",
            ConsensusError::Internal(_) => "internal",
            ConsensusError::ValidatorSetChange(_) => "validator_set",
            ConsensusError::Slashing(_) => "slashing",
            ConsensusError::Reputation(_) => "reputation",
            ConsensusError::CrossChain(_) => "cross_chain",
            ConsensusError::ProofVerification(_) => "proof",
        }
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
