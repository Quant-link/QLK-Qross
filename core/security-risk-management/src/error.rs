//! Error types for security and risk management

use thiserror::Error;
use qross_consensus::ValidatorId;

/// Security error types
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Invalid threshold: {0}")]
    InvalidThreshold(String),
    
    #[error("Scheme not found: {0:?}")]
    SchemeNotFound(crate::threshold_signatures::SchemeId),
    
    #[error("Insufficient signers: {0}")]
    InsufficientSigners(String),
    
    #[error("Invalid signer: {0:?}")]
    InvalidSigner(ValidatorId),
    
    #[error("No signatures to aggregate")]
    NoSignaturesToAggregate,
    
    #[error("Key generation error: {0}")]
    KeyGenerationError(String),
    
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    
    #[error("Ceremony coordination failed: {0}")]
    CeremonyCoordinationFailed(String),
    
    #[error("Random beacon generation failed: {0}")]
    RandomBeaconGenerationFailed(String),
    
    #[error("Participant coordination failed: {0}")]
    ParticipantCoordinationFailed(String),
    
    #[error("Key refresh failed: {0}")]
    KeyRefreshFailed(String),
    
    #[error("Governance proposal invalid: {0}")]
    InvalidGovernanceProposal(String),
    
    #[error("Voting period expired")]
    VotingPeriodExpired,
    
    #[error("Insufficient votes: {0}")]
    InsufficientVotes(String),
    
    #[error("Emergency pause active")]
    EmergencyPauseActive,
    
    #[error("Emergency threshold not met: {0}")]
    EmergencyThresholdNotMet(String),
    
    #[error("Formal verification failed: {0}")]
    FormalVerificationFailed(String),
    
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),
    
    #[error("Security monitoring error: {0}")]
    SecurityMonitoringError(String),
    
    #[error("Cryptographic error: {0}")]
    CryptographicError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Timeout error: {0}")]
    TimeoutError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for security operations
pub type Result<T> = std::result::Result<T, SecurityError>;

impl From<serde_json::Error> for SecurityError {
    fn from(err: serde_json::Error) -> Self {
        SecurityError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for SecurityError {
    fn from(err: std::io::Error) -> Self {
        SecurityError::StorageError(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for SecurityError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        SecurityError::TimeoutError(err.to_string())
    }
}

/// Security warning types
#[derive(Debug, Clone)]
pub enum SecurityWarning {
    ThresholdApproaching {
        current: u32,
        threshold: u32,
        message: String,
    },
    KeyRefreshDue {
        scheme_id: crate::threshold_signatures::SchemeId,
        due_date: chrono::DateTime<chrono::Utc>,
    },
    UnusualActivity {
        validator_id: ValidatorId,
        activity_type: String,
        severity: WarningSeverity,
    },
    SystemPerformance {
        metric: String,
        current_value: f64,
        threshold_value: f64,
    },
    SecurityVulnerability {
        vulnerability_type: String,
        severity: WarningSeverity,
        description: String,
    },
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Security alert types
#[derive(Debug, Clone)]
pub struct SecurityAlert {
    pub alert_id: uuid::Uuid,
    pub alert_type: SecurityAlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub affected_components: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub acknowledged: bool,
    pub resolved: bool,
}

/// Security alert types
#[derive(Debug, Clone)]
pub enum SecurityAlertType {
    ThresholdBreach,
    KeyCompromise,
    UnauthorizedAccess,
    SystemFailure,
    NetworkAttack,
    GovernanceViolation,
    EmergencyActivation,
    VerificationFailure,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Emergency,
}

impl SecurityAlert {
    pub fn new(
        alert_type: SecurityAlertType,
        severity: AlertSeverity,
        message: String,
    ) -> Self {
        Self {
            alert_id: uuid::Uuid::new_v4(),
            alert_type,
            severity,
            message,
            affected_components: Vec::new(),
            recommended_actions: Vec::new(),
            created_at: chrono::Utc::now(),
            acknowledged: false,
            resolved: false,
        }
    }
    
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }
    
    pub fn resolve(&mut self) {
        self.resolved = true;
    }
    
    pub fn add_affected_component(&mut self, component: String) {
        self.affected_components.push(component);
    }
    
    pub fn add_recommended_action(&mut self, action: String) {
        self.recommended_actions.push(action);
    }
}
