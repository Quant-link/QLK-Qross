//! Type definitions for security and risk management

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use qross_consensus::ValidatorId;
use bls12_381::Scalar;

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub threshold_config: ThresholdConfig,
    pub governance_config: GovernanceConfig,
    pub emergency_config: EmergencyConfig,
    pub verification_config: VerificationConfig,
    pub monitoring_config: MonitoringConfig,
}

/// Threshold signature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub default_threshold_ratio: f64,
    pub min_participants: u32,
    pub max_participants: u32,
    pub key_refresh_interval: chrono::Duration,
    pub signature_cache_size: usize,
    pub security_level: String,
}

/// Governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceConfig {
    pub proposal_threshold: u32,
    pub voting_period: chrono::Duration,
    pub execution_delay: chrono::Duration,
    pub quorum_requirement: f64,
    pub super_majority_threshold: f64,
}

/// Emergency configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyConfig {
    pub emergency_threshold: u32,
    pub pause_duration: chrono::Duration,
    pub recovery_threshold: u32,
    pub emergency_contacts: Vec<String>,
}

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub enable_formal_verification: bool,
    pub verification_timeout: chrono::Duration,
    pub proof_cache_size: usize,
    pub verification_parallelism: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub monitoring_interval: chrono::Duration,
    pub alert_thresholds: HashMap<String, f64>,
    pub log_retention_period: chrono::Duration,
    pub metrics_collection_enabled: bool,
}

/// Ceremony result from key generation
#[derive(Debug, Clone)]
pub struct CeremonyResult {
    pub scheme_id: crate::threshold_signatures::SchemeId,
    pub master_secret: Scalar,
    pub public_key: crate::threshold_signatures::PublicKey,
    pub random_beacon: RandomBeacon,
    pub transcript: CeremonyTranscript,
    pub participants: HashSet<ValidatorId>,
}

/// Random beacon for ceremonies
#[derive(Debug, Clone)]
pub struct RandomBeacon {
    pub beacon_id: uuid::Uuid,
    pub beacon_data: Vec<u8>,
    pub round_number: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub contributors: HashSet<ValidatorId>,
}

/// Ceremony transcript for verification
#[derive(Debug, Clone)]
pub struct CeremonyTranscript {
    pub transcript_id: uuid::Uuid,
    pub ceremony_type: String,
    pub participants: HashSet<ValidatorId>,
    pub commitments: Vec<ParticipantCommitment>,
    pub random_beacon: RandomBeacon,
    pub verification_data: Vec<u8>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Participant commitment in ceremony
#[derive(Debug, Clone)]
pub struct ParticipantCommitment {
    pub validator_id: ValidatorId,
    pub commitment_data: Vec<u8>,
    pub proof_data: Vec<u8>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Key refresh result
#[derive(Debug, Clone)]
pub struct RefreshResult {
    pub ceremony_result: CeremonyResult,
    pub refresh_proof: Vec<u8>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            threshold_config: ThresholdConfig::default(),
            governance_config: GovernanceConfig::default(),
            emergency_config: EmergencyConfig::default(),
            verification_config: VerificationConfig::default(),
            monitoring_config: MonitoringConfig::default(),
        }
    }
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            default_threshold_ratio: 0.67, // 2/3 threshold
            min_participants: 3,
            max_participants: 100,
            key_refresh_interval: chrono::Duration::days(30),
            signature_cache_size: 10000,
            security_level: "standard".to_string(),
        }
    }
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            proposal_threshold: 5,
            voting_period: chrono::Duration::days(7),
            execution_delay: chrono::Duration::days(2),
            quorum_requirement: 0.5, // 50%
            super_majority_threshold: 0.67, // 67%
        }
    }
}

impl Default for EmergencyConfig {
    fn default() -> Self {
        Self {
            emergency_threshold: 3,
            pause_duration: chrono::Duration::hours(24),
            recovery_threshold: 5,
            emergency_contacts: Vec::new(),
        }
    }
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            enable_formal_verification: true,
            verification_timeout: chrono::Duration::minutes(30),
            proof_cache_size: 1000,
            verification_parallelism: 4,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: chrono::Duration::seconds(30),
            alert_thresholds: HashMap::new(),
            log_retention_period: chrono::Duration::days(90),
            metrics_collection_enabled: true,
        }
    }
}
