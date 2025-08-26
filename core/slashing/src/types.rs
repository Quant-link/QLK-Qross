//! Core types for the slashing module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Validator identifier
pub type ValidatorId = String;

/// Evidence identifier
pub type EvidenceId = Uuid;

/// Investigation identifier
pub type InvestigationId = Uuid;

/// Stake amount (in smallest unit)
pub type Stake = u128;

/// Misbehavior types that can trigger slashing
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MisbehaviorType {
    /// Incorrect attestation (light slashing - 5%)
    IncorrectAttestation,
    
    /// Conflicting votes in the same round (medium slashing - 15%)
    ConflictingVotes,
    
    /// Double signing (severe slashing - 50%)
    DoubleSigning,
    
    /// Validator unavailability (reputation penalty only)
    Unavailability,
    
    /// Invalid proposal submission (medium slashing - 15%)
    InvalidProposal,
}

/// Evidence of validator misbehavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: EvidenceId,
    pub validator_id: ValidatorId,
    pub misbehavior_type: MisbehaviorType,
    pub evidence_data: Vec<u8>,
    pub reporter: ValidatorId,
    pub timestamp: DateTime<Utc>,
    pub block_height: u64,
    pub verified: bool,
}

/// Investigation into validator misbehavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Investigation {
    pub id: InvestigationId,
    pub validator_id: ValidatorId,
    pub misbehavior_type: MisbehaviorType,
    pub evidence: Vec<Evidence>,
    pub status: InvestigationStatus,
    pub started_at: DateTime<Utc>,
    pub deadline: DateTime<Utc>,
    pub locked_stake: Stake,
}

/// Investigation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvestigationStatus {
    Pending,
    Proven,
    Dismissed,
    TimedOut,
}

/// Penalty types corresponding to slashing severity
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PenaltyType {
    Light,   // 5% stake
    Medium,  // 15% stake
    Severe,  // 50% stake
    None,    // No slashing, reputation penalty only
}

impl std::fmt::Display for PenaltyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenaltyType::Light => write!(f, "light"),
            PenaltyType::Medium => write!(f, "medium"),
            PenaltyType::Severe => write!(f, "severe"),
            PenaltyType::None => write!(f, "none"),
        }
    }
}

/// Calculated penalty for misbehavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Penalty {
    pub penalty_type: PenaltyType,
    pub amount: Stake,
    pub reason: String,
    pub evidence_count: usize,
    pub calculated_at: DateTime<Utc>,
}

/// Slashing event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    pub id: Uuid,
    pub validator_id: ValidatorId,
    pub misbehavior_type: MisbehaviorType,
    pub penalty_type: PenaltyType,
    pub slashed_amount: Stake,
    pub remaining_stake: Stake,
    pub timestamp: DateTime<Utc>,
    pub evidence_count: usize,
    pub redistribution_recipients: usize,
}

/// Slashing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingConfig {
    /// Light slashing percentage for incorrect attestation
    pub light_slashing_percentage: u8,
    
    /// Medium slashing percentage for conflicting votes
    pub medium_slashing_percentage: u8,
    
    /// Severe slashing percentage for double signing
    pub severe_slashing_percentage: u8,
    
    /// Minimum evidence age in blocks
    pub min_evidence_age: u64,
    
    /// Maximum evidence age in blocks
    pub max_evidence_age: u64,
    
    /// Investigation timeout in seconds
    pub investigation_timeout: u64,
    
    /// Minimum stake required for slashing
    pub min_slashable_stake: Stake,
    
    /// Maximum slashing per validator per epoch
    pub max_slashing_per_epoch: Stake,
    
    /// Cooldown period between slashing events (seconds)
    pub slashing_cooldown: u64,
    
    /// Percentage of slashed stake to redistribute to honest validators
    pub redistribution_percentage: u8,
    
    /// Percentage of slashed stake to burn
    pub burn_percentage: u8,
    
    /// Grace period for new validators (blocks)
    pub new_validator_grace_period: u64,
}

impl Default for SlashingConfig {
    fn default() -> Self {
        Self {
            light_slashing_percentage: 5,
            medium_slashing_percentage: 15,
            severe_slashing_percentage: 50,
            min_evidence_age: 1,
            max_evidence_age: 10000,
            investigation_timeout: 3600, // 1 hour
            min_slashable_stake: 1_000_000_000_000_000_000, // 1 token
            max_slashing_per_epoch: 1_000_000_000_000_000_000_000, // 1000 tokens
            slashing_cooldown: 86400, // 24 hours
            redistribution_percentage: 80,
            burn_percentage: 20,
            new_validator_grace_period: 1000, // 1000 blocks
        }
    }
}

/// Slashing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingStatistics {
    pub total_investigations: u64,
    pub active_investigations: usize,
    pub total_slashing_events: usize,
    pub total_slashed_amount: Stake,
    pub penalty_type_counts: HashMap<PenaltyType, usize>,
    pub average_investigation_time: f64,
}

/// Validator slashing history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSlashingHistory {
    pub validator_id: ValidatorId,
    pub total_slashed: Stake,
    pub slashing_events: Vec<SlashingEvent>,
    pub current_investigations: Vec<InvestigationId>,
    pub reputation_impact: f64,
    pub last_slashed_at: Option<DateTime<Utc>>,
}

/// Evidence submission request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSubmission {
    pub validator_id: ValidatorId,
    pub misbehavior_type: MisbehaviorType,
    pub evidence_data: Vec<u8>,
    pub reporter: ValidatorId,
    pub block_height: u64,
    pub signature: Vec<u8>,
}

/// Slashing prevention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingPrevention {
    /// Maximum percentage of total stake that can be slashed in one epoch
    pub max_total_slashing_per_epoch: u8,
    
    /// Minimum number of validators that must remain active
    pub min_active_validators: usize,
    
    /// Emergency halt threshold (percentage of validators slashed)
    pub emergency_halt_threshold: u8,
    
    /// Network partition detection enabled
    pub partition_detection_enabled: bool,
    
    /// Minimum network connectivity percentage
    pub min_network_connectivity: u8,
}

impl Default for SlashingPrevention {
    fn default() -> Self {
        Self {
            max_total_slashing_per_epoch: 33, // Max 33% of total stake
            min_active_validators: 4, // Minimum for 2f+1 consensus
            emergency_halt_threshold: 25, // Halt if 25% of validators slashed
            partition_detection_enabled: true,
            min_network_connectivity: 67, // 67% connectivity required
        }
    }
}

/// Redistribution target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedistributionTarget {
    pub validator_id: ValidatorId,
    pub amount: Stake,
    pub reason: RedistributionReason,
}

/// Reason for stake redistribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedistributionReason {
    HonestValidator,
    ReporterReward,
    CommunityPool,
    Burn,
}

/// Slashing appeal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingAppeal {
    pub id: Uuid,
    pub validator_id: ValidatorId,
    pub slashing_event_id: Uuid,
    pub appeal_reason: String,
    pub evidence: Vec<u8>,
    pub submitted_at: DateTime<Utc>,
    pub status: AppealStatus,
    pub reviewed_by: Option<ValidatorId>,
    pub reviewed_at: Option<DateTime<Utc>>,
}

/// Appeal status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppealStatus {
    Pending,
    Approved,
    Rejected,
    UnderReview,
}

/// Network health metrics for slashing prevention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    pub total_validators: usize,
    pub active_validators: usize,
    pub slashed_validators: usize,
    pub network_connectivity: f64,
    pub consensus_participation: f64,
    pub average_block_time: f64,
    pub is_partitioned: bool,
    pub emergency_halt_active: bool,
}
