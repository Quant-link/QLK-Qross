//! Core types for the validator selection module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Validator identifier
pub type ValidatorId = String;

/// Stake amount (in smallest unit)
pub type Stake = u128;

/// Validator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub id: ValidatorId,
    pub public_key: Vec<u8>,
    pub network_address: String,
    pub stake: Stake,
    pub is_active: bool,
    pub is_slashed: bool,
    pub joined_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub commission_rate: f64,
    pub self_stake: Stake,
    pub delegated_stake: Stake,
}

/// Validator candidate for selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorCandidate {
    pub validator_info: ValidatorInfo,
    pub stake: Stake,
    pub reputation_score: ReputationScore,
    pub performance_metrics: PerformanceMetrics,
    pub selection_score: f64,
    pub last_selected_epoch: Option<u64>,
    pub consecutive_selections: u32,
    pub eligibility_status: EligibilityStatus,
}

/// Selected validator for an epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedValidator {
    pub validator_id: ValidatorId,
    pub stake: Stake,
    pub reputation_score: ReputationScore,
    pub performance_metrics: PerformanceMetrics,
    pub selection_score: f64,
    pub selection_reason: SelectionReason,
}

/// Eligibility status for validator selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EligibilityStatus {
    Eligible,
    InsufficientStake,
    LowReputation,
    RecentlySlashed,
    InGracePeriod,
    MaxConsecutiveSelections,
    Inactive,
}

/// Reason for validator selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionReason {
    HighStake,
    HighReputation,
    HighPerformance,
    Diversity,
    Balanced,
}

/// Reputation score with breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    pub score: u8, // 0-100
    pub uptime_score: f64,
    pub performance_score: f64,
    pub slashing_penalty: f64,
    pub governance_participation: f64,
    pub last_updated: DateTime<Utc>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub overall_score: f64, // 0-1
    pub uptime_percentage: f64,
    pub block_production_rate: f64,
    pub attestation_rate: f64,
    pub response_time_ms: f64,
    pub missed_blocks: u64,
    pub missed_attestations: u64,
    pub last_calculated: DateTime<Utc>,
}

/// Performance record for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub epoch: u64,
    pub validator_id: ValidatorId,
    pub blocks_produced: u64,
    pub blocks_expected: u64,
    pub attestations_made: u64,
    pub attestations_expected: u64,
    pub uptime_seconds: u64,
    pub total_seconds: u64,
    pub average_response_time: f64,
    pub slashing_events: u32,
    pub timestamp: DateTime<Utc>,
}

/// Slashing record for reputation calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingRecord {
    pub validator_id: ValidatorId,
    pub slashing_type: SlashingType,
    pub amount_slashed: Stake,
    pub reason: String,
    pub epoch: u64,
    pub timestamp: DateTime<Utc>,
}

/// Types of slashing events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlashingType {
    Light,
    Medium,
    Severe,
}

/// Selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Minimum number of validators to select
    pub min_validators: usize,
    
    /// Maximum number of validators to select
    pub max_validators: usize,
    
    /// Minimum stake required for selection
    pub min_stake: Stake,
    
    /// Maximum stake for normalization
    pub max_stake: Stake,
    
    /// Minimum reputation score required
    pub min_reputation: u8,
    
    /// Minimum average reputation for selected set
    pub min_average_reputation: u8,
    
    /// Maximum stake concentration percentage
    pub max_stake_concentration: u8,
    
    /// Maximum consecutive selections allowed
    pub max_consecutive_selections: u32,
    
    /// Grace period for new validators (seconds)
    pub grace_period_seconds: u64,
    
    /// Weight for stake in selection score
    pub stake_weight: f64,
    
    /// Weight for reputation in selection score
    pub reputation_weight: f64,
    
    /// Weight for performance in selection score
    pub performance_weight: f64,
    
    /// Reputation calculation configuration
    pub reputation_config: ReputationConfig,
    
    /// Performance analysis configuration
    pub performance_config: PerformanceConfig,
}

/// Reputation calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Base reputation score for new validators
    pub base_score: u8,
    
    /// Maximum reputation score
    pub max_score: u8,
    
    /// Uptime weight in reputation calculation
    pub uptime_weight: f64,
    
    /// Performance weight in reputation calculation
    pub performance_weight: f64,
    
    /// Slashing penalty weight
    pub slashing_penalty_weight: f64,
    
    /// Governance participation weight
    pub governance_weight: f64,
    
    /// Reputation decay rate per epoch
    pub decay_rate: f64,
    
    /// Recovery rate after slashing
    pub recovery_rate: f64,
    
    /// Minimum epochs for reputation recovery
    pub min_recovery_epochs: u64,
}

/// Performance analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of epochs to analyze for performance
    pub history_epochs: u64,
    
    /// Weight for uptime in performance score
    pub uptime_weight: f64,
    
    /// Weight for block production in performance score
    pub block_production_weight: f64,
    
    /// Weight for attestation rate in performance score
    pub attestation_weight: f64,
    
    /// Weight for response time in performance score
    pub response_time_weight: f64,
    
    /// Maximum acceptable response time (ms)
    pub max_response_time: f64,
    
    /// Minimum uptime percentage for good performance
    pub min_uptime_percentage: f64,
    
    /// Performance decay rate for old data
    pub performance_decay_rate: f64,
}

/// Selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub epoch: u64,
    pub selected_validators: Vec<SelectedValidator>,
    pub total_candidates: usize,
    pub filtered_candidates: usize,
    pub selection_strategy: String,
    pub selection_time: std::time::Duration,
    pub timestamp: DateTime<Utc>,
}

/// Selection event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionEvent {
    pub id: Uuid,
    pub epoch: u64,
    pub selected_count: usize,
    pub total_candidates: usize,
    pub strategy_used: String,
    pub average_reputation: f64,
    pub average_performance: f64,
    pub total_stake: Stake,
    pub timestamp: DateTime<Utc>,
}

/// Selection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStatistics {
    pub total_selections: usize,
    pub average_selection_time: f64,
    pub average_validators_selected: f64,
    pub selection_strategy: String,
    pub last_selection: Option<DateTime<Utc>>,
}

/// Performance data update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub epoch: u64,
    pub blocks_produced: u64,
    pub blocks_expected: u64,
    pub attestations_made: u64,
    pub attestations_expected: u64,
    pub uptime_seconds: u64,
    pub response_times: Vec<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Reputation update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationUpdate {
    pub update_type: ReputationUpdateType,
    pub value: f64,
    pub reason: String,
    pub epoch: u64,
    pub timestamp: DateTime<Utc>,
}

/// Types of reputation updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReputationUpdateType {
    PerformanceBonus,
    UptimeBonus,
    GovernanceParticipation,
    SlashingPenalty,
    MissedBlockPenalty,
    MissedAttestationPenalty,
    Recovery,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            min_validators: 4,
            max_validators: 100,
            min_stake: 1_000_000_000_000_000_000_000, // 1000 tokens
            max_stake: 1_000_000_000_000_000_000_000_000, // 1M tokens
            min_reputation: 50,
            min_average_reputation: 60,
            max_stake_concentration: 33, // 33% max
            max_consecutive_selections: 5,
            grace_period_seconds: 86400 * 7, // 1 week
            stake_weight: 0.4,
            reputation_weight: 0.4,
            performance_weight: 0.2,
            reputation_config: ReputationConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            base_score: 50,
            max_score: 100,
            uptime_weight: 0.3,
            performance_weight: 0.3,
            slashing_penalty_weight: 0.3,
            governance_weight: 0.1,
            decay_rate: 0.01,
            recovery_rate: 0.05,
            min_recovery_epochs: 10,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            history_epochs: 10,
            uptime_weight: 0.3,
            block_production_weight: 0.3,
            attestation_weight: 0.3,
            response_time_weight: 0.1,
            max_response_time: 1000.0, // 1 second
            min_uptime_percentage: 95.0,
            performance_decay_rate: 0.05,
        }
    }
}
