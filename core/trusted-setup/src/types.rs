//! Core types for distributed trusted setup ceremony

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use qross_consensus::ValidatorId;

/// Ceremony identifier
pub type CeremonyId = Uuid;

/// Parameter set identifier
pub type ParameterSetId = Uuid;

/// Round number in ceremony
pub type RoundNumber = usize;

/// Types of trusted setup ceremonies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CeremonyType {
    /// Universal ceremony for general circuits
    Universal,
    /// Circuit-specific ceremony
    CircuitSpecific { circuit_id: u64 },
    /// Update ceremony based on previous parameters
    Update { previous_ceremony: CeremonyId },
}

/// Ceremony session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonySession {
    pub id: CeremonyId,
    pub ceremony_type: CeremonyType,
    pub participants: Vec<ValidatorId>,
    pub state: CeremonyState,
    pub current_round: RoundNumber,
    pub total_rounds: usize,
    pub random_beacon: RandomBeacon,
    pub parameters: Option<VerifiedParameters>,
    pub created_at: DateTime<Utc>,
    pub timeout_at: DateTime<Utc>,
    pub contributions: HashMap<RoundNumber, HashMap<ValidatorId, ValidatorContribution>>,
    pub verification_results: HashMap<RoundNumber, HashMap<ValidatorId, VerificationResult>>,
}

/// Ceremony execution state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CeremonyState {
    Initializing,
    Running,
    Completed,
    Failed,
    Halted,
    Restarting,
}

/// Random beacon for ceremony entropy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomBeacon {
    pub beacon_id: Uuid,
    pub round: RoundNumber,
    pub entropy: Vec<u8>,
    pub contributors: Vec<ValidatorId>,
    pub signatures: HashMap<ValidatorId, BeaconSignature>,
    pub generated_at: DateTime<Utc>,
}

/// Beacon signature from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconSignature {
    pub validator_id: ValidatorId,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub signed_at: DateTime<Utc>,
}

/// Validator information for ceremony participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub validator_id: ValidatorId,
    pub reputation_score: f64,
    pub performance_metrics: ValidatorPerformanceMetrics,
    pub ceremony_history: CeremonyHistory,
    pub public_key: Vec<u8>,
    pub network_address: String,
}

/// Validator performance metrics for ceremony selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformanceMetrics {
    pub uptime_percentage: f64,
    pub response_time_ms: f64,
    pub computation_score: f64,
    pub network_reliability: f64,
    pub ceremony_participation_rate: f64,
}

/// Validator ceremony participation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyHistory {
    pub total_ceremonies: usize,
    pub successful_ceremonies: usize,
    pub failed_ceremonies: usize,
    pub average_contribution_time: f64,
    pub last_participation: Option<DateTime<Utc>>,
}

/// Validator contribution to ceremony round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorContribution {
    pub ceremony_id: CeremonyId,
    pub round: RoundNumber,
    pub validator_id: ValidatorId,
    pub contribution_data: ContributionData,
    pub proof_of_computation: ProofOfComputation,
    pub signature: Vec<u8>,
    pub submitted_at: DateTime<Utc>,
}

/// Contribution data containing cryptographic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionData {
    pub tau_powers: Vec<Vec<u8>>, // Powers of tau
    pub alpha_tau_powers: Vec<Vec<u8>>, // Alpha times powers of tau
    pub beta_tau_powers: Vec<Vec<u8>>, // Beta times powers of tau
    pub gamma_inverse: Vec<u8>, // Inverse of gamma
    pub delta_inverse: Vec<u8>, // Inverse of delta
    pub entropy_contribution: Vec<u8>, // Additional entropy
}

/// Proof that contribution was computed correctly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfComputation {
    pub computation_proof: Vec<u8>,
    pub verification_key: Vec<u8>,
    pub computation_time: std::time::Duration,
    pub memory_usage: usize,
}

/// Request for validator contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionRequest {
    pub ceremony_id: CeremonyId,
    pub round: RoundNumber,
    pub validator_id: ValidatorId,
    pub beacon: RandomBeacon,
    pub deadline: DateTime<Utc>,
}

/// Result of contribution verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub verification_time: std::time::Duration,
    pub failure_reason: Option<String>,
    pub verified_at: DateTime<Utc>,
    pub verifier_id: String,
}

/// Final verified parameters from ceremony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedParameters {
    pub id: ParameterSetId,
    pub ceremony_id: CeremonyId,
    pub parameters: CeremonyParameters,
    pub verification: FinalVerification,
    pub participants: Vec<ValidatorId>,
    pub created_at: DateTime<Utc>,
}

/// Final ceremony parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyParameters {
    pub parameter_type: CeremonyType,
    pub tau_powers: Vec<Vec<u8>>,
    pub alpha_tau_powers: Vec<Vec<u8>>,
    pub beta_tau_powers: Vec<Vec<u8>>,
    pub gamma_inverse: Vec<u8>,
    pub delta_inverse: Vec<u8>,
    pub verification_key: Vec<u8>,
    pub parameter_hash: Vec<u8>,
}

/// Final verification of ceremony parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalVerification {
    pub is_valid: bool,
    pub verification_proofs: Vec<VerificationProof>,
    pub parameter_integrity_check: bool,
    pub contribution_chain_valid: bool,
    pub randomness_quality_score: f64,
    pub verified_at: DateTime<Utc>,
}

/// Individual verification proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProof {
    pub proof_type: VerificationProofType,
    pub proof_data: Vec<u8>,
    pub verifier_signature: Vec<u8>,
}

/// Types of verification proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationProofType {
    ParameterConsistency,
    ContributionChain,
    RandomnessQuality,
    ComputationCorrectness,
}

/// Ceremony violation for slashing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CeremonyViolation {
    InvalidContribution { round: RoundNumber, reason: String },
    MissedContribution { round: RoundNumber },
    InvalidSignature { round: RoundNumber },
    ComputationFraud { round: RoundNumber, proof: Vec<u8> },
    NetworkDisruption { duration: std::time::Duration },
}

/// Ceremony configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyConfig {
    pub ceremony_timeout: u64,
    pub round_timeout: u64,
    pub contribution_timeout: u64,
    pub min_participants: usize,
    pub max_participants: usize,
    pub universal_ceremony_rounds: usize,
    pub circuit_specific_rounds: usize,
    pub update_ceremony_rounds: usize,
    pub restart_attempts: usize,
    pub emergency_halt_integration: bool,
    pub coordinator_config: CoordinatorConfig,
    pub beacon_config: BeaconConfig,
    pub parameter_config: ParameterConfig,
    pub verification_config: VerificationConfig,
    pub restart_config: RestartConfig,
}

/// Coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub validator_selection_strategy: ValidatorSelectionStrategy,
    pub reputation_threshold: f64,
    pub performance_threshold: f64,
    pub geographic_distribution: bool,
    pub max_validator_reuse: usize,
}

/// Validator selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorSelectionStrategy {
    ReputationBased,
    PerformanceBased,
    Random,
    Hybrid,
}

/// Random beacon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconConfig {
    pub entropy_sources: Vec<EntropySource>,
    pub signature_threshold: usize,
    pub beacon_timeout: u64,
    pub quality_threshold: f64,
}

/// Sources of entropy for beacon generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropySource {
    ValidatorSignatures,
    BlockHashes,
    NetworkTimestamps,
    ExternalRandomness,
}

/// Parameter generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConfig {
    pub tau_power_count: usize,
    pub security_level: SecurityLevel,
    pub computation_timeout: u64,
    pub memory_limit: usize,
    pub parallel_computation: bool,
}

/// Security levels for parameter generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Standard,
    High,
    Maximum,
}

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub verification_timeout: u64,
    pub parallel_verification: bool,
    pub redundant_verification: bool,
    pub quality_checks: Vec<QualityCheck>,
}

/// Quality checks for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityCheck {
    ParameterConsistency,
    RandomnessQuality,
    ComputationCorrectness,
    ContributionChain,
}

/// Restart mechanism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartConfig {
    pub max_restart_attempts: usize,
    pub restart_delay: u64,
    pub participant_replacement_rate: f64,
    pub consensus_threshold: f64,
}

/// Ceremony statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyStatistics {
    pub active_ceremonies: usize,
    pub cached_parameters: usize,
    pub total_ceremonies: u64,
    pub success_rate: f64,
    pub average_ceremony_duration: f64,
}

impl Default for CeremonyConfig {
    fn default() -> Self {
        Self {
            ceremony_timeout: 3600, // 1 hour
            round_timeout: 300, // 5 minutes
            contribution_timeout: 120, // 2 minutes
            min_participants: 7, // Minimum for Byzantine fault tolerance
            max_participants: 21, // Maximum validator set size
            universal_ceremony_rounds: 5,
            circuit_specific_rounds: 3,
            update_ceremony_rounds: 2,
            restart_attempts: 3,
            emergency_halt_integration: true,
            coordinator_config: CoordinatorConfig::default(),
            beacon_config: BeaconConfig::default(),
            parameter_config: ParameterConfig::default(),
            verification_config: VerificationConfig::default(),
            restart_config: RestartConfig::default(),
        }
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            validator_selection_strategy: ValidatorSelectionStrategy::Hybrid,
            reputation_threshold: 0.7,
            performance_threshold: 0.8,
            geographic_distribution: true,
            max_validator_reuse: 3,
        }
    }
}

impl Default for BeaconConfig {
    fn default() -> Self {
        Self {
            entropy_sources: vec![
                EntropySource::ValidatorSignatures,
                EntropySource::BlockHashes,
                EntropySource::NetworkTimestamps,
            ],
            signature_threshold: 14, // 2/3 of 21 validators
            beacon_timeout: 60,
            quality_threshold: 0.9,
        }
    }
}

impl Default for ParameterConfig {
    fn default() -> Self {
        Self {
            tau_power_count: 1048576, // 2^20 for production security
            security_level: SecurityLevel::High,
            computation_timeout: 600, // 10 minutes
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            parallel_computation: true,
        }
    }
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            verification_timeout: 300, // 5 minutes
            parallel_verification: true,
            redundant_verification: true,
            quality_checks: vec![
                QualityCheck::ParameterConsistency,
                QualityCheck::RandomnessQuality,
                QualityCheck::ComputationCorrectness,
                QualityCheck::ContributionChain,
            ],
        }
    }
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restart_attempts: 3,
            restart_delay: 300, // 5 minutes
            participant_replacement_rate: 0.3, // Replace 30% of participants
            consensus_threshold: 0.67, // 2/3 consensus for restart
        }
    }
}

/// Ceremony statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyStatistics {
    pub active_ceremonies: usize,
    pub cached_parameters: usize,
    pub total_ceremonies: u64,
    pub success_rate: f64,
    pub average_ceremony_duration: f64,
}

/// Network partition information affecting ceremony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartition {
    pub partition_id: Uuid,
    pub affected_validators: Vec<ValidatorId>,
    pub detected_at: DateTime<Utc>,
    pub estimated_duration: Option<std::time::Duration>,
    pub impact_severity: PartitionSeverity,
}

/// Severity of network partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionSeverity {
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Emergency halt information for ceremony coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyHalt {
    pub halt_id: Uuid,
    pub reason: String,
    pub initiated_by: ValidatorId,
    pub initiated_at: DateTime<Utc>,
    pub affected_ceremonies: Vec<CeremonyId>,
    pub recovery_plan: Option<String>,
}

/// Ceremony restart information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyRestart {
    pub restart_id: Uuid,
    pub original_ceremony: CeremonyId,
    pub restart_reason: RestartReason,
    pub new_participants: Vec<ValidatorId>,
    pub retained_participants: Vec<ValidatorId>,
    pub restart_round: RoundNumber,
    pub initiated_at: DateTime<Utc>,
}

/// Reasons for ceremony restart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartReason {
    Timeout,
    InsufficientParticipation,
    VerificationFailure,
    NetworkPartition,
    ValidatorFailure,
    EmergencyHalt,
}

impl Default for CeremonyConfig {
    fn default() -> Self {
        Self {
            ceremony_timeout: 3600, // 1 hour
            round_timeout: 300, // 5 minutes
            contribution_timeout: 120, // 2 minutes
            min_participants: 7, // Minimum for Byzantine fault tolerance
            max_participants: 21, // Maximum validator set size
            universal_ceremony_rounds: 5,
            circuit_specific_rounds: 3,
            update_ceremony_rounds: 2,
            restart_attempts: 3,
            emergency_halt_integration: true,
            coordinator_config: CoordinatorConfig::default(),
            beacon_config: BeaconConfig::default(),
            parameter_config: ParameterConfig::default(),
            verification_config: VerificationConfig::default(),
            restart_config: RestartConfig::default(),
        }
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            validator_selection_strategy: ValidatorSelectionStrategy::Hybrid,
            reputation_threshold: 0.7,
            performance_threshold: 0.8,
            geographic_distribution: true,
            max_validator_reuse: 3,
        }
    }
}

impl Default for BeaconConfig {
    fn default() -> Self {
        Self {
            entropy_sources: vec![
                EntropySource::ValidatorSignatures,
                EntropySource::BlockHashes,
                EntropySource::NetworkTimestamps,
            ],
            signature_threshold: 14, // 2/3 of 21 validators
            beacon_timeout: 60,
            quality_threshold: 0.9,
        }
    }
}

impl Default for ParameterConfig {
    fn default() -> Self {
        Self {
            tau_power_count: 1048576, // 2^20 for production security
            security_level: SecurityLevel::High,
            computation_timeout: 600, // 10 minutes
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            parallel_computation: true,
        }
    }
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            verification_timeout: 300, // 5 minutes
            parallel_verification: true,
            redundant_verification: true,
            quality_checks: vec![
                QualityCheck::ParameterConsistency,
                QualityCheck::RandomnessQuality,
                QualityCheck::ComputationCorrectness,
                QualityCheck::ContributionChain,
            ],
        }
    }
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restart_attempts: 3,
            restart_delay: 300, // 5 minutes
            participant_replacement_rate: 0.3, // Replace 30% of participants
            consensus_threshold: 0.67, // 2/3 consensus for restart
        }
    }
}

/// Types of trusted setup ceremonies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CeremonyType {
    /// Universal ceremony for general circuits
    Universal,
    /// Circuit-specific ceremony
    CircuitSpecific { circuit_id: u64 },
    /// Update ceremony based on previous parameters
    Update { previous_ceremony: CeremonyId },
}

/// Ceremony session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonySession {
    pub id: CeremonyId,
    pub ceremony_type: CeremonyType,
    pub participants: Vec<ValidatorId>,
    pub state: CeremonyState,
    pub current_round: RoundNumber,
    pub total_rounds: usize,
    pub random_beacon: RandomBeacon,
    pub parameters: Option<VerifiedParameters>,
    pub created_at: DateTime<Utc>,
    pub timeout_at: DateTime<Utc>,
    pub contributions: HashMap<RoundNumber, HashMap<ValidatorId, ValidatorContribution>>,
    pub verification_results: HashMap<RoundNumber, HashMap<ValidatorId, VerificationResult>>,
}

/// Ceremony execution state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CeremonyState {
    Initializing,
    Running,
    Completed,
    Failed,
    Halted,
    Restarting,
}

/// Random beacon for ceremony entropy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomBeacon {
    pub beacon_id: Uuid,
    pub round: RoundNumber,
    pub entropy: Vec<u8>,
    pub contributors: Vec<ValidatorId>,
    pub signatures: HashMap<ValidatorId, BeaconSignature>,
    pub generated_at: DateTime<Utc>,
}

/// Beacon signature from validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconSignature {
    pub validator_id: ValidatorId,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub signed_at: DateTime<Utc>,
}

/// Validator information for ceremony participation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub validator_id: ValidatorId,
    pub reputation_score: f64,
    pub performance_metrics: ValidatorPerformanceMetrics,
    pub ceremony_history: CeremonyHistory,
    pub public_key: Vec<u8>,
    pub network_address: String,
}

/// Validator performance metrics for ceremony selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformanceMetrics {
    pub uptime_percentage: f64,
    pub response_time_ms: f64,
    pub computation_score: f64,
    pub network_reliability: f64,
    pub ceremony_participation_rate: f64,
}

/// Validator ceremony participation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyHistory {
    pub total_ceremonies: usize,
    pub successful_ceremonies: usize,
    pub failed_ceremonies: usize,
    pub average_contribution_time: f64,
    pub last_participation: Option<DateTime<Utc>>,
}

/// Validator contribution to ceremony round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorContribution {
    pub ceremony_id: CeremonyId,
    pub round: RoundNumber,
    pub validator_id: ValidatorId,
    pub contribution_data: ContributionData,
    pub proof_of_computation: ProofOfComputation,
    pub signature: Vec<u8>,
    pub submitted_at: DateTime<Utc>,
}

/// Contribution data containing cryptographic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionData {
    pub tau_powers: Vec<Vec<u8>>, // Powers of tau
    pub alpha_tau_powers: Vec<Vec<u8>>, // Alpha times powers of tau
    pub beta_tau_powers: Vec<Vec<u8>>, // Beta times powers of tau
    pub gamma_inverse: Vec<u8>, // Inverse of gamma
    pub delta_inverse: Vec<u8>, // Inverse of delta
    pub entropy_contribution: Vec<u8>, // Additional entropy
}

/// Proof that contribution was computed correctly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfComputation {
    pub computation_proof: Vec<u8>,
    pub verification_key: Vec<u8>,
    pub computation_time: std::time::Duration,
    pub memory_usage: usize,
}

/// Request for validator contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionRequest {
    pub ceremony_id: CeremonyId,
    pub round: RoundNumber,
    pub validator_id: ValidatorId,
    pub beacon: RandomBeacon,
    pub deadline: DateTime<Utc>,
}

/// Result of contribution verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub verification_time: std::time::Duration,
    pub failure_reason: Option<String>,
    pub verified_at: DateTime<Utc>,
    pub verifier_id: String,
}

/// Final verified parameters from ceremony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedParameters {
    pub id: ParameterSetId,
    pub ceremony_id: CeremonyId,
    pub parameters: CeremonyParameters,
    pub verification: FinalVerification,
    pub participants: Vec<ValidatorId>,
    pub created_at: DateTime<Utc>,
}

/// Final ceremony parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyParameters {
    pub parameter_type: CeremonyType,
    pub tau_powers: Vec<Vec<u8>>,
    pub alpha_tau_powers: Vec<Vec<u8>>,
    pub beta_tau_powers: Vec<Vec<u8>>,
    pub gamma_inverse: Vec<u8>,
    pub delta_inverse: Vec<u8>,
    pub verification_key: Vec<u8>,
    pub parameter_hash: Vec<u8>,
}

/// Final verification of ceremony parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalVerification {
    pub is_valid: bool,
    pub verification_proofs: Vec<VerificationProof>,
    pub parameter_integrity_check: bool,
    pub contribution_chain_valid: bool,
    pub randomness_quality_score: f64,
    pub verified_at: DateTime<Utc>,
}

/// Individual verification proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationProof {
    pub proof_type: VerificationProofType,
    pub proof_data: Vec<u8>,
    pub verifier_signature: Vec<u8>,
}

/// Types of verification proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationProofType {
    ParameterConsistency,
    ContributionChain,
    RandomnessQuality,
    ComputationCorrectness,
}

/// Ceremony violation for slashing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CeremonyViolation {
    InvalidContribution { round: RoundNumber, reason: String },
    MissedContribution { round: RoundNumber },
    InvalidSignature { round: RoundNumber },
    ComputationFraud { round: RoundNumber, proof: Vec<u8> },
    NetworkDisruption { duration: std::time::Duration },
}

/// Ceremony configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyConfig {
    pub ceremony_timeout: u64,
    pub round_timeout: u64,
    pub contribution_timeout: u64,
    pub min_participants: usize,
    pub max_participants: usize,
    pub universal_ceremony_rounds: usize,
    pub circuit_specific_rounds: usize,
    pub update_ceremony_rounds: usize,
    pub restart_attempts: usize,
    pub emergency_halt_integration: bool,
    pub coordinator_config: CoordinatorConfig,
    pub beacon_config: BeaconConfig,
    pub parameter_config: ParameterConfig,
    pub verification_config: VerificationConfig,
    pub restart_config: RestartConfig,
}

/// Coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub validator_selection_strategy: ValidatorSelectionStrategy,
    pub reputation_threshold: f64,
    pub performance_threshold: f64,
    pub geographic_distribution: bool,
    pub max_validator_reuse: usize,
}

/// Validator selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorSelectionStrategy {
    ReputationBased,
    PerformanceBased,
    Random,
    Hybrid,
}

/// Random beacon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeaconConfig {
    pub entropy_sources: Vec<EntropySource>,
    pub signature_threshold: usize,
    pub beacon_timeout: u64,
    pub quality_threshold: f64,
}

/// Sources of entropy for beacon generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropySource {
    ValidatorSignatures,
    BlockHashes,
    NetworkTimestamps,
    ExternalRandomness,
}

/// Parameter generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConfig {
    pub tau_power_count: usize,
    pub security_level: SecurityLevel,
    pub computation_timeout: u64,
    pub memory_limit: usize,
    pub parallel_computation: bool,
}

/// Security levels for parameter generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Standard,
    High,
    Maximum,
}

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub verification_timeout: u64,
    pub parallel_verification: bool,
    pub redundant_verification: bool,
    pub quality_checks: Vec<QualityCheck>,
}

/// Quality checks for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityCheck {
    ParameterConsistency,
    RandomnessQuality,
    ComputationCorrectness,
    ContributionChain,
}

/// Restart mechanism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartConfig {
    pub max_restart_attempts: usize,
    pub restart_delay: u64,
    pub participant_replacement_rate: f64,
    pub consensus_threshold: f64,
}

/// Ceremony statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyStatistics {
    pub active_ceremonies: usize,
    pub cached_parameters: usize,
    pub total_ceremonies: u64,
    pub success_rate: f64,
    pub average_ceremony_duration: f64,
}

/// Network partition information affecting ceremony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartition {
    pub partition_id: Uuid,
    pub affected_validators: Vec<ValidatorId>,
    pub detected_at: DateTime<Utc>,
    pub estimated_duration: Option<std::time::Duration>,
    pub impact_severity: PartitionSeverity,
}

/// Severity of network partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionSeverity {
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Emergency halt information for ceremony coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyHalt {
    pub halt_id: Uuid,
    pub reason: String,
    pub initiated_by: ValidatorId,
    pub initiated_at: DateTime<Utc>,
    pub affected_ceremonies: Vec<CeremonyId>,
    pub recovery_plan: Option<String>,
}

/// Ceremony restart information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonyRestart {
    pub restart_id: Uuid,
    pub original_ceremony: CeremonyId,
    pub restart_reason: RestartReason,
    pub new_participants: Vec<ValidatorId>,
    pub retained_participants: Vec<ValidatorId>,
    pub restart_round: RoundNumber,
    pub initiated_at: DateTime<Utc>,
}

/// Reasons for ceremony restart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RestartReason {
    Timeout,
    InsufficientParticipation,
    VerificationFailure,
    NetworkPartition,
    ValidatorFailure,
    EmergencyHalt,
}

impl Default for CeremonyConfig {
    fn default() -> Self {
        Self {
            ceremony_timeout: 3600, // 1 hour
            round_timeout: 300, // 5 minutes
            contribution_timeout: 120, // 2 minutes
            min_participants: 7, // Minimum for Byzantine fault tolerance
            max_participants: 21, // Maximum validator set size
            universal_ceremony_rounds: 5,
            circuit_specific_rounds: 3,
            update_ceremony_rounds: 2,
            restart_attempts: 3,
            emergency_halt_integration: true,
            coordinator_config: CoordinatorConfig::default(),
            beacon_config: BeaconConfig::default(),
            parameter_config: ParameterConfig::default(),
            verification_config: VerificationConfig::default(),
            restart_config: RestartConfig::default(),
        }
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            validator_selection_strategy: ValidatorSelectionStrategy::Hybrid,
            reputation_threshold: 0.7,
            performance_threshold: 0.8,
            geographic_distribution: true,
            max_validator_reuse: 3,
        }
    }
}

impl Default for BeaconConfig {
    fn default() -> Self {
        Self {
            entropy_sources: vec![
                EntropySource::ValidatorSignatures,
                EntropySource::BlockHashes,
                EntropySource::NetworkTimestamps,
            ],
            signature_threshold: 14, // 2/3 of 21 validators
            beacon_timeout: 60,
            quality_threshold: 0.9,
        }
    }
}

impl Default for ParameterConfig {
    fn default() -> Self {
        Self {
            tau_power_count: 1048576, // 2^20 for production security
            security_level: SecurityLevel::High,
            computation_timeout: 600, // 10 minutes
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            parallel_computation: true,
        }
    }
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            verification_timeout: 300, // 5 minutes
            parallel_verification: true,
            redundant_verification: true,
            quality_checks: vec![
                QualityCheck::ParameterConsistency,
                QualityCheck::RandomnessQuality,
                QualityCheck::ComputationCorrectness,
                QualityCheck::ContributionChain,
            ],
        }
    }
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restart_attempts: 3,
            restart_delay: 300, // 5 minutes
            participant_replacement_rate: 0.3, // Replace 30% of participants
            consensus_threshold: 0.67, // 2/3 consensus for restart
        }
    }
}
