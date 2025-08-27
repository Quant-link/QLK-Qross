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

/// Parameter identifier for governance
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParameterId(pub String);

impl ParameterId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
}

/// Parameter impact analysis
#[derive(Debug, Clone)]
pub struct ParameterImpactAnalysis {
    pub security_impact_score: rust_decimal::Decimal,
    pub performance_impact_score: rust_decimal::Decimal,
    pub economic_impact_score: rust_decimal::Decimal,
    pub operational_complexity: rust_decimal::Decimal,
    pub rollback_difficulty: rust_decimal::Decimal,
    pub affected_components: Vec<String>,
}

/// Validation requirement for parameters
#[derive(Debug, Clone)]
pub struct ValidationRequirement {
    pub requirement_type: ValidationType,
    pub validation_criteria: String,
    pub required_confidence: rust_decimal::Decimal,
    pub timeout: chrono::Duration,
}

/// Validation types
#[derive(Debug, Clone)]
pub enum ValidationType {
    MathematicalProof,
    SecurityAudit,
    PerformanceBenchmark,
    EconomicModeling,
    SimulationTesting,
}

/// Dependency analysis result
#[derive(Debug, Clone)]
pub struct DependencyAnalysis {
    pub direct_dependencies: Vec<ParameterId>,
    pub indirect_dependencies: Vec<ParameterId>,
    pub circular_dependencies: Vec<Vec<ParameterId>>,
    pub dependency_graph: serde_json::Value,
    pub critical_path: Vec<ParameterId>,
}

/// Validation results for proposals
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub mathematical_validation: bool,
    pub security_validation: bool,
    pub performance_validation: bool,
    pub economic_validation: bool,
    pub invariant_validation: bool,
    pub validation_errors: Vec<String>,
    pub validation_warnings: Vec<String>,
    pub confidence_score: rust_decimal::Decimal,
}

impl ValidationResults {
    pub fn new() -> Self {
        Self {
            mathematical_validation: false,
            security_validation: false,
            performance_validation: false,
            economic_validation: false,
            invariant_validation: false,
            validation_errors: Vec::new(),
            validation_warnings: Vec::new(),
            confidence_score: rust_decimal::Decimal::ZERO,
        }
    }
}

/// Vote outcome for proposals
#[derive(Debug, Clone)]
pub enum VoteOutcome {
    Approved,
    Rejected,
    Pending,
}

/// Vote result calculation
#[derive(Debug, Clone)]
pub struct VoteResult {
    pub total_voting_power: rust_decimal::Decimal,
    pub yes_votes: rust_decimal::Decimal,
    pub no_votes: rust_decimal::Decimal,
    pub abstain_votes: rust_decimal::Decimal,
    pub participation_rate: rust_decimal::Decimal,
    pub quorum_met: bool,
    pub threshold_met: bool,
    pub decision_reached: bool,
    pub outcome: VoteOutcome,
}

/// Validator weight update
#[derive(Debug, Clone)]
pub struct ValidatorWeightUpdate {
    pub validator_id: ValidatorId,
    pub stake_weight: u64, // Simplified for now
    pub reputation_score: u64, // Simplified for now
    pub governance_participation: rust_decimal::Decimal,
    pub delegation_power: rust_decimal::Decimal,
}

/// Execution result for proposals
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub execution_time: std::time::Duration,
    pub affected_components: Vec<String>,
    pub rollback_required: bool,
    pub error_messages: Vec<String>,
    pub success_metrics: HashMap<String, rust_decimal::Decimal>,
}

/// Validation result for execution
#[derive(Debug, Clone)]
pub struct ExecutionValidationResult {
    pub is_valid: bool,
    pub validation_errors: Vec<String>,
    pub risk_assessment: rust_decimal::Decimal,
    pub recommended_delay: Option<chrono::Duration>,
}

/// Rollback plan for failed executions
#[derive(Debug, Clone)]
pub struct RollbackPlan {
    pub rollback_steps: Vec<RollbackStep>,
    pub rollback_timeout: chrono::Duration,
    pub rollback_validation: Vec<ValidationCheckpoint>,
}

/// Rollback step
#[derive(Debug, Clone)]
pub struct RollbackStep {
    pub step_id: uuid::Uuid,
    pub rollback_action: RollbackAction,
    pub target_parameter: ParameterId,
    pub previous_value: String, // Simplified for now
    pub validation_required: bool,
}

/// Rollback action types
#[derive(Debug, Clone)]
pub enum RollbackAction {
    RestoreParameter,
    RestartComponent,
    RevertConfiguration,
    RestoreDatabase,
    EmergencyStop,
}

/// Validation checkpoint
#[derive(Debug, Clone)]
pub struct ValidationCheckpoint {
    pub checkpoint_id: uuid::Uuid,
    pub validation_type: ValidationType,
    pub success_criteria: Vec<String>,
    pub timeout: chrono::Duration,
}

/// Rollback trigger
#[derive(Debug, Clone)]
pub enum RollbackTrigger {
    PerformanceDegradation { threshold: rust_decimal::Decimal },
    SecurityViolation { severity: String },
    SystemFailure { component: String },
    UserDefinedMetric { metric: String, threshold: rust_decimal::Decimal },
}

/// Success criterion for execution
#[derive(Debug, Clone)]
pub struct SuccessCriterion {
    pub criterion_id: uuid::Uuid,
    pub metric_name: String,
    pub target_value: rust_decimal::Decimal,
    pub tolerance: rust_decimal::Decimal,
    pub measurement_window: chrono::Duration,
}

/// Monitoring requirements for execution
#[derive(Debug, Clone)]
pub struct MonitoringRequirements {
    pub metrics_to_monitor: Vec<String>,
    pub monitoring_duration: chrono::Duration,
    pub alert_thresholds: HashMap<String, rust_decimal::Decimal>,
    pub reporting_frequency: chrono::Duration,
}

/// Emergency system status
#[derive(Debug, Clone)]
pub struct EmergencySystemStatus {
    pub active_emergencies: usize,
    pub highest_emergency_level: crate::emergency_mechanisms::EmergencyLevel,
    pub affected_components: std::collections::HashSet<crate::emergency_mechanisms::ComponentId>,
    pub system_health_score: rust_decimal::Decimal,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Detected threat information
#[derive(Debug, Clone)]
pub struct DetectedThreat {
    pub threat_id: uuid::Uuid,
    pub threat_type: crate::emergency_mechanisms::EmergencyType,
    pub threat_description: String,
    pub detector_type: String,
    pub confidence_score: rust_decimal::Decimal,
    pub severity_score: rust_decimal::Decimal,
    pub affected_components: Vec<crate::emergency_mechanisms::ComponentId>,
    pub evidence: Vec<u8>,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Recovery plan for system restoration
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    pub plan_id: uuid::Uuid,
    pub affected_components: std::collections::HashSet<crate::emergency_mechanisms::ComponentId>,
    pub recovery_steps: Vec<RecoveryStep>,
    pub estimated_duration: chrono::Duration,
    pub success_criteria: Vec<String>,
    pub rollback_plan: Option<String>,
}

/// Recovery step in restoration process
#[derive(Debug, Clone)]
pub struct RecoveryStep {
    pub step_id: uuid::Uuid,
    pub step_type: RecoveryStepType,
    pub target_component: crate::emergency_mechanisms::ComponentId,
    pub step_description: String,
    pub dependencies: Vec<uuid::Uuid>,
    pub timeout: chrono::Duration,
    pub validation_required: bool,
}

/// Recovery step types
#[derive(Debug, Clone)]
pub enum RecoveryStepType {
    HealthCheck,
    ComponentRestart,
    StateValidation,
    IntegrityVerification,
    PerformanceTest,
    SecurityAudit,
}

/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub execution_time: std::time::Duration,
    pub recovered_components: Vec<crate::emergency_mechanisms::ComponentId>,
    pub failed_components: Vec<crate::emergency_mechanisms::ComponentId>,
    pub error_messages: Vec<String>,
    pub integrity_verified: bool,
}

/// Isolation result
#[derive(Debug, Clone)]
pub struct IsolationResult {
    pub success: bool,
    pub execution_time: std::time::Duration,
    pub affected_components: Vec<crate::emergency_mechanisms::ComponentId>,
    pub side_effects: Vec<String>,
    pub error_messages: Vec<String>,
}

/// System integrity verification result
#[derive(Debug, Clone)]
pub struct IntegrityVerificationResult {
    pub is_valid: bool,
    pub verification_time: std::time::Duration,
    pub verified_components: Vec<crate::emergency_mechanisms::ComponentId>,
    pub failed_verifications: Vec<String>,
    pub confidence_score: rust_decimal::Decimal,
}
