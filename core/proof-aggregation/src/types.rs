//! Core types for proof aggregation protocol

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use qross_zk_circuits::{ZkStarkProof, ZkStarkConfig, CircuitId};
use petgraph::Graph;

/// Aggregation identifier
pub type AggregationId = Uuid;

/// Proof identifier
pub type ProofId = Uuid;

/// Validator identifier
pub type ValidatorId = String;

/// Dependency identifier
pub type DependencyId = Uuid;

/// State transition identifier
pub type StateTransitionId = Uuid;

/// Proof submission for aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSubmission {
    pub proof: ZkStarkProof,
    pub state_transition: Option<StateTransition>,
    pub priority: AggregationPriority,
    pub dependencies: Vec<DependencyId>,
    pub submitted_by: ValidatorId,
    pub submitted_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
}

/// Cross-chain state transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub id: StateTransitionId,
    pub source_chain: String,
    pub target_chain: String,
    pub source_block_height: u64,
    pub target_block_height: u64,
    pub state_root_before: Vec<u8>,
    pub state_root_after: Vec<u8>,
    pub transition_data: Vec<u8>,
    pub merkle_proofs: Vec<MerkleProofData>,
}

/// Merkle proof data for state verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProofData {
    pub leaf_index: usize,
    pub leaf_value: Vec<u8>,
    pub proof_path: Vec<Vec<u8>>,
    pub root: Vec<u8>,
}

/// State dependency between chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDependency {
    pub id: DependencyId,
    pub chain_id: String,
    pub required_block_height: u64,
    pub dependency_type: DependencyType,
    pub timeout: Option<DateTime<Utc>>,
}

/// Types of dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    /// Must wait for block finalization
    BlockFinalization,
    /// Must wait for state root update
    StateRootUpdate,
    /// Must wait for cross-chain message
    CrossChainMessage,
    /// Must wait for liquidity availability
    LiquidityAvailability,
}

/// Chain state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainState {
    pub chain_id: String,
    pub block_height: u64,
    pub block_hash: Vec<u8>,
    pub state_root: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub finalized: bool,
}

/// Aggregation priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AggregationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Aggregation session tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationSession {
    pub id: AggregationId,
    pub proofs: Vec<ProofSubmission>,
    pub dependency_graph: DependencyGraph,
    pub status: AggregationStatus,
    pub created_at: DateTime<Utc>,
    pub target_composition_depth: usize,
    pub assigned_validators: Vec<ValidatorId>,
    pub progress: AggregationProgress,
}

/// Dependency graph for proof ordering
pub type DependencyGraph = Graph<ProofId, DependencyType>;

/// Aggregation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregationStatus {
    Pending,
    Processing,
    AwaitingFinality,
    Finalized,
    Failed,
    Cancelled,
}

/// Aggregation progress tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregationProgress {
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub completed_batches: usize,
    pub total_batches: usize,
    pub current_phase: AggregationPhase,
}

/// Phases of aggregation process
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AggregationPhase {
    #[default]
    Initialization,
    DependencyAnalysis,
    ValidatorAllocation,
    ProofGeneration,
    Verification,
    FinalitySubmission,
    Completed,
}

/// Aggregated proof result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProof {
    pub id: ProofId,
    pub component_proof_ids: Vec<ProofId>,
    pub proof: ZkStarkProof,
    pub aggregation_metadata: AggregationMetadata,
    pub created_at: DateTime<Utc>,
}

/// Metadata about aggregation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationMetadata {
    pub composition_depth: usize,
    pub compression_ratio: f64,
    pub validator_signatures: Vec<ValidatorSignature>,
}

/// Validator signature on aggregated proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    pub validator_id: ValidatorId,
    pub signature: Vec<u8>,
    pub signed_at: DateTime<Utc>,
}

/// Cached aggregated proof
#[derive(Debug, Clone)]
pub struct CachedAggregatedProof {
    pub proof: AggregatedProof,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
}

/// Finality status from consensus layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinalityStatus {
    Pending,
    Finalized,
    Rejected,
}

/// Validator performance metrics for allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub proof_generation_score: f64,
    pub verification_speed: f64,
    pub uptime_percentage: f64,
    pub resource_efficiency: f64,
    pub network_latency: f64,
}

/// Aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub max_proofs_per_aggregation: usize,
    pub max_composition_depth: usize,
    pub aggregation_factor: usize,
    pub max_validators_per_aggregation: usize,
    pub max_cached_proofs: usize,
    pub aggregation_timeout: u64,
    pub batch_size: usize,
    pub parallel_processing: bool,
    pub emergency_halt_coordination: bool,
    pub final_composition_circuit_id: CircuitId,
    pub zk_config: ZkStarkConfig,
    pub dependency_config: DependencyConfig,
    pub batch_config: BatchConfig,
    pub finality_config: FinalityConfig,
}

/// Dependency analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConfig {
    pub max_dependency_depth: usize,
    pub dependency_timeout: u64,
    pub parallel_dependency_resolution: bool,
    pub cross_chain_validation: bool,
}

/// Batch processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_timeout: u64,
    pub parallel_batch_processing: bool,
    pub batch_optimization: bool,
}

/// Finality coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityConfig {
    pub finality_timeout: u64,
    pub consensus_integration: bool,
    pub emergency_halt_monitoring: bool,
    pub validator_signature_threshold: usize,
}

/// Aggregation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStatistics {
    pub active_aggregations: usize,
    pub cached_proofs: usize,
    pub total_aggregations: u64,
    pub average_aggregation_time: f64,
    pub compression_ratio: f64,
}

/// Batch processing result
pub type BatchResult = HashMap<ProofId, AggregatedProof>;

/// Processing order for dependency resolution
pub type ProcessingOrder = Vec<Vec<ProofId>>;

/// Network partition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartition {
    pub partition_id: Uuid,
    pub affected_chains: Vec<String>,
    pub detected_at: DateTime<Utc>,
    pub estimated_duration: Option<std::time::Duration>,
    pub severity: PartitionSeverity,
}

/// Partition severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionSeverity {
    Minor,
    Moderate,
    Severe,
    Critical,
}

/// Emergency halt information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyHalt {
    pub halt_id: Uuid,
    pub reason: String,
    pub initiated_by: ValidatorId,
    pub initiated_at: DateTime<Utc>,
    pub affected_operations: Vec<String>,
    pub recovery_plan: Option<String>,
}

/// Prover allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProverAllocationStrategy {
    /// Allocate based on performance metrics
    PerformanceBased,
    /// Round-robin allocation
    RoundRobin,
    /// Random allocation
    Random,
    /// Load-balanced allocation
    LoadBalanced,
}

/// Resource requirements for aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub storage_gb: usize,
    pub network_bandwidth_mbps: usize,
    pub estimated_duration: std::time::Duration,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            max_proofs_per_aggregation: 1000,
            max_composition_depth: 10,
            aggregation_factor: 16,
            max_validators_per_aggregation: 21,
            max_cached_proofs: 10000,
            aggregation_timeout: 3600, // 1 hour
            batch_size: 32,
            parallel_processing: true,
            emergency_halt_coordination: true,
            final_composition_circuit_id: 1000,
            zk_config: ZkStarkConfig::default(),
            dependency_config: DependencyConfig::default(),
            batch_config: BatchConfig::default(),
            finality_config: FinalityConfig::default(),
        }
    }
}

impl Default for DependencyConfig {
    fn default() -> Self {
        Self {
            max_dependency_depth: 5,
            dependency_timeout: 300, // 5 minutes
            parallel_dependency_resolution: true,
            cross_chain_validation: true,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            batch_timeout: 600, // 10 minutes
            parallel_batch_processing: true,
            batch_optimization: true,
        }
    }
}

impl Default for FinalityConfig {
    fn default() -> Self {
        Self {
            finality_timeout: 1800, // 30 minutes
            consensus_integration: true,
            emergency_halt_monitoring: true,
            validator_signature_threshold: 14, // 2/3 of 21 validators
        }
    }
}
