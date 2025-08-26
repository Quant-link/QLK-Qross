//! Core types for the zk-STARK circuit library

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use winterfell::{StarkProof, ProofOptions};
use math::fields::f64::BaseElement;

/// Circuit identifier
pub type CircuitId = u32;

/// Proof identifier
pub type ProofId = Uuid;

/// Circuit inputs containing public, private, and auxiliary data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInputs {
    pub public_inputs: Vec<BaseElement>,
    pub private_inputs: Vec<BaseElement>,
    pub auxiliary_inputs: HashMap<String, Vec<BaseElement>>,
}

/// zk-STARK proof wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkStarkProof {
    pub id: ProofId,
    pub circuit_id: CircuitId,
    pub stark_proof: StarkProof,
    pub inputs: CircuitInputs,
    pub options: ProofOptions,
    pub generated_at: DateTime<Utc>,
    pub generation_time: std::time::Duration,
    pub proof_size: usize,
}

/// Cached proof with access tracking
#[derive(Debug, Clone)]
pub struct CachedProof {
    pub proof: ZkStarkProof,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
}

/// Circuit complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitComplexity {
    pub constraint_count: usize,
    pub trace_length: usize,
    pub trace_width: usize,
    pub degree: usize,
    pub memory_usage: usize,
    pub estimated_proving_time: std::time::Duration,
    pub estimated_verification_time: std::time::Duration,
}

impl CircuitComplexity {
    /// Get complexity category for statistics
    pub fn category(&self) -> ComplexityCategory {
        match self.constraint_count {
            0..=100 => ComplexityCategory::Low,
            101..=1000 => ComplexityCategory::Medium,
            1001..=10000 => ComplexityCategory::High,
            _ => ComplexityCategory::VeryHigh,
        }
    }
}

/// Complexity categories for circuit classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityCategory {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Optimization constraints for circuit optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConstraints {
    pub max_proving_time: std::time::Duration,
    pub max_verification_time: std::time::Duration,
    pub max_memory_usage: usize,
    pub max_proof_size: usize,
    pub target_security_level: SecurityLevel,
    pub parallelization_factor: usize,
}

/// Security levels for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 80-bit security
    Standard,
    /// 128-bit security
    High,
    /// 256-bit security
    Maximum,
}

/// zk-STARK engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkStarkConfig {
    pub max_cached_proofs: usize,
    pub default_proof_options: ProofOptions,
    pub recursive_proof_options: ProofOptions,
    pub optimization_enabled: bool,
    pub parallel_proving: bool,
    pub proof_compression: bool,
    pub security_level: SecurityLevel,
}

impl Default for ZkStarkConfig {
    fn default() -> Self {
        Self {
            max_cached_proofs: 1000,
            default_proof_options: ProofOptions::default(),
            recursive_proof_options: ProofOptions::default(),
            optimization_enabled: true,
            parallel_proving: true,
            proof_compression: true,
            security_level: SecurityLevel::High,
        }
    }
}

/// Circuit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStatistics {
    pub total_circuits: usize,
    pub cached_proofs: usize,
    pub total_proofs_generated: u64,
    pub total_proofs_verified: u64,
    pub average_proof_generation_time: f64,
    pub average_verification_time: f64,
    pub complexity_distribution: HashMap<ComplexityCategory, usize>,
}

/// Merkle tree configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleConfig {
    pub tree_height: usize,
    pub hash_function: HashFunction,
    pub leaf_count: usize,
    pub batch_size: usize,
}

/// Supported hash functions for Merkle trees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashFunction {
    Blake3,
    Sha256,
    Poseidon,
    Rescue,
}

/// Merkle proof data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub leaf_value: Vec<u8>,
    pub proof_path: Vec<MerkleNode>,
    pub root: MerkleNode,
}

/// Merkle tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    pub hash: Vec<u8>,
    pub is_left: bool,
}

/// Polynomial commitment scheme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    pub degree_bound: usize,
    pub commitment_scheme: CommitmentScheme,
    pub field_size: usize,
    pub evaluation_domain_size: usize,
}

/// Supported polynomial commitment schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommitmentScheme {
    KZG,
    FRI,
    IPA,
    Bulletproofs,
}

/// Polynomial commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialCommitment {
    pub commitment: Vec<u8>,
    pub degree: usize,
    pub scheme: CommitmentScheme,
    pub parameters: Vec<u8>,
}

/// Polynomial evaluation proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationProof {
    pub point: BaseElement,
    pub value: BaseElement,
    pub proof: Vec<u8>,
    pub commitment: PolynomialCommitment,
}

/// Recursive proof composition data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveProofData {
    pub base_proofs: Vec<ProofId>,
    pub composition_circuit: CircuitId,
    pub aggregation_factor: usize,
    pub depth_level: usize,
}

/// Cross-chain state transition proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionProof {
    pub source_chain: String,
    pub target_chain: String,
    pub source_state_root: Vec<u8>,
    pub target_state_root: Vec<u8>,
    pub transition_proof: ZkStarkProof,
    pub merkle_proofs: Vec<MerkleProof>,
    pub block_number: u64,
    pub timestamp: DateTime<Utc>,
}

/// Batch proof for multiple state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStateProof {
    pub batch_id: Uuid,
    pub state_transitions: Vec<StateTransitionProof>,
    pub aggregated_proof: ZkStarkProof,
    pub batch_root: Vec<u8>,
    pub batch_size: usize,
    pub created_at: DateTime<Utc>,
}

/// Trusted setup ceremony parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedSetupParams {
    pub ceremony_id: Uuid,
    pub participants: Vec<String>,
    pub parameters: Vec<u8>,
    pub verification_key: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub is_verified: bool,
}

/// Circuit optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub original_complexity: CircuitComplexity,
    pub optimized_complexity: CircuitComplexity,
    pub optimization_time: std::time::Duration,
    pub improvements: HashMap<String, f64>,
}

/// Proof generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRequest {
    pub id: Uuid,
    pub circuit_id: CircuitId,
    pub inputs: CircuitInputs,
    pub options: ProofOptions,
    pub priority: ProofPriority,
    pub requested_at: DateTime<Utc>,
    pub deadline: Option<DateTime<Utc>>,
}

/// Proof generation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Proof generation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Proof generation job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofJob {
    pub request: ProofRequest,
    pub status: ProofStatus,
    pub assigned_worker: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub result: Option<ZkStarkProof>,
    pub error: Option<String>,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            max_proving_time: std::time::Duration::from_secs(60),
            max_verification_time: std::time::Duration::from_millis(100),
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            max_proof_size: 1024 * 1024, // 1MB
            target_security_level: SecurityLevel::High,
            parallelization_factor: num_cpus::get(),
        }
    }
}

impl Default for MerkleConfig {
    fn default() -> Self {
        Self {
            tree_height: 20,
            hash_function: HashFunction::Blake3,
            leaf_count: 1024,
            batch_size: 32,
        }
    }
}

impl Default for PolynomialConfig {
    fn default() -> Self {
        Self {
            degree_bound: 1024,
            commitment_scheme: CommitmentScheme::FRI,
            field_size: 64,
            evaluation_domain_size: 2048,
        }
    }
}
