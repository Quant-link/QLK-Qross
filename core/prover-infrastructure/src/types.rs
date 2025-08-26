//! Core types for prover infrastructure

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use qross_consensus::ValidatorId;
use qross_zk_circuits::{ZkStarkProof, CircuitId, CircuitInputs};

/// Prover identifier
pub type ProverId = Uuid;

/// Proof job identifier
pub type ProofJobId = Uuid;

/// Prover node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverNode {
    pub id: ProverId,
    pub validator_id: ValidatorId,
    pub capabilities: ProverCapabilities,
    pub resource_capacity: ResourceCapacity,
    pub current_load: ResourceUsage,
    pub status: ProverStatus,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub performance_history: Vec<PerformanceRecord>,
}

/// Prover capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverCapabilities {
    pub gpu_acceleration: bool,
    pub gpu_memory_gb: Option<u32>,
    pub gpu_compute_capability: Option<String>,
    pub cpu_cores: u32,
    pub supports_parallel_proving: bool,
    pub max_circuit_size: usize,
    pub supported_curves: Vec<String>,
}

/// Resource capacity of prover
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_memory_gb: u32,
    pub storage_gb: u32,
    pub network_bandwidth_mbps: u32,
}

/// Current resource usage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    pub cpu_used: f64,
    pub memory_used: u64,
    pub gpu_memory_used: u64,
    pub storage_used: u64,
    pub network_used: u64,
}

/// Prover status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProverStatus {
    Available,
    Busy,
    Offline,
    Maintenance,
    Error,
}

/// Performance record for prover
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord {
    pub timestamp: DateTime<Utc>,
    pub circuit_id: CircuitId,
    pub proof_generation_time: std::time::Duration,
    pub memory_peak_usage: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Proof job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofJob {
    pub id: ProofJobId,
    pub circuit_id: CircuitId,
    pub inputs: CircuitInputs,
    pub priority: ProofPriority,
    pub deadline: Option<DateTime<Utc>>,
    pub resource_requirements: ResourceRequirements,
    pub status: ProofJobStatus,
    pub assigned_prover: Option<ProverId>,
    pub submitted_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub result: Option<ProofJobResult>,
    pub retry_count: u32,
}

/// Proof job priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Resource requirements for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub estimated_cpu_time: std::time::Duration,
    pub estimated_memory_gb: u32,
    pub estimated_gpu_memory_gb: u32,
    pub requires_gpu: bool,
    pub parallel_threads: u32,
    pub estimated_duration: std::time::Duration,
}

/// Proof job status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProofJobStatus {
    Queued,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Proof job result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofJobResult {
    Success(ZkStarkProof),
    Failure(String),
}

/// Prover node registration info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverNodeInfo {
    pub validator_id: ValidatorId,
    pub capabilities: ProverCapabilities,
    pub resource_capacity: ResourceCapacity,
}

/// Validator performance metrics for prover allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformanceMetrics {
    pub proof_generation_score: f64,
    pub reliability_score: f64,
    pub response_time_score: f64,
    pub resource_efficiency_score: f64,
    pub uptime_percentage: f64,
}

/// Proof generation performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofGenerationPerformance {
    pub circuit_id: CircuitId,
    pub generation_time: std::time::Duration,
    pub memory_usage: u64,
    pub success: bool,
}

/// Proof metadata for aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub job_id: ProofJobId,
    pub prover_id: ProverId,
    pub generation_time: std::time::Duration,
    pub resource_usage: ResourceUsage,
}

/// Batch processing requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequirements {
    pub preferred_batch_size: usize,
    pub max_batch_size: usize,
    pub batch_timeout: std::time::Duration,
    pub priority_grouping: bool,
}

/// Batch information for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInfo {
    pub batch_id: Uuid,
    pub job_ids: Vec<ProofJobId>,
    pub estimated_completion: DateTime<Utc>,
    pub resource_allocation: ResourceAllocation,
}

/// Resource allocation for batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub assigned_provers: Vec<ProverId>,
    pub total_cpu_cores: u32,
    pub total_memory_gb: u32,
    pub total_gpu_memory_gb: u32,
}

/// Scaling decision for infrastructure
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    NoChange,
}

/// Infrastructure statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureStatistics {
    pub active_provers: usize,
    pub queued_jobs: usize,
    pub running_jobs: usize,
    pub total_jobs: usize,
    pub average_job_time: f64,
    pub throughput: f64,
}

/// Prover infrastructure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProverConfig {
    pub max_retries: u32,
    pub target_queue_size: usize,
    pub min_provers: usize,
    pub max_provers: usize,
    pub heartbeat_interval: u64,
    pub job_timeout: u64,
    pub scaling_interval: u64,
    pub orchestrator_config: OrchestratorConfig,
    pub workload_config: WorkloadConfig,
    pub resource_config: ResourceConfig,
    pub scheduler_config: SchedulerConfig,
    pub kubernetes_config: KubernetesConfig,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub validator_performance_weight: f64,
    pub resource_efficiency_weight: f64,
    pub reliability_weight: f64,
    pub geographic_distribution: bool,
    pub load_balancing_strategy: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    PerformanceBased,
    Hybrid,
}

/// Workload balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    pub balancing_algorithm: BalancingAlgorithm,
    pub resource_utilization_threshold: f64,
    pub performance_history_weight: f64,
    pub circuit_specialization: bool,
}

/// Balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BalancingAlgorithm {
    WeightedRoundRobin,
    LeastConnections,
    ResourceAware,
    ConsistentHashing,
}

/// Resource manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub memory_safety_margin: f64,
    pub cpu_oversubscription_ratio: f64,
    pub gpu_memory_reservation: f64,
    pub resource_monitoring_interval: u64,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub scheduling_algorithm: SchedulingAlgorithm,
    pub priority_queue_size: usize,
    pub deadline_awareness: bool,
    pub batch_optimization: bool,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    FIFO,
    PriorityBased,
    DeadlineAware,
    ResourceOptimized,
}

/// Kubernetes configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    pub namespace: String,
    pub prover_image: String,
    pub resource_requests: KubernetesResources,
    pub resource_limits: KubernetesResources,
    pub node_selector: HashMap<String, String>,
    pub tolerations: Vec<KubernetesToleration>,
}

/// Kubernetes resource specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesResources {
    pub cpu: String,
    pub memory: String,
    pub gpu: Option<String>,
}

/// Kubernetes toleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesToleration {
    pub key: String,
    pub operator: String,
    pub value: Option<String>,
    pub effect: String,
}

/// GPU acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub device_selection: GpuDeviceSelection,
    pub memory_pool_size: u64,
    pub compute_stream_count: u32,
    pub optimization_level: GpuOptimizationLevel,
}

/// GPU device selection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuDeviceSelection {
    Automatic,
    ByIndex(u32),
    ByMemory,
    ByComputeCapability,
}

/// GPU optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuOptimizationLevel {
    Balanced,
    Speed,
    Memory,
    Power,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            target_queue_size: 100,
            min_provers: 3,
            max_provers: 50,
            heartbeat_interval: 30,
            job_timeout: 3600,
            scaling_interval: 60,
            orchestrator_config: OrchestratorConfig::default(),
            workload_config: WorkloadConfig::default(),
            resource_config: ResourceConfig::default(),
            scheduler_config: SchedulerConfig::default(),
            kubernetes_config: KubernetesConfig::default(),
        }
    }
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            validator_performance_weight: 0.4,
            resource_efficiency_weight: 0.3,
            reliability_weight: 0.3,
            geographic_distribution: true,
            load_balancing_strategy: LoadBalancingStrategy::Hybrid,
        }
    }
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            balancing_algorithm: BalancingAlgorithm::ResourceAware,
            resource_utilization_threshold: 0.8,
            performance_history_weight: 0.3,
            circuit_specialization: true,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            memory_safety_margin: 0.2,
            cpu_oversubscription_ratio: 1.5,
            gpu_memory_reservation: 0.1,
            resource_monitoring_interval: 10,
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduling_algorithm: SchedulingAlgorithm::PriorityBased,
            priority_queue_size: 1000,
            deadline_awareness: true,
            batch_optimization: true,
        }
    }
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            namespace: "qross-provers".to_string(),
            prover_image: "qross/prover:latest".to_string(),
            resource_requests: KubernetesResources {
                cpu: "2".to_string(),
                memory: "8Gi".to_string(),
                gpu: Some("1".to_string()),
            },
            resource_limits: KubernetesResources {
                cpu: "4".to_string(),
                memory: "16Gi".to_string(),
                gpu: Some("1".to_string()),
            },
            node_selector: HashMap::new(),
            tolerations: Vec::new(),
        }
    }
}
