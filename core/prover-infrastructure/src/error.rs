//! Error types for prover infrastructure

use thiserror::Error;
use crate::{ProverId, ProofJobId};

/// Result type alias
pub type Result<T> = std::result::Result<T, ProverError>;

/// Prover infrastructure error types
#[derive(Error, Debug)]
pub enum ProverError {
    /// Prover not found
    #[error("Prover not found: {0}")]
    ProverNotFound(ProverId),
    
    /// Proof job not found
    #[error("Proof job not found: {0}")]
    ProofJobNotFound(ProofJobId),
    
    /// No prover assigned to job
    #[error("No prover assigned to job: {0}")]
    NoProverAssigned(ProofJobId),
    
    /// Insufficient GPU resources
    #[error("Insufficient GPU resources: required {required_memory}GB, available devices: {available_devices}")]
    InsufficientGpuResources { required_memory: u32, available_devices: usize },
    
    /// GPU device not found
    #[error("GPU device not found: {0}")]
    GpuDeviceNotFound(u32),
    
    /// Insufficient GPU memory
    #[error("Insufficient GPU memory: required {required} bytes")]
    InsufficientGpuMemory { required: u64 },
    
    /// Insufficient GPU streams
    #[error("Insufficient GPU streams: required {required}, available {available}")]
    InsufficientGpuStreams { required: usize, available: usize },
    
    /// GPU computation failed
    #[error("GPU computation failed: {0}")]
    GpuComputationFailed(String),
    
    /// Kernel compilation failed
    #[error("Kernel compilation failed: {0}")]
    KernelCompilationFailed(String),
    
    /// Resource allocation failed
    #[error("Resource allocation failed: {0}")]
    ResourceAllocationFailed(String),
    
    /// Workload balancing failed
    #[error("Workload balancing failed: {0}")]
    WorkloadBalancingFailed(String),
    
    /// Scheduling failed
    #[error("Scheduling failed: {0}")]
    SchedulingFailed(String),
    
    /// Kubernetes operation failed
    #[error("Kubernetes operation failed: {0}")]
    KubernetesOperationFailed(String),
    
    /// Prover registration failed
    #[error("Prover registration failed: {0}")]
    ProverRegistrationFailed(String),
    
    /// Invalid prover configuration
    #[error("Invalid prover configuration: {0}")]
    InvalidProverConfiguration(String),
    
    /// Proof generation timeout
    #[error("Proof generation timeout")]
    ProofGenerationTimeout,
    
    /// Resource exhaustion
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    /// Memory limit exceeded
    #[error("Memory limit exceeded: used {used}GB, limit {limit}GB")]
    MemoryLimitExceeded { used: u32, limit: u32 },
    
    /// CPU limit exceeded
    #[error("CPU limit exceeded: used {used}%, limit {limit}%")]
    CpuLimitExceeded { used: f64, limit: f64 },
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Metrics error
    #[error("Metrics error: {0}")]
    Metrics(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ProverError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            ProverError::ProofGenerationTimeout => true,
            ProverError::ResourceExhaustion(_) => true,
            ProverError::NetworkError(_) => true,
            ProverError::GpuComputationFailed(_) => true,
            ProverError::Io(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ProverError::ProverNotFound(_) => ErrorSeverity::Medium,
            ProverError::ProofJobNotFound(_) => ErrorSeverity::Medium,
            ProverError::NoProverAssigned(_) => ErrorSeverity::Medium,
            ProverError::InsufficientGpuResources { .. } => ErrorSeverity::High,
            ProverError::GpuDeviceNotFound(_) => ErrorSeverity::High,
            ProverError::InsufficientGpuMemory { .. } => ErrorSeverity::Medium,
            ProverError::InsufficientGpuStreams { .. } => ErrorSeverity::Medium,
            ProverError::GpuComputationFailed(_) => ErrorSeverity::High,
            ProverError::KernelCompilationFailed(_) => ErrorSeverity::High,
            ProverError::ResourceAllocationFailed(_) => ErrorSeverity::Medium,
            ProverError::WorkloadBalancingFailed(_) => ErrorSeverity::Medium,
            ProverError::SchedulingFailed(_) => ErrorSeverity::Medium,
            ProverError::KubernetesOperationFailed(_) => ErrorSeverity::High,
            ProverError::ProverRegistrationFailed(_) => ErrorSeverity::Medium,
            ProverError::InvalidProverConfiguration(_) => ErrorSeverity::High,
            ProverError::ProofGenerationTimeout => ErrorSeverity::Medium,
            ProverError::ResourceExhaustion(_) => ErrorSeverity::Medium,
            ProverError::MemoryLimitExceeded { .. } => ErrorSeverity::Medium,
            ProverError::CpuLimitExceeded { .. } => ErrorSeverity::Medium,
            ProverError::NetworkError(_) => ErrorSeverity::Medium,
            ProverError::Serialization(_) => ErrorSeverity::Medium,
            ProverError::Io(_) => ErrorSeverity::Medium,
            ProverError::Configuration(_) => ErrorSeverity::High,
            ProverError::Metrics(_) => ErrorSeverity::Low,
            ProverError::Internal(_) => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            ProverError::ProverNotFound(_) => "prover_not_found",
            ProverError::ProofJobNotFound(_) => "proof_job_not_found",
            ProverError::NoProverAssigned(_) => "no_prover_assigned",
            ProverError::InsufficientGpuResources { .. } => "insufficient_gpu_resources",
            ProverError::GpuDeviceNotFound(_) => "gpu_device_not_found",
            ProverError::InsufficientGpuMemory { .. } => "insufficient_gpu_memory",
            ProverError::InsufficientGpuStreams { .. } => "insufficient_gpu_streams",
            ProverError::GpuComputationFailed(_) => "gpu_computation_failed",
            ProverError::KernelCompilationFailed(_) => "kernel_compilation_failed",
            ProverError::ResourceAllocationFailed(_) => "resource_allocation_failed",
            ProverError::WorkloadBalancingFailed(_) => "workload_balancing_failed",
            ProverError::SchedulingFailed(_) => "scheduling_failed",
            ProverError::KubernetesOperationFailed(_) => "kubernetes_operation_failed",
            ProverError::ProverRegistrationFailed(_) => "prover_registration_failed",
            ProverError::InvalidProverConfiguration(_) => "invalid_prover_configuration",
            ProverError::ProofGenerationTimeout => "proof_generation_timeout",
            ProverError::ResourceExhaustion(_) => "resource_exhaustion",
            ProverError::MemoryLimitExceeded { .. } => "memory_limit_exceeded",
            ProverError::CpuLimitExceeded { .. } => "cpu_limit_exceeded",
            ProverError::NetworkError(_) => "network_error",
            ProverError::Serialization(_) => "serialization",
            ProverError::Io(_) => "io",
            ProverError::Configuration(_) => "configuration",
            ProverError::Metrics(_) => "metrics",
            ProverError::Internal(_) => "internal",
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
