//! Quantlink Qross Distributed Prover Infrastructure
//! 
//! This module implements distributed proof generation network with GPU acceleration,
//! memory optimization, and workload balancing integrated with Kubernetes orchestration
//! and validator performance metrics for optimal resource allocation.

pub mod orchestrator;
pub mod prover;
pub mod workload;
pub mod scheduler;
pub mod resources;
pub mod gpu;
pub mod kubernetes;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use qross_consensus::ValidatorId;
use qross_zk_circuits::{ZkStarkProof, CircuitId};
use qross_proof_aggregation::ProofId;

pub use error::{ProverError, Result};
pub use types::*;

/// Main prover infrastructure engine
pub struct ProverInfrastructureEngine {
    orchestrator: orchestrator::ProverOrchestrator,
    workload_balancer: workload::WorkloadBalancer,
    resource_manager: resources::ResourceManager,
    scheduler: scheduler::ProofScheduler,
    kubernetes_manager: kubernetes::KubernetesManager,
    metrics: metrics::ProverMetrics,
    config: ProverConfig,
    active_provers: dashmap::DashMap<ProverId, ProverNode>,
    proof_queue: dashmap::DashMap<ProofJobId, ProofJob>,
}

/// Trait for validator performance integration
#[async_trait]
pub trait ValidatorPerformanceIntegration: Send + Sync {
    /// Get validator performance metrics for prover allocation
    async fn get_validator_performance(&self, validator_id: &ValidatorId) -> Result<ValidatorPerformanceMetrics>;
    
    /// Get all validator performance data
    async fn get_all_validator_performance(&self) -> Result<HashMap<ValidatorId, ValidatorPerformanceMetrics>>;
    
    /// Update validator performance based on proof generation
    async fn update_proof_generation_performance(
        &self,
        validator_id: &ValidatorId,
        performance_data: &ProofGenerationPerformance,
    ) -> Result<()>;
}

/// Trait for proof aggregation integration
#[async_trait]
pub trait ProofAggregationIntegration: Send + Sync {
    /// Submit completed proof for aggregation
    async fn submit_proof_for_aggregation(&self, proof: &ZkStarkProof, metadata: &ProofMetadata) -> Result<ProofId>;
    
    /// Get batch processing requirements
    async fn get_batch_requirements(&self) -> Result<BatchRequirements>;
    
    /// Coordinate with aggregation scheduling
    async fn coordinate_batch_scheduling(&self, batch_info: &BatchInfo) -> Result<()>;
}

impl ProverInfrastructureEngine {
    /// Create a new prover infrastructure engine
    pub fn new(
        config: ProverConfig,
        validator_integration: Box<dyn ValidatorPerformanceIntegration>,
        aggregation_integration: Box<dyn ProofAggregationIntegration>,
    ) -> Self {
        let orchestrator = orchestrator::ProverOrchestrator::new(
            config.orchestrator_config.clone(),
            validator_integration,
        );
        let workload_balancer = workload::WorkloadBalancer::new(config.workload_config.clone());
        let resource_manager = resources::ResourceManager::new(config.resource_config.clone());
        let scheduler = scheduler::ProofScheduler::new(config.scheduler_config.clone());
        let kubernetes_manager = kubernetes::KubernetesManager::new(config.kubernetes_config.clone());
        let metrics = metrics::ProverMetrics::new();
        
        Self {
            orchestrator,
            workload_balancer,
            resource_manager,
            scheduler,
            kubernetes_manager,
            metrics,
            config,
            active_provers: dashmap::DashMap::new(),
            proof_queue: dashmap::DashMap::new(),
        }
    }
    
    /// Submit proof generation job
    pub async fn submit_proof_job(
        &self,
        circuit_id: CircuitId,
        inputs: qross_zk_circuits::CircuitInputs,
        priority: ProofPriority,
        deadline: Option<DateTime<Utc>>,
    ) -> Result<ProofJobId> {
        let job_id = Uuid::new_v4();
        let start_time = std::time::Instant::now();
        
        // Estimate resource requirements
        let resource_requirements = self.resource_manager.estimate_proof_requirements(
            circuit_id,
            &inputs,
        ).await?;
        
        // Create proof job
        let proof_job = ProofJob {
            id: job_id,
            circuit_id,
            inputs,
            priority,
            deadline,
            resource_requirements,
            status: ProofJobStatus::Queued,
            assigned_prover: None,
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            result: None,
            retry_count: 0,
        };
        
        self.proof_queue.insert(job_id, proof_job);
        
        // Schedule proof generation
        self.scheduler.schedule_proof_job(job_id).await?;
        
        self.metrics.increment_jobs_submitted();
        self.metrics.record_job_submission_time(start_time.elapsed().as_secs_f64());
        
        tracing::info!(
            "Submitted proof job {} for circuit {} with priority {:?}",
            job_id,
            circuit_id,
            priority
        );
        
        Ok(job_id)
    }
    
    /// Process proof generation queue
    pub async fn process_proof_queue(&self) -> Result<()> {
        // Get available provers
        let available_provers = self.orchestrator.get_available_provers().await?;
        
        if available_provers.is_empty() {
            tracing::debug!("No available provers for job processing");
            return Ok(());
        }
        
        // Get pending jobs sorted by priority and deadline
        let pending_jobs = self.get_pending_jobs_sorted().await?;
        
        for job_id in pending_jobs {
            if let Some(mut job) = self.proof_queue.get_mut(&job_id) {
                if job.status != ProofJobStatus::Queued {
                    continue;
                }
                
                // Find optimal prover for job
                let optimal_prover = self.workload_balancer.find_optimal_prover(
                    &job.resource_requirements,
                    &available_provers,
                ).await?;
                
                if let Some(prover_id) = optimal_prover {
                    // Assign job to prover
                    job.assigned_prover = Some(prover_id.clone());
                    job.status = ProofJobStatus::Assigned;
                    job.started_at = Some(Utc::now());
                    
                    // Start proof generation
                    let job_clone = job.clone();
                    drop(job);
                    
                    let engine_clone = self.clone_for_async();
                    tokio::spawn(async move {
                        if let Err(e) = engine_clone.execute_proof_job(job_clone).await {
                            tracing::error!("Proof job {} failed: {}", job_id, e);
                        }
                    });
                    
                    self.metrics.increment_jobs_assigned();
                    
                    tracing::info!("Assigned job {} to prover {}", job_id, prover_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute proof generation job
    async fn execute_proof_job(&self, mut job: ProofJob) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Get assigned prover
        let prover_id = job.assigned_prover.as_ref()
            .ok_or_else(|| ProverError::NoProverAssigned(job.id))?;
        
        let prover = self.active_provers.get(prover_id)
            .ok_or_else(|| ProverError::ProverNotFound(prover_id.clone()))?;
        
        // Update job status
        job.status = ProofJobStatus::Running;
        self.proof_queue.insert(job.id, job.clone());
        
        // Generate proof
        let proof_result = prover.generate_proof(
            job.circuit_id,
            &job.inputs,
        ).await;
        
        match proof_result {
            Ok(proof) => {
                // Update job with result
                job.status = ProofJobStatus::Completed;
                job.completed_at = Some(Utc::now());
                job.result = Some(ProofJobResult::Success(proof.clone()));
                
                // Submit to aggregation
                let metadata = ProofMetadata {
                    job_id: job.id,
                    prover_id: prover_id.clone(),
                    generation_time: start_time.elapsed(),
                    resource_usage: prover.get_resource_usage().await?,
                };
                
                self.orchestrator.submit_proof_for_aggregation(&proof, &metadata).await?;
                
                // Update performance metrics
                let performance_data = ProofGenerationPerformance {
                    circuit_id: job.circuit_id,
                    generation_time: start_time.elapsed(),
                    memory_usage: metadata.resource_usage.memory_used,
                    success: true,
                };
                
                self.orchestrator.update_validator_performance(
                    &prover.validator_id,
                    &performance_data,
                ).await?;
                
                self.metrics.increment_jobs_completed();
                self.metrics.record_proof_generation_time(start_time.elapsed().as_secs_f64());
                
                tracing::info!("Completed proof job {} in {:.2}s", job.id, start_time.elapsed().as_secs_f64());
            }
            Err(e) => {
                // Handle failure
                job.retry_count += 1;
                
                if job.retry_count < self.config.max_retries {
                    job.status = ProofJobStatus::Queued;
                    job.assigned_prover = None;
                    
                    tracing::warn!("Proof job {} failed, retrying (attempt {}): {}", job.id, job.retry_count, e);
                } else {
                    job.status = ProofJobStatus::Failed;
                    job.completed_at = Some(Utc::now());
                    job.result = Some(ProofJobResult::Failure(e.to_string()));
                    
                    self.metrics.increment_jobs_failed();
                    
                    tracing::error!("Proof job {} failed permanently after {} retries: {}", job.id, job.retry_count, e);
                }
                
                // Update performance metrics for failure
                let performance_data = ProofGenerationPerformance {
                    circuit_id: job.circuit_id,
                    generation_time: start_time.elapsed(),
                    memory_usage: 0, // Unknown on failure
                    success: false,
                };
                
                self.orchestrator.update_validator_performance(
                    &prover.validator_id,
                    &performance_data,
                ).await?;
            }
        }
        
        // Update job in queue
        self.proof_queue.insert(job.id, job);
        
        Ok(())
    }
    
    /// Get pending jobs sorted by priority and deadline
    async fn get_pending_jobs_sorted(&self) -> Result<Vec<ProofJobId>> {
        let mut pending_jobs: Vec<_> = self.proof_queue.iter()
            .filter(|entry| entry.value().status == ProofJobStatus::Queued)
            .map(|entry| entry.value().clone())
            .collect();
        
        // Sort by priority and deadline
        pending_jobs.sort_by(|a, b| {
            // First by priority
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }
            
            // Then by deadline
            match (a.deadline, b.deadline) {
                (Some(a_deadline), Some(b_deadline)) => a_deadline.cmp(&b_deadline),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.submitted_at.cmp(&b.submitted_at),
            }
        });
        
        Ok(pending_jobs.into_iter().map(|job| job.id).collect())
    }
    
    /// Scale prover infrastructure based on demand
    pub async fn scale_infrastructure(&self) -> Result<()> {
        let queue_size = self.proof_queue.len();
        let active_provers = self.active_provers.len();
        
        // Calculate scaling requirements
        let scaling_decision = self.calculate_scaling_requirements(queue_size, active_provers).await?;
        
        match scaling_decision {
            ScalingDecision::ScaleUp(target_provers) => {
                self.kubernetes_manager.scale_up_provers(target_provers).await?;
                tracing::info!("Scaling up to {} provers", target_provers);
            }
            ScalingDecision::ScaleDown(target_provers) => {
                self.kubernetes_manager.scale_down_provers(target_provers).await?;
                tracing::info!("Scaling down to {} provers", target_provers);
            }
            ScalingDecision::NoChange => {
                tracing::debug!("No scaling required");
            }
        }
        
        Ok(())
    }
    
    /// Calculate scaling requirements
    async fn calculate_scaling_requirements(
        &self,
        queue_size: usize,
        active_provers: usize,
    ) -> Result<ScalingDecision> {
        let target_queue_size = self.config.target_queue_size;
        let min_provers = self.config.min_provers;
        let max_provers = self.config.max_provers;
        
        if queue_size > target_queue_size && active_provers < max_provers {
            // Scale up
            let additional_provers = ((queue_size - target_queue_size) / 10).max(1);
            let target_provers = (active_provers + additional_provers).min(max_provers);
            Ok(ScalingDecision::ScaleUp(target_provers))
        } else if queue_size < target_queue_size / 2 && active_provers > min_provers {
            // Scale down
            let target_provers = (active_provers - 1).max(min_provers);
            Ok(ScalingDecision::ScaleDown(target_provers))
        } else {
            Ok(ScalingDecision::NoChange)
        }
    }
    
    /// Register new prover node
    pub async fn register_prover(&self, prover_info: ProverNodeInfo) -> Result<ProverId> {
        let prover_id = Uuid::new_v4();
        
        let prover_node = ProverNode {
            id: prover_id,
            validator_id: prover_info.validator_id,
            capabilities: prover_info.capabilities,
            resource_capacity: prover_info.resource_capacity,
            current_load: ResourceUsage::default(),
            status: ProverStatus::Available,
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            performance_history: Vec::new(),
        };
        
        self.active_provers.insert(prover_id, prover_node);
        self.metrics.increment_provers_registered();
        
        tracing::info!("Registered prover {} for validator {}", prover_id, prover_info.validator_id);
        
        Ok(prover_id)
    }
    
    /// Clone for async processing
    fn clone_for_async(&self) -> Self {
        // TODO: Implement proper cloning for async context
        // This is a placeholder - in practice, you'd use Arc<Self> or similar
        unimplemented!("Async cloning not implemented")
    }
    
    /// Get proof job status
    pub fn get_job_status(&self, job_id: ProofJobId) -> Option<ProofJobStatus> {
        self.proof_queue.get(&job_id).map(|job| job.status.clone())
    }
    
    /// Get infrastructure statistics
    pub fn get_infrastructure_statistics(&self) -> InfrastructureStatistics {
        let active_provers = self.active_provers.len();
        let queued_jobs = self.proof_queue.iter()
            .filter(|entry| entry.value().status == ProofJobStatus::Queued)
            .count();
        let running_jobs = self.proof_queue.iter()
            .filter(|entry| entry.value().status == ProofJobStatus::Running)
            .count();
        
        InfrastructureStatistics {
            active_provers,
            queued_jobs,
            running_jobs,
            total_jobs: self.proof_queue.len(),
            average_job_time: self.metrics.get_average_proof_generation_time(),
            throughput: self.calculate_current_throughput(),
        }
    }
    
    /// Calculate current throughput
    fn calculate_current_throughput(&self) -> f64 {
        // TODO: Calculate actual throughput based on completed jobs
        0.0
    }
}
