//! Quantlink Qross Proof Aggregation Protocol
//! 
//! This module implements recursive proof composition system for batching multiple
//! cross-chain state transitions with coordination to consensus aggregator finality
//! determination and Byzantine fault tolerance.

pub mod aggregator;
pub mod dependency;
pub mod coordination;
pub mod batch;
pub mod finality;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use qross_zk_circuits::{ZkStarkProof, ZkStarkEngine, CircuitId};
use qross_consensus::ConsensusState;

pub use error::{AggregationError, Result};
pub use types::*;

/// Main proof aggregation engine coordinating with consensus layer
pub struct ProofAggregationEngine {
    zk_engine: ZkStarkEngine,
    dependency_manager: dependency::DependencyManager,
    batch_processor: batch::BatchProcessor,
    finality_coordinator: finality::FinalityCoordinator,
    consensus_coordinator: coordination::ConsensusCoordinator,
    metrics: metrics::AggregationMetrics,
    config: AggregationConfig,
    active_aggregations: dashmap::DashMap<AggregationId, AggregationSession>,
    proof_cache: dashmap::DashMap<ProofId, CachedAggregatedProof>,
}

/// Trait for consensus integration
#[async_trait]
pub trait ConsensusIntegration: Send + Sync {
    /// Get current consensus state
    async fn get_consensus_state(&self) -> Result<ConsensusState>;
    
    /// Check if emergency halt is active
    async fn is_emergency_halt_active(&self) -> Result<bool>;
    
    /// Submit aggregated proof for finality determination
    async fn submit_for_finality(&self, proof: &AggregatedProof) -> Result<FinalityStatus>;
    
    /// Get validator performance metrics for prover allocation
    async fn get_validator_performance(&self) -> Result<HashMap<ValidatorId, PerformanceMetrics>>;
    
    /// Report proof verification failure for slashing
    async fn report_invalid_proof(&self, proof_id: ProofId, validator_id: ValidatorId) -> Result<()>;
}

/// Trait for state synchronization
#[async_trait]
pub trait StateSynchronization: Send + Sync {
    /// Get cross-chain state dependencies
    async fn get_state_dependencies(&self, chain_id: &str, block_height: u64) -> Result<Vec<StateDependency>>;
    
    /// Verify state transition validity
    async fn verify_state_transition(&self, transition: &StateTransition) -> Result<bool>;
    
    /// Get latest finalized state for chain
    async fn get_latest_finalized_state(&self, chain_id: &str) -> Result<ChainState>;
}

impl ProofAggregationEngine {
    /// Create a new proof aggregation engine
    pub fn new(
        config: AggregationConfig,
        consensus_integration: Box<dyn ConsensusIntegration>,
        state_sync: Box<dyn StateSynchronization>,
    ) -> Self {
        let zk_engine = ZkStarkEngine::new(config.zk_config.clone());
        let dependency_manager = dependency::DependencyManager::new(config.dependency_config.clone());
        let batch_processor = batch::BatchProcessor::new(config.batch_config.clone());
        let finality_coordinator = finality::FinalityCoordinator::new(config.finality_config.clone());
        let consensus_coordinator = coordination::ConsensusCoordinator::new(
            consensus_integration,
            state_sync,
        );
        let metrics = metrics::AggregationMetrics::new();
        
        Self {
            zk_engine,
            dependency_manager,
            batch_processor,
            finality_coordinator,
            consensus_coordinator,
            metrics,
            config,
            active_aggregations: dashmap::DashMap::new(),
            proof_cache: dashmap::DashMap::new(),
        }
    }
    
    /// Submit proofs for aggregation with dependency analysis
    pub async fn submit_proofs_for_aggregation(
        &self,
        proofs: Vec<ProofSubmission>,
    ) -> Result<AggregationId> {
        let aggregation_id = Uuid::new_v4();
        let start_time = std::time::Instant::now();
        
        // Check emergency halt status
        if self.consensus_coordinator.is_emergency_halt_active().await? {
            return Err(AggregationError::EmergencyHaltActive);
        }
        
        // Validate proof submissions
        self.validate_proof_submissions(&proofs).await?;
        
        // Analyze dependencies between proofs
        let dependency_graph = self.dependency_manager.analyze_dependencies(&proofs).await?;
        
        // Create aggregation session
        let session = AggregationSession {
            id: aggregation_id,
            proofs: proofs.clone(),
            dependency_graph,
            status: AggregationStatus::Pending,
            created_at: Utc::now(),
            target_composition_depth: self.calculate_optimal_composition_depth(&proofs),
            assigned_validators: Vec::new(),
            progress: AggregationProgress::default(),
        };
        
        self.active_aggregations.insert(aggregation_id, session);
        
        // Start asynchronous aggregation process
        let engine_clone = self.clone_for_async();
        tokio::spawn(async move {
            if let Err(e) = engine_clone.process_aggregation(aggregation_id).await {
                tracing::error!("Aggregation {} failed: {}", aggregation_id, e);
            }
        });
        
        self.metrics.increment_aggregation_requests();
        self.metrics.record_submission_time(start_time.elapsed().as_secs_f64());
        
        tracing::info!(
            "Started proof aggregation {} with {} proofs",
            aggregation_id,
            proofs.len()
        );
        
        Ok(aggregation_id)
    }
    
    /// Process aggregation with consensus coordination
    async fn process_aggregation(&self, aggregation_id: AggregationId) -> Result<()> {
        let mut session = self.active_aggregations.get_mut(&aggregation_id)
            .ok_or_else(|| AggregationError::AggregationNotFound(aggregation_id))?;
        
        session.status = AggregationStatus::Processing;
        session.progress.started_at = Some(Utc::now());
        
        // Get optimal validator allocation for proof generation
        let validator_performance = self.consensus_coordinator.get_validator_performance().await?;
        let assigned_validators = self.allocate_validators_for_aggregation(
            &session.proofs,
            &validator_performance,
        )?;
        
        session.assigned_validators = assigned_validators;
        
        // Process dependency graph in topological order
        let processing_order = self.dependency_manager.get_processing_order(&session.dependency_graph)?;
        
        let mut aggregated_proofs = HashMap::new();
        
        for batch in processing_order {
            // Process batch of independent proofs
            let batch_result = self.batch_processor.process_batch(
                &batch,
                &session.proofs,
                &aggregated_proofs,
            ).await?;
            
            // Verify batch result
            if !self.verify_batch_result(&batch_result).await? {
                session.status = AggregationStatus::Failed;
                return Err(AggregationError::BatchVerificationFailed);
            }
            
            // Store intermediate results
            for (proof_id, aggregated_proof) in batch_result {
                aggregated_proofs.insert(proof_id, aggregated_proof);
            }
            
            // Update progress
            session.progress.completed_batches += 1;
            session.progress.total_batches = processing_order.len();
        }
        
        // Create final aggregated proof
        let final_proof = self.create_final_aggregated_proof(aggregated_proofs).await?;
        
        // Submit to consensus for finality determination
        let finality_status = self.consensus_coordinator.submit_for_finality(&final_proof).await?;
        
        // Update session status
        session.status = match finality_status {
            FinalityStatus::Finalized => AggregationStatus::Finalized,
            FinalityStatus::Pending => AggregationStatus::AwaitingFinality,
            FinalityStatus::Rejected => AggregationStatus::Failed,
        };
        
        session.progress.completed_at = Some(Utc::now());
        
        // Cache the result
        self.cache_aggregated_proof(&final_proof);
        
        // Update metrics
        self.metrics.increment_completed_aggregations();
        if let Some(started_at) = session.progress.started_at {
            let duration = (Utc::now() - started_at).num_seconds() as f64;
            self.metrics.record_aggregation_time(duration);
        }
        
        tracing::info!(
            "Completed aggregation {} with status {:?}",
            aggregation_id,
            session.status
        );
        
        Ok(())
    }
    
    /// Validate proof submissions for aggregation
    async fn validate_proof_submissions(&self, proofs: &[ProofSubmission]) -> Result<()> {
        if proofs.is_empty() {
            return Err(AggregationError::EmptyProofSet);
        }
        
        if proofs.len() > self.config.max_proofs_per_aggregation {
            return Err(AggregationError::TooManyProofs {
                submitted: proofs.len(),
                max_allowed: self.config.max_proofs_per_aggregation,
            });
        }
        
        // Validate each proof
        for proof_submission in proofs {
            // Check proof validity
            if !self.zk_engine.verify_proof(&proof_submission.proof).await? {
                return Err(AggregationError::InvalidProof(proof_submission.proof.id));
            }
            
            // Verify state transition if applicable
            if let Some(state_transition) = &proof_submission.state_transition {
                if !self.consensus_coordinator.verify_state_transition(state_transition).await? {
                    return Err(AggregationError::InvalidStateTransition(
                        state_transition.id
                    ));
                }
            }
            
            // Check cross-chain dependencies
            self.validate_cross_chain_dependencies(proof_submission).await?;
        }
        
        Ok(())
    }
    
    /// Validate cross-chain dependencies
    async fn validate_cross_chain_dependencies(&self, submission: &ProofSubmission) -> Result<()> {
        if let Some(state_transition) = &submission.state_transition {
            let dependencies = self.consensus_coordinator.get_state_dependencies(
                &state_transition.source_chain,
                state_transition.source_block_height,
            ).await?;
            
            // Check that all dependencies are satisfied
            for dependency in dependencies {
                if !self.is_dependency_satisfied(&dependency).await? {
                    return Err(AggregationError::UnsatisfiedDependency(dependency.id));
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if a state dependency is satisfied
    async fn is_dependency_satisfied(&self, dependency: &StateDependency) -> Result<bool> {
        let latest_state = self.consensus_coordinator.get_latest_finalized_state(
            &dependency.chain_id
        ).await?;
        
        Ok(latest_state.block_height >= dependency.required_block_height)
    }
    
    /// Calculate optimal composition depth for proof set
    fn calculate_optimal_composition_depth(&self, proofs: &[ProofSubmission]) -> usize {
        let proof_count = proofs.len();
        let max_depth = self.config.max_composition_depth;
        let aggregation_factor = self.config.aggregation_factor;
        
        // Calculate depth needed for logarithmic scaling
        let calculated_depth = (proof_count as f64).log(aggregation_factor as f64).ceil() as usize;
        
        calculated_depth.min(max_depth).max(1)
    }
    
    /// Allocate validators for aggregation based on performance
    fn allocate_validators_for_aggregation(
        &self,
        proofs: &[ProofSubmission],
        validator_performance: &HashMap<ValidatorId, PerformanceMetrics>,
    ) -> Result<Vec<ValidatorId>> {
        let required_validators = self.calculate_required_validators(proofs.len());
        
        // Sort validators by proof generation performance
        let mut sorted_validators: Vec<_> = validator_performance.iter()
            .filter(|(_, metrics)| metrics.proof_generation_score > 0.7) // Minimum performance threshold
            .collect();
        
        sorted_validators.sort_by(|a, b| {
            b.1.proof_generation_score.partial_cmp(&a.1.proof_generation_score).unwrap()
        });
        
        let allocated: Vec<ValidatorId> = sorted_validators.into_iter()
            .take(required_validators)
            .map(|(validator_id, _)| validator_id.clone())
            .collect();
        
        if allocated.len() < required_validators {
            return Err(AggregationError::InsufficientValidators {
                required: required_validators,
                available: allocated.len(),
            });
        }
        
        Ok(allocated)
    }
    
    /// Calculate required validators for aggregation
    fn calculate_required_validators(&self, proof_count: usize) -> usize {
        // Use 2f+1 model from consensus layer
        let base_requirement = 3; // Minimum for Byzantine fault tolerance
        let scaling_factor = (proof_count as f64).log2().ceil() as usize;
        
        (base_requirement + scaling_factor).min(self.config.max_validators_per_aggregation)
    }
    
    /// Verify batch processing result
    async fn verify_batch_result(
        &self,
        batch_result: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<bool> {
        for aggregated_proof in batch_result.values() {
            if !self.zk_engine.verify_proof(&aggregated_proof.proof).await? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Create final aggregated proof from intermediate results
    async fn create_final_aggregated_proof(
        &self,
        aggregated_proofs: HashMap<ProofId, AggregatedProof>,
    ) -> Result<AggregatedProof> {
        let proof_ids: Vec<ProofId> = aggregated_proofs.keys().cloned().collect();
        let proofs: Vec<ZkStarkProof> = aggregated_proofs.values()
            .map(|ap| ap.proof.clone())
            .collect();
        
        // Use recursive composition from zk-circuits
        let final_zk_proof = self.zk_engine.compose_recursive_proof(
            proofs,
            self.config.final_composition_circuit_id,
        ).await?;
        
        Ok(AggregatedProof {
            id: Uuid::new_v4(),
            component_proof_ids: proof_ids,
            proof: final_zk_proof,
            aggregation_metadata: AggregationMetadata {
                composition_depth: self.calculate_composition_depth(&aggregated_proofs),
                compression_ratio: self.calculate_compression_ratio(&aggregated_proofs),
                validator_signatures: Vec::new(), // TODO: Collect validator signatures
            },
            created_at: Utc::now(),
        })
    }
    
    /// Calculate composition depth
    fn calculate_composition_depth(&self, aggregated_proofs: &HashMap<ProofId, AggregatedProof>) -> usize {
        aggregated_proofs.values()
            .map(|ap| ap.aggregation_metadata.composition_depth)
            .max()
            .unwrap_or(0) + 1
    }
    
    /// Calculate compression ratio
    fn calculate_compression_ratio(&self, aggregated_proofs: &HashMap<ProofId, AggregatedProof>) -> f64 {
        let total_original_size: usize = aggregated_proofs.values()
            .map(|ap| ap.proof.proof_size)
            .sum();
        
        let final_size = aggregated_proofs.values()
            .map(|ap| ap.proof.proof_size)
            .max()
            .unwrap_or(1);
        
        total_original_size as f64 / final_size as f64
    }
    
    /// Cache aggregated proof
    fn cache_aggregated_proof(&self, proof: &AggregatedProof) {
        let cached_proof = CachedAggregatedProof {
            proof: proof.clone(),
            cached_at: Utc::now(),
            access_count: 0,
        };
        
        self.proof_cache.insert(proof.id, cached_proof);
        
        // Enforce cache size limit
        if self.proof_cache.len() > self.config.max_cached_proofs {
            self.evict_oldest_cached_proof();
        }
    }
    
    /// Evict oldest cached proof
    fn evict_oldest_cached_proof(&self) {
        if let Some(oldest_entry) = self.proof_cache.iter()
            .min_by_key(|entry| entry.value().cached_at) {
            let oldest_id = *oldest_entry.key();
            self.proof_cache.remove(&oldest_id);
        }
    }
    
    /// Clone for async processing
    fn clone_for_async(&self) -> Self {
        // TODO: Implement proper cloning for async context
        // This is a placeholder - in practice, you'd use Arc<Self> or similar
        unimplemented!("Async cloning not implemented")
    }
    
    /// Get aggregation status
    pub fn get_aggregation_status(&self, aggregation_id: AggregationId) -> Option<AggregationStatus> {
        self.active_aggregations.get(&aggregation_id)
            .map(|session| session.status.clone())
    }
    
    /// Get aggregation statistics
    pub fn get_aggregation_statistics(&self) -> AggregationStatistics {
        let active_aggregations = self.active_aggregations.len();
        let cached_proofs = self.proof_cache.len();
        
        AggregationStatistics {
            active_aggregations,
            cached_proofs,
            total_aggregations: self.metrics.get_total_aggregations(),
            average_aggregation_time: self.metrics.get_average_aggregation_time(),
            compression_ratio: self.calculate_average_compression_ratio(),
        }
    }
    
    /// Calculate average compression ratio
    fn calculate_average_compression_ratio(&self) -> f64 {
        let ratios: Vec<f64> = self.proof_cache.iter()
            .map(|entry| entry.value().proof.aggregation_metadata.compression_ratio)
            .collect();
        
        if ratios.is_empty() {
            1.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        }
    }
}
