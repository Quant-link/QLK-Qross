//! Prover orchestration and validator integration

use crate::{types::*, error::*, ValidatorPerformanceIntegration, ProofAggregationIntegration};
use std::collections::HashMap;
use qross_consensus::ValidatorId;
use qross_zk_circuits::ZkStarkProof;

/// Prover orchestrator managing validator integration and proof coordination
pub struct ProverOrchestrator {
    config: OrchestratorConfig,
    validator_integration: Box<dyn ValidatorPerformanceIntegration>,
    aggregation_integration: Box<dyn ProofAggregationIntegration>,
    validator_performance_cache: HashMap<ValidatorId, ValidatorPerformanceMetrics>,
    prover_assignments: HashMap<ProverId, ValidatorId>,
}

impl ProverOrchestrator {
    /// Create a new prover orchestrator
    pub fn new(
        config: OrchestratorConfig,
        validator_integration: Box<dyn ValidatorPerformanceIntegration>,
    ) -> Self {
        Self {
            config,
            validator_integration,
            aggregation_integration: Box::new(MockAggregationIntegration), // TODO: Replace with actual integration
            validator_performance_cache: HashMap::new(),
            prover_assignments: HashMap::new(),
        }
    }
    
    /// Get available provers based on validator performance
    pub async fn get_available_provers(&mut self) -> Result<Vec<ProverId>> {
        // Refresh validator performance cache
        self.refresh_performance_cache().await?;
        
        // Filter provers based on performance thresholds
        let available_provers: Vec<ProverId> = self.prover_assignments.iter()
            .filter_map(|(prover_id, validator_id)| {
                if let Some(metrics) = self.validator_performance_cache.get(validator_id) {
                    if self.meets_performance_threshold(metrics) {
                        Some(*prover_id)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        Ok(available_provers)
    }
    
    /// Refresh validator performance cache
    async fn refresh_performance_cache(&mut self) -> Result<()> {
        let all_performance = self.validator_integration.get_all_validator_performance().await
            .map_err(|e| ProverError::Internal(e.to_string()))?;
        
        self.validator_performance_cache = all_performance;
        
        tracing::debug!("Refreshed performance cache for {} validators", self.validator_performance_cache.len());
        
        Ok(())
    }
    
    /// Check if validator meets performance threshold
    fn meets_performance_threshold(&self, metrics: &ValidatorPerformanceMetrics) -> bool {
        let weighted_score = 
            metrics.proof_generation_score * self.config.validator_performance_weight +
            metrics.reliability_score * self.config.reliability_weight +
            metrics.resource_efficiency_score * self.config.resource_efficiency_weight;
        
        weighted_score >= 0.7 // Minimum threshold
    }
    
    /// Submit proof for aggregation
    pub async fn submit_proof_for_aggregation(
        &self,
        proof: &ZkStarkProof,
        metadata: &ProofMetadata,
    ) -> Result<qross_proof_aggregation::ProofId> {
        self.aggregation_integration.submit_proof_for_aggregation(proof, metadata).await
            .map_err(|e| ProverError::Internal(e.to_string()))
    }
    
    /// Update validator performance based on proof generation
    pub async fn update_validator_performance(
        &self,
        validator_id: &ValidatorId,
        performance_data: &ProofGenerationPerformance,
    ) -> Result<()> {
        self.validator_integration.update_proof_generation_performance(validator_id, performance_data).await
            .map_err(|e| ProverError::Internal(e.to_string()))
    }
    
    /// Assign prover to validator
    pub fn assign_prover_to_validator(&mut self, prover_id: ProverId, validator_id: ValidatorId) {
        self.prover_assignments.insert(prover_id, validator_id);
        tracing::info!("Assigned prover {} to validator {}", prover_id, validator_id);
    }
    
    /// Remove prover assignment
    pub fn remove_prover_assignment(&mut self, prover_id: ProverId) {
        if let Some(validator_id) = self.prover_assignments.remove(&prover_id) {
            tracing::info!("Removed assignment of prover {} from validator {}", prover_id, validator_id);
        }
    }
    
    /// Get validator for prover
    pub fn get_validator_for_prover(&self, prover_id: ProverId) -> Option<&ValidatorId> {
        self.prover_assignments.get(&prover_id)
    }
    
    /// Get orchestration statistics
    pub fn get_orchestration_statistics(&self) -> OrchestrationStatistics {
        let total_assignments = self.prover_assignments.len();
        let active_validators = self.validator_performance_cache.len();
        
        let average_performance = if !self.validator_performance_cache.is_empty() {
            self.validator_performance_cache.values()
                .map(|metrics| metrics.proof_generation_score)
                .sum::<f64>() / self.validator_performance_cache.len() as f64
        } else {
            0.0
        };
        
        OrchestrationStatistics {
            total_assignments,
            active_validators,
            average_performance,
            performance_threshold: 0.7,
        }
    }
}

/// Mock aggregation integration for testing
struct MockAggregationIntegration;

#[async_trait::async_trait]
impl ProofAggregationIntegration for MockAggregationIntegration {
    async fn submit_proof_for_aggregation(
        &self,
        _proof: &ZkStarkProof,
        _metadata: &ProofMetadata,
    ) -> Result<qross_proof_aggregation::ProofId> {
        Ok(uuid::Uuid::new_v4())
    }
    
    async fn get_batch_requirements(&self) -> Result<BatchRequirements> {
        Ok(BatchRequirements {
            preferred_batch_size: 32,
            max_batch_size: 64,
            batch_timeout: std::time::Duration::from_secs(300),
            priority_grouping: true,
        })
    }
    
    async fn coordinate_batch_scheduling(&self, _batch_info: &BatchInfo) -> Result<()> {
        Ok(())
    }
}

/// Orchestration statistics
#[derive(Debug, Clone)]
pub struct OrchestrationStatistics {
    pub total_assignments: usize,
    pub active_validators: usize,
    pub average_performance: f64,
    pub performance_threshold: f64,
}
