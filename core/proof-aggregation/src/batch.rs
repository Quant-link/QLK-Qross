//! Batch processing for proof aggregation

use crate::{types::*, error::*};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use rayon::prelude::*;

/// Batch processor for parallel proof aggregation
pub struct BatchProcessor {
    config: BatchConfig,
    processing_stats: BatchProcessingStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchProcessingStats {
    pub total_batches_processed: usize,
    pub average_batch_size: f64,
    pub average_processing_time: f64,
    pub success_rate: f64,
    pub parallel_efficiency: f64,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            processing_stats: BatchProcessingStats::default(),
        }
    }
    
    /// Process a batch of proofs with parallel optimization
    pub async fn process_batch(
        &mut self,
        batch: &[ProofId],
        proof_submissions: &[ProofSubmission],
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<BatchResult> {
        let start_time = std::time::Instant::now();
        
        if batch.is_empty() {
            return Ok(HashMap::new());
        }
        
        if batch.len() > self.config.max_batch_size {
            return Err(AggregationError::TooManyProofs {
                submitted: batch.len(),
                max_allowed: self.config.max_batch_size,
            });
        }
        
        // Collect proofs for this batch
        let batch_proofs = self.collect_batch_proofs(batch, proof_submissions)?;
        
        // Process batch with timeout
        let batch_timeout = Duration::from_secs(self.config.batch_timeout);
        let result = timeout(batch_timeout, self.process_batch_internal(batch_proofs, existing_aggregations)).await
            .map_err(|_| AggregationError::BatchTimeout)?;
        
        // Update statistics
        self.update_processing_stats(batch.len(), start_time.elapsed(), result.is_ok());
        
        result
    }
    
    /// Internal batch processing logic
    async fn process_batch_internal(
        &self,
        batch_proofs: Vec<ProofSubmission>,
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<BatchResult> {
        let mut batch_result = HashMap::new();
        
        if self.config.parallel_batch_processing && batch_proofs.len() > 1 {
            // Parallel processing for independent proofs
            batch_result = self.process_batch_parallel(batch_proofs, existing_aggregations).await?;
        } else {
            // Sequential processing
            batch_result = self.process_batch_sequential(batch_proofs, existing_aggregations).await?;
        }
        
        // Verify batch result integrity
        self.verify_batch_integrity(&batch_result).await?;
        
        Ok(batch_result)
    }
    
    /// Process batch in parallel
    async fn process_batch_parallel(
        &self,
        batch_proofs: Vec<ProofSubmission>,
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<BatchResult> {
        let chunk_size = self.calculate_optimal_chunk_size(batch_proofs.len());
        let chunks: Vec<_> = batch_proofs.chunks(chunk_size).collect();
        
        // Process chunks in parallel
        let chunk_results: Result<Vec<_>> = futures::future::try_join_all(
            chunks.into_iter().map(|chunk| {
                self.process_proof_chunk(chunk.to_vec(), existing_aggregations)
            })
        ).await;
        
        // Combine results
        let mut combined_result = HashMap::new();
        for chunk_result in chunk_results? {
            combined_result.extend(chunk_result);
        }
        
        Ok(combined_result)
    }
    
    /// Process batch sequentially
    async fn process_batch_sequential(
        &self,
        batch_proofs: Vec<ProofSubmission>,
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<BatchResult> {
        let mut batch_result = HashMap::new();
        
        for proof_submission in batch_proofs {
            let aggregated_proof = self.process_single_proof(proof_submission, existing_aggregations).await?;
            batch_result.insert(aggregated_proof.id, aggregated_proof);
        }
        
        Ok(batch_result)
    }
    
    /// Process a chunk of proofs
    async fn process_proof_chunk(
        &self,
        chunk: Vec<ProofSubmission>,
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<BatchResult> {
        let mut chunk_result = HashMap::new();
        
        for proof_submission in chunk {
            let aggregated_proof = self.process_single_proof(proof_submission, existing_aggregations).await?;
            chunk_result.insert(aggregated_proof.id, aggregated_proof);
        }
        
        Ok(chunk_result)
    }
    
    /// Process a single proof submission
    async fn process_single_proof(
        &self,
        proof_submission: ProofSubmission,
        existing_aggregations: &HashMap<ProofId, AggregatedProof>,
    ) -> Result<AggregatedProof> {
        // Check if proof is already aggregated
        if let Some(existing) = existing_aggregations.get(&proof_submission.proof.id) {
            return Ok(existing.clone());
        }
        
        // Create new aggregated proof
        let aggregated_proof = AggregatedProof {
            id: uuid::Uuid::new_v4(),
            component_proof_ids: vec![proof_submission.proof.id],
            proof: proof_submission.proof.clone(),
            aggregation_metadata: AggregationMetadata {
                composition_depth: 1,
                compression_ratio: 1.0,
                validator_signatures: Vec::new(),
            },
            created_at: chrono::Utc::now(),
        };
        
        Ok(aggregated_proof)
    }
    
    /// Collect proofs for batch processing
    fn collect_batch_proofs(
        &self,
        batch: &[ProofId],
        proof_submissions: &[ProofSubmission],
    ) -> Result<Vec<ProofSubmission>> {
        let mut batch_proofs = Vec::new();
        
        for proof_id in batch {
            let proof_submission = proof_submissions.iter()
                .find(|submission| submission.proof.id == *proof_id)
                .ok_or_else(|| AggregationError::InvalidProof(*proof_id))?;
            
            batch_proofs.push(proof_submission.clone());
        }
        
        Ok(batch_proofs)
    }
    
    /// Calculate optimal chunk size for parallel processing
    fn calculate_optimal_chunk_size(&self, total_proofs: usize) -> usize {
        let cpu_count = num_cpus::get();
        let base_chunk_size = total_proofs / cpu_count;
        
        // Ensure minimum chunk size for efficiency
        base_chunk_size.max(1).min(self.config.max_batch_size / 4)
    }
    
    /// Verify batch result integrity
    async fn verify_batch_integrity(&self, batch_result: &BatchResult) -> Result<()> {
        // Check for duplicate proof IDs
        let mut seen_ids = std::collections::HashSet::new();
        for aggregated_proof in batch_result.values() {
            for component_id in &aggregated_proof.component_proof_ids {
                if !seen_ids.insert(*component_id) {
                    return Err(AggregationError::Internal(
                        format!("Duplicate proof ID in batch: {}", component_id)
                    ));
                }
            }
        }
        
        // Verify proof consistency
        for aggregated_proof in batch_result.values() {
            if aggregated_proof.component_proof_ids.is_empty() {
                return Err(AggregationError::Internal(
                    "Aggregated proof has no component proofs".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Update processing statistics
    fn update_processing_stats(
        &mut self,
        batch_size: usize,
        processing_time: std::time::Duration,
        success: bool,
    ) {
        self.processing_stats.total_batches_processed += 1;
        
        // Update average batch size
        let total_batches = self.processing_stats.total_batches_processed as f64;
        self.processing_stats.average_batch_size = 
            (self.processing_stats.average_batch_size * (total_batches - 1.0) + batch_size as f64) / total_batches;
        
        // Update average processing time
        let processing_time_secs = processing_time.as_secs_f64();
        self.processing_stats.average_processing_time = 
            (self.processing_stats.average_processing_time * (total_batches - 1.0) + processing_time_secs) / total_batches;
        
        // Update success rate
        let success_count = if success { 1.0 } else { 0.0 };
        self.processing_stats.success_rate = 
            (self.processing_stats.success_rate * (total_batches - 1.0) + success_count) / total_batches;
    }
    
    /// Optimize batch configuration based on performance
    pub fn optimize_batch_config(&mut self) -> Result<()> {
        if self.processing_stats.total_batches_processed < 10 {
            return Ok(()); // Need more data for optimization
        }
        
        // Adjust batch size based on success rate
        if self.processing_stats.success_rate < 0.9 {
            // Reduce batch size if success rate is low
            self.config.max_batch_size = (self.config.max_batch_size * 3) / 4;
        } else if self.processing_stats.success_rate > 0.95 && 
                  self.processing_stats.average_processing_time < 30.0 {
            // Increase batch size if performance is good
            self.config.max_batch_size = (self.config.max_batch_size * 5) / 4;
        }
        
        // Adjust timeout based on average processing time
        let target_timeout = (self.processing_stats.average_processing_time * 2.0) as u64;
        self.config.batch_timeout = target_timeout.max(60).min(600); // Between 1 and 10 minutes
        
        tracing::info!(
            "Optimized batch config: max_size={}, timeout={}s",
            self.config.max_batch_size,
            self.config.batch_timeout
        );
        
        Ok(())
    }
    
    /// Get batch processing statistics
    pub fn get_statistics(&self) -> BatchProcessingStats {
        self.processing_stats.clone()
    }
    
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.processing_stats = BatchProcessingStats::default();
    }
    
    /// Estimate batch processing time
    pub fn estimate_processing_time(&self, batch_size: usize) -> std::time::Duration {
        if self.processing_stats.total_batches_processed == 0 {
            // Default estimate for new processor
            return std::time::Duration::from_secs(batch_size as u64 * 2);
        }
        
        // Scale based on historical data
        let base_time = self.processing_stats.average_processing_time;
        let size_factor = batch_size as f64 / self.processing_stats.average_batch_size;
        
        std::time::Duration::from_secs_f64(base_time * size_factor)
    }
    
    /// Check if batch size is optimal
    pub fn is_batch_size_optimal(&self, batch_size: usize) -> bool {
        if self.processing_stats.total_batches_processed < 5 {
            return true; // Not enough data to determine
        }
        
        let optimal_size = self.processing_stats.average_batch_size as usize;
        let tolerance = optimal_size / 4; // 25% tolerance
        
        batch_size >= optimal_size.saturating_sub(tolerance) && 
        batch_size <= optimal_size + tolerance
    }
    
    /// Get recommended batch size
    pub fn get_recommended_batch_size(&self) -> usize {
        if self.processing_stats.total_batches_processed < 5 {
            return self.config.max_batch_size / 2; // Conservative default
        }
        
        self.processing_stats.average_batch_size as usize
    }
}

/// Batch optimization strategies
#[derive(Debug, Clone)]
pub enum BatchOptimizationStrategy {
    /// Optimize for throughput
    Throughput,
    /// Optimize for latency
    Latency,
    /// Optimize for resource efficiency
    ResourceEfficiency,
    /// Balanced optimization
    Balanced,
}

impl BatchProcessor {
    /// Apply optimization strategy
    pub fn apply_optimization_strategy(&mut self, strategy: BatchOptimizationStrategy) {
        match strategy {
            BatchOptimizationStrategy::Throughput => {
                self.config.max_batch_size = self.config.max_batch_size * 2;
                self.config.batch_timeout = self.config.batch_timeout * 2;
                self.config.parallel_batch_processing = true;
            }
            BatchOptimizationStrategy::Latency => {
                self.config.max_batch_size = self.config.max_batch_size / 2;
                self.config.batch_timeout = self.config.batch_timeout / 2;
                self.config.parallel_batch_processing = true;
            }
            BatchOptimizationStrategy::ResourceEfficiency => {
                self.config.max_batch_size = num_cpus::get() * 2;
                self.config.parallel_batch_processing = false;
            }
            BatchOptimizationStrategy::Balanced => {
                // Keep current configuration as balanced default
            }
        }
        
        tracing::info!("Applied optimization strategy: {:?}", strategy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qross_zk_circuits::{ZkStarkProof, CircuitInputs};
    
    #[tokio::test]
    async fn test_batch_processing() {
        let config = BatchConfig::default();
        let mut processor = BatchProcessor::new(config);
        
        // Create test proof submissions
        let proof_submissions = vec![
            create_test_proof_submission(),
        ];
        
        let batch = vec![proof_submissions[0].proof.id];
        let existing_aggregations = HashMap::new();
        
        let result = processor.process_batch(&batch, &proof_submissions, &existing_aggregations).await;
        assert!(result.is_ok());
        
        let batch_result = result.unwrap();
        assert_eq!(batch_result.len(), 1);
    }
    
    fn create_test_proof_submission() -> ProofSubmission {
        ProofSubmission {
            proof: ZkStarkProof {
                id: uuid::Uuid::new_v4(),
                circuit_id: 1,
                stark_proof: winterfell::StarkProof::new_dummy(),
                inputs: CircuitInputs {
                    public_inputs: Vec::new(),
                    private_inputs: Vec::new(),
                    auxiliary_inputs: std::collections::HashMap::new(),
                },
                options: winterfell::ProofOptions::default(),
                generated_at: chrono::Utc::now(),
                generation_time: std::time::Duration::from_secs(1),
                proof_size: 1024,
            },
            state_transition: None,
            priority: AggregationPriority::Normal,
            dependencies: Vec::new(),
            submitted_by: "validator1".to_string(),
            submitted_at: chrono::Utc::now(),
            deadline: None,
        }
    }
}
