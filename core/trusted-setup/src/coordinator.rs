//! Ceremony coordination with validator network integration

use crate::{types::*, error::*, ValidatorNetworkIntegration, ConsensusIntegration};
use std::collections::HashMap;
use qross_consensus::ValidatorId;

/// Ceremony coordinator managing validator selection and network integration
pub struct CeremonyCoordinator {
    config: CoordinatorConfig,
    validator_integration: Box<dyn ValidatorNetworkIntegration>,
    validator_cache: HashMap<ValidatorId, ValidatorInfo>,
    selection_history: Vec<SelectionRecord>,
}

/// Record of validator selection for ceremony
#[derive(Debug, Clone)]
pub struct SelectionRecord {
    pub ceremony_id: CeremonyId,
    pub ceremony_type: CeremonyType,
    pub selected_validators: Vec<ValidatorId>,
    pub selection_criteria: SelectionCriteria,
    pub selection_time: chrono::DateTime<chrono::Utc>,
    pub selection_duration: std::time::Duration,
}

/// Criteria used for validator selection
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    pub reputation_threshold: f64,
    pub performance_threshold: f64,
    pub geographic_distribution: bool,
    pub exclude_recent_participants: bool,
    pub strategy: ValidatorSelectionStrategy,
}

impl CeremonyCoordinator {
    /// Create a new ceremony coordinator
    pub fn new(
        config: CoordinatorConfig,
        validator_integration: Box<dyn ValidatorNetworkIntegration>,
    ) -> Self {
        Self {
            config,
            validator_integration,
            validator_cache: HashMap::new(),
            selection_history: Vec::new(),
        }
    }
    
    /// Select validators for ceremony participation
    pub async fn select_ceremony_validators(
        &mut self,
        required_participants: usize,
        ceremony_type: &CeremonyType,
    ) -> Result<Vec<ValidatorId>> {
        let start_time = std::time::Instant::now();
        
        // Refresh validator information
        self.refresh_validator_cache().await?;
        
        // Apply selection criteria
        let criteria = self.create_selection_criteria(ceremony_type);
        let eligible_validators = self.filter_eligible_validators(&criteria).await?;
        
        if eligible_validators.len() < required_participants {
            return Err(CeremonyError::InsufficientParticipants {
                required: required_participants,
                available: eligible_validators.len(),
            });
        }
        
        // Select validators based on strategy
        let selected_validators = self.apply_selection_strategy(
            &eligible_validators,
            required_participants,
            &criteria,
        ).await?;
        
        // Record selection
        let selection_record = SelectionRecord {
            ceremony_id: uuid::Uuid::new_v4(), // Will be updated with actual ceremony ID
            ceremony_type: ceremony_type.clone(),
            selected_validators: selected_validators.clone(),
            selection_criteria: criteria,
            selection_time: chrono::Utc::now(),
            selection_duration: start_time.elapsed(),
        };
        
        self.selection_history.push(selection_record);
        
        tracing::info!(
            "Selected {} validators for {:?} ceremony in {:.2}ms",
            selected_validators.len(),
            ceremony_type,
            start_time.elapsed().as_millis()
        );
        
        Ok(selected_validators)
    }
    
    /// Refresh validator cache with latest information
    async fn refresh_validator_cache(&mut self) -> Result<()> {
        let validator_set = self.validator_integration.get_validator_set().await
            .map_err(|e| CeremonyError::ValidatorNetworkIntegration(e.to_string()))?;
        
        self.validator_cache = validator_set;
        
        tracing::debug!("Refreshed validator cache with {} validators", self.validator_cache.len());
        
        Ok(())
    }
    
    /// Create selection criteria for ceremony type
    fn create_selection_criteria(&self, ceremony_type: &CeremonyType) -> SelectionCriteria {
        let mut criteria = SelectionCriteria {
            reputation_threshold: self.config.reputation_threshold,
            performance_threshold: self.config.performance_threshold,
            geographic_distribution: self.config.geographic_distribution,
            exclude_recent_participants: true,
            strategy: self.config.validator_selection_strategy.clone(),
        };
        
        // Adjust criteria based on ceremony type
        match ceremony_type {
            CeremonyType::Universal => {
                // Highest standards for universal ceremony
                criteria.reputation_threshold = criteria.reputation_threshold.max(0.8);
                criteria.performance_threshold = criteria.performance_threshold.max(0.8);
            }
            CeremonyType::CircuitSpecific { .. } => {
                // Standard criteria for circuit-specific ceremony
            }
            CeremonyType::Update { .. } => {
                // Slightly relaxed criteria for update ceremony
                criteria.reputation_threshold *= 0.9;
                criteria.performance_threshold *= 0.9;
            }
        }
        
        criteria
    }
    
    /// Filter validators based on eligibility criteria
    async fn filter_eligible_validators(&self, criteria: &SelectionCriteria) -> Result<Vec<ValidatorId>> {
        let mut eligible_validators = Vec::new();
        
        for (validator_id, validator_info) in &self.validator_cache {
            // Check reputation threshold
            if validator_info.reputation_score < criteria.reputation_threshold {
                continue;
            }
            
            // Check performance threshold
            let performance_score = self.calculate_performance_score(&validator_info.performance_metrics);
            if performance_score < criteria.performance_threshold {
                continue;
            }
            
            // Check recent participation exclusion
            if criteria.exclude_recent_participants && self.has_recent_participation(validator_id) {
                continue;
            }
            
            // Check emergency halt status
            if self.validator_integration.is_emergency_halt_active().await
                .map_err(|e| CeremonyError::ValidatorNetworkIntegration(e.to_string()))? {
                continue;
            }
            
            eligible_validators.push(validator_id.clone());
        }
        
        Ok(eligible_validators)
    }
    
    /// Calculate overall performance score for validator
    fn calculate_performance_score(&self, metrics: &ValidatorPerformanceMetrics) -> f64 {
        // Weighted average of performance metrics
        let weights = [0.3, 0.2, 0.2, 0.2, 0.1]; // uptime, response_time, computation, network, ceremony_participation
        let scores = [
            metrics.uptime_percentage,
            1.0 - (metrics.response_time_ms / 1000.0).min(1.0), // Invert response time
            metrics.computation_score,
            metrics.network_reliability,
            metrics.ceremony_participation_rate,
        ];
        
        weights.iter().zip(scores.iter())
            .map(|(weight, score)| weight * score)
            .sum()
    }
    
    /// Check if validator has participated in recent ceremonies
    fn has_recent_participation(&self, validator_id: &ValidatorId) -> bool {
        let recent_threshold = chrono::Utc::now() - chrono::Duration::days(7);
        
        self.selection_history.iter()
            .filter(|record| record.selection_time > recent_threshold)
            .any(|record| record.selected_validators.contains(validator_id))
    }
    
    /// Apply selection strategy to choose final validators
    async fn apply_selection_strategy(
        &self,
        eligible_validators: &[ValidatorId],
        required_participants: usize,
        criteria: &SelectionCriteria,
    ) -> Result<Vec<ValidatorId>> {
        match criteria.strategy {
            ValidatorSelectionStrategy::ReputationBased => {
                self.select_by_reputation(eligible_validators, required_participants).await
            }
            ValidatorSelectionStrategy::PerformanceBased => {
                self.select_by_performance(eligible_validators, required_participants).await
            }
            ValidatorSelectionStrategy::Random => {
                self.select_randomly(eligible_validators, required_participants).await
            }
            ValidatorSelectionStrategy::Hybrid => {
                self.select_hybrid(eligible_validators, required_participants, criteria).await
            }
        }
    }
    
    /// Select validators based on reputation scores
    async fn select_by_reputation(
        &self,
        eligible_validators: &[ValidatorId],
        required_participants: usize,
    ) -> Result<Vec<ValidatorId>> {
        let mut scored_validators: Vec<_> = eligible_validators.iter()
            .filter_map(|id| {
                self.validator_cache.get(id).map(|info| (id.clone(), info.reputation_score))
            })
            .collect();
        
        scored_validators.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(scored_validators.into_iter()
            .take(required_participants)
            .map(|(id, _)| id)
            .collect())
    }
    
    /// Select validators based on performance metrics
    async fn select_by_performance(
        &self,
        eligible_validators: &[ValidatorId],
        required_participants: usize,
    ) -> Result<Vec<ValidatorId>> {
        let mut scored_validators: Vec<_> = eligible_validators.iter()
            .filter_map(|id| {
                self.validator_cache.get(id).map(|info| {
                    let score = self.calculate_performance_score(&info.performance_metrics);
                    (id.clone(), score)
                })
            })
            .collect();
        
        scored_validators.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(scored_validators.into_iter()
            .take(required_participants)
            .map(|(id, _)| id)
            .collect())
    }
    
    /// Select validators randomly
    async fn select_randomly(
        &self,
        eligible_validators: &[ValidatorId],
        required_participants: usize,
    ) -> Result<Vec<ValidatorId>> {
        use rand::seq::SliceRandom;
        
        let mut rng = rand::thread_rng();
        let mut validators = eligible_validators.to_vec();
        validators.shuffle(&mut rng);
        
        Ok(validators.into_iter().take(required_participants).collect())
    }
    
    /// Select validators using hybrid approach
    async fn select_hybrid(
        &self,
        eligible_validators: &[ValidatorId],
        required_participants: usize,
        criteria: &SelectionCriteria,
    ) -> Result<Vec<ValidatorId>> {
        let reputation_weight = 0.4;
        let performance_weight = 0.4;
        let randomness_weight = 0.2;
        
        let mut scored_validators: Vec<_> = eligible_validators.iter()
            .filter_map(|id| {
                self.validator_cache.get(id).map(|info| {
                    let reputation_score = info.reputation_score;
                    let performance_score = self.calculate_performance_score(&info.performance_metrics);
                    let random_score: f64 = rand::random();
                    
                    let combined_score = reputation_score * reputation_weight +
                                       performance_score * performance_weight +
                                       random_score * randomness_weight;
                    
                    (id.clone(), combined_score)
                })
            })
            .collect();
        
        scored_validators.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut selected = scored_validators.into_iter()
            .take(required_participants)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        
        // Apply geographic distribution if required
        if criteria.geographic_distribution {
            selected = self.apply_geographic_distribution(selected).await?;
        }
        
        Ok(selected)
    }
    
    /// Apply geographic distribution to validator selection
    async fn apply_geographic_distribution(&self, mut validators: Vec<ValidatorId>) -> Result<Vec<ValidatorId>> {
        // TODO: Implement geographic distribution logic
        // This would analyze validator network addresses and ensure geographic diversity
        // For now, return validators as-is
        Ok(validators)
    }
    
    /// Report invalid ceremony participation for slashing
    pub async fn report_invalid_participation(
        &self,
        validator_id: &ValidatorId,
        violation: CeremonyViolation,
    ) -> Result<()> {
        self.validator_integration.report_invalid_participation(validator_id, violation).await
            .map_err(|e| CeremonyError::ValidatorNetworkIntegration(e.to_string()))
    }
    
    /// Check if emergency halt is active
    pub async fn is_emergency_halt_active(&self) -> Result<bool> {
        self.validator_integration.is_emergency_halt_active().await
            .map_err(|e| CeremonyError::ValidatorNetworkIntegration(e.to_string()))
    }
    
    /// Trigger emergency halt
    pub async fn trigger_emergency_halt(&self, reason: String) -> Result<()> {
        self.validator_integration.trigger_emergency_halt(reason).await
            .map_err(|e| CeremonyError::ValidatorNetworkIntegration(e.to_string()))
    }
    
    /// Submit ceremony parameters for consensus validation
    pub async fn submit_ceremony_parameters(&self, parameters: &VerifiedParameters) -> Result<bool> {
        // TODO: Implement consensus integration
        // For now, return true as placeholder
        Ok(true)
    }
    
    /// Get validator selection statistics
    pub fn get_selection_statistics(&self) -> SelectionStatistics {
        let total_selections = self.selection_history.len();
        let average_selection_time = if total_selections > 0 {
            self.selection_history.iter()
                .map(|record| record.selection_duration.as_secs_f64())
                .sum::<f64>() / total_selections as f64
        } else {
            0.0
        };
        
        let cached_validators = self.validator_cache.len();
        
        SelectionStatistics {
            total_selections,
            cached_validators,
            average_selection_time,
            recent_selections: self.selection_history.iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
        }
    }
    
    /// Clear selection history
    pub fn clear_selection_history(&mut self) {
        self.selection_history.clear();
    }
    
    /// Get validator information
    pub fn get_validator_info(&self, validator_id: &ValidatorId) -> Option<&ValidatorInfo> {
        self.validator_cache.get(validator_id)
    }
    
    /// Update validator performance metrics
    pub async fn update_validator_performance(
        &mut self,
        validator_id: &ValidatorId,
        metrics: ValidatorPerformanceMetrics,
    ) -> Result<()> {
        if let Some(validator_info) = self.validator_cache.get_mut(validator_id) {
            validator_info.performance_metrics = metrics;
        }
        
        Ok(())
    }
}

/// Selection statistics
#[derive(Debug, Clone)]
pub struct SelectionStatistics {
    pub total_selections: usize,
    pub cached_validators: usize,
    pub average_selection_time: f64,
    pub recent_selections: Vec<SelectionRecord>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    
    struct MockValidatorIntegration;
    
    #[async_trait]
    impl ValidatorNetworkIntegration for MockValidatorIntegration {
        async fn get_validator_set(&self) -> Result<HashMap<ValidatorId, ValidatorInfo>> {
            let mut validators = HashMap::new();
            
            for i in 1..=10 {
                let validator_id = format!("validator_{}", i);
                let validator_info = ValidatorInfo {
                    validator_id: validator_id.clone(),
                    reputation_score: 0.8 + (i as f64 * 0.02),
                    performance_metrics: ValidatorPerformanceMetrics {
                        uptime_percentage: 0.95,
                        response_time_ms: 100.0,
                        computation_score: 0.9,
                        network_reliability: 0.95,
                        ceremony_participation_rate: 0.8,
                    },
                    ceremony_history: CeremonyHistory {
                        total_ceremonies: 5,
                        successful_ceremonies: 5,
                        failed_ceremonies: 0,
                        average_contribution_time: 30.0,
                        last_participation: None,
                    },
                    public_key: vec![0u8; 32],
                    network_address: format!("127.0.0.1:800{}", i),
                };
                
                validators.insert(validator_id, validator_info);
            }
            
            Ok(validators)
        }
        
        async fn get_validator_performance(&self, _validator_id: &ValidatorId) -> Result<qross_validator::PerformanceMetrics> {
            // TODO: Return mock performance metrics
            unimplemented!()
        }
        
        async fn report_invalid_participation(&self, _validator_id: &ValidatorId, _violation: CeremonyViolation) -> Result<()> {
            Ok(())
        }
        
        async fn is_emergency_halt_active(&self) -> Result<bool> {
            Ok(false)
        }
        
        async fn trigger_emergency_halt(&self, _reason: String) -> Result<()> {
            Ok(())
        }
    }
    
    #[tokio::test]
    async fn test_validator_selection() {
        let config = CoordinatorConfig::default();
        let integration = Box::new(MockValidatorIntegration);
        let mut coordinator = CeremonyCoordinator::new(config, integration);
        
        let ceremony_type = CeremonyType::Universal;
        let selected = coordinator.select_ceremony_validators(5, &ceremony_type).await.unwrap();
        
        assert_eq!(selected.len(), 5);
    }
}
