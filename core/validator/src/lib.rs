//! Quantlink Qross Validator Selection Protocol
//! 
//! This module implements a sophisticated validator selection algorithm based on
//! reputation scoring, performance metrics, and stake-weighted selection to ensure
//! optimal network security and performance.

pub mod selection;
pub mod reputation;
pub mod performance;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub use error::{ValidatorError, Result};
pub use types::*;

/// Trait for validator data providers
#[async_trait]
pub trait ValidatorDataProvider: Send + Sync {
    /// Get validator information
    async fn get_validator(&self, validator_id: &ValidatorId) -> Result<Option<ValidatorInfo>>;
    
    /// Get all active validators
    async fn get_active_validators(&self) -> Result<Vec<ValidatorInfo>>;
    
    /// Get validator stake
    async fn get_validator_stake(&self, validator_id: &ValidatorId) -> Result<Stake>;
    
    /// Get validator performance history
    async fn get_performance_history(&self, validator_id: &ValidatorId, epochs: u64) -> Result<Vec<PerformanceRecord>>;
    
    /// Get validator slashing history
    async fn get_slashing_history(&self, validator_id: &ValidatorId) -> Result<Vec<SlashingRecord>>;
}

/// Trait for validator selection strategies
#[async_trait]
pub trait SelectionStrategy: Send + Sync {
    /// Select validators for the next epoch
    async fn select_validators(
        &self,
        candidates: &[ValidatorCandidate],
        target_count: usize,
        config: &SelectionConfig,
    ) -> Result<Vec<SelectedValidator>>;
    
    /// Get strategy name
    fn get_name(&self) -> &'static str;
    
    /// Get strategy parameters
    fn get_parameters(&self) -> HashMap<String, f64>;
}

/// Main validator selection engine
pub struct ValidatorSelectionEngine {
    data_provider: Box<dyn ValidatorDataProvider>,
    reputation_engine: reputation::ReputationEngine,
    performance_analyzer: performance::PerformanceAnalyzer,
    selection_strategy: Box<dyn SelectionStrategy>,
    config: SelectionConfig,
    metrics: metrics::SelectionMetrics,
    selection_history: Vec<SelectionEvent>,
}

impl ValidatorSelectionEngine {
    /// Create a new validator selection engine
    pub fn new(
        data_provider: Box<dyn ValidatorDataProvider>,
        selection_strategy: Box<dyn SelectionStrategy>,
        config: SelectionConfig,
    ) -> Self {
        Self {
            data_provider,
            reputation_engine: reputation::ReputationEngine::new(config.reputation_config.clone()),
            performance_analyzer: performance::PerformanceAnalyzer::new(config.performance_config.clone()),
            selection_strategy,
            config,
            metrics: metrics::SelectionMetrics::new(),
            selection_history: Vec::new(),
        }
    }
    
    /// Select validators for the next epoch
    pub async fn select_validators_for_epoch(&mut self, epoch: u64, target_count: usize) -> Result<SelectionResult> {
        let start_time = std::time::Instant::now();
        
        // Get all active validators
        let active_validators = self.data_provider.get_active_validators().await?;
        
        if active_validators.len() < target_count {
            return Err(ValidatorError::InsufficientValidators {
                available: active_validators.len(),
                required: target_count,
            });
        }
        
        // Build validator candidates with comprehensive scoring
        let mut candidates = Vec::new();
        for validator in active_validators {
            match self.build_validator_candidate(&validator).await {
                Ok(candidate) => candidates.push(candidate),
                Err(e) => {
                    tracing::warn!("Failed to build candidate for validator {}: {}", validator.id, e);
                    self.metrics.increment_candidate_build_failures();
                }
            }
        }
        
        // Apply pre-selection filters
        let filtered_candidates = self.apply_pre_selection_filters(&candidates)?;
        
        if filtered_candidates.len() < target_count {
            return Err(ValidatorError::InsufficientValidators {
                available: filtered_candidates.len(),
                required: target_count,
            });
        }
        
        // Perform selection using the configured strategy
        let selected_validators = self.selection_strategy.select_validators(
            &filtered_candidates,
            target_count,
            &self.config,
        ).await?;
        
        // Validate selection results
        self.validate_selection(&selected_validators)?;
        
        // Create selection result
        let selection_result = SelectionResult {
            epoch,
            selected_validators: selected_validators.clone(),
            total_candidates: candidates.len(),
            filtered_candidates: filtered_candidates.len(),
            selection_strategy: self.selection_strategy.get_name().to_string(),
            selection_time: start_time.elapsed(),
            timestamp: Utc::now(),
        };
        
        // Record selection event
        let selection_event = SelectionEvent {
            id: Uuid::new_v4(),
            epoch,
            selected_count: selected_validators.len(),
            total_candidates: candidates.len(),
            strategy_used: self.selection_strategy.get_name().to_string(),
            average_reputation: self.calculate_average_reputation(&selected_validators),
            average_performance: self.calculate_average_performance(&selected_validators),
            total_stake: self.calculate_total_stake(&selected_validators),
            timestamp: Utc::now(),
        };
        
        self.selection_history.push(selection_event);
        
        // Update metrics
        self.metrics.record_selection_time(start_time.elapsed().as_secs_f64());
        self.metrics.set_selected_validators(selected_validators.len() as i64);
        self.metrics.set_total_candidates(candidates.len() as i64);
        self.metrics.increment_selections();
        
        tracing::info!(
            "Selected {} validators for epoch {} from {} candidates using {} strategy",
            selected_validators.len(),
            epoch,
            candidates.len(),
            self.selection_strategy.get_name()
        );
        
        Ok(selection_result)
    }
    
    /// Build a comprehensive validator candidate
    async fn build_validator_candidate(&mut self, validator: &ValidatorInfo) -> Result<ValidatorCandidate> {
        // Get stake information
        let stake = self.data_provider.get_validator_stake(&validator.id).await?;
        
        // Get performance history
        let performance_history = self.data_provider.get_performance_history(
            &validator.id,
            self.config.performance_config.history_epochs,
        ).await?;
        
        // Get slashing history
        let slashing_history = self.data_provider.get_slashing_history(&validator.id).await?;
        
        // Calculate reputation score
        let reputation_score = self.reputation_engine.calculate_reputation(
            &validator.id,
            &performance_history,
            &slashing_history,
        ).await?;
        
        // Calculate performance metrics
        let performance_metrics = self.performance_analyzer.analyze_performance(
            &performance_history,
        ).await?;
        
        // Calculate selection score
        let selection_score = self.calculate_selection_score(
            stake,
            reputation_score,
            &performance_metrics,
        );
        
        Ok(ValidatorCandidate {
            validator_info: validator.clone(),
            stake,
            reputation_score,
            performance_metrics,
            selection_score,
            last_selected_epoch: self.get_last_selected_epoch(&validator.id),
            consecutive_selections: self.get_consecutive_selections(&validator.id),
            eligibility_status: EligibilityStatus::Eligible,
        })
    }
    
    /// Calculate composite selection score
    fn calculate_selection_score(
        &self,
        stake: Stake,
        reputation_score: ReputationScore,
        performance_metrics: &PerformanceMetrics,
    ) -> f64 {
        let stake_weight = self.config.stake_weight;
        let reputation_weight = self.config.reputation_weight;
        let performance_weight = self.config.performance_weight;
        
        // Normalize stake (log scale to prevent dominance)
        let normalized_stake = (stake as f64).ln() / (self.config.max_stake as f64).ln();
        
        // Normalize reputation (0-100 scale)
        let normalized_reputation = reputation_score.score as f64 / 100.0;
        
        // Normalize performance (0-1 scale)
        let normalized_performance = performance_metrics.overall_score;
        
        // Calculate weighted score
        let score = (normalized_stake * stake_weight) +
                   (normalized_reputation * reputation_weight) +
                   (normalized_performance * performance_weight);
        
        // Apply diversity bonus for validators not recently selected
        let diversity_bonus = if self.get_consecutive_selections(&"".to_string()) > 3 {
            0.9 // 10% penalty for over-selection
        } else {
            1.0
        };
        
        score * diversity_bonus
    }
    
    /// Apply pre-selection filters
    fn apply_pre_selection_filters(&self, candidates: &[ValidatorCandidate]) -> Result<Vec<ValidatorCandidate>> {
        let mut filtered = Vec::new();
        
        for candidate in candidates {
            // Check minimum stake requirement
            if candidate.stake < self.config.min_stake {
                continue;
            }
            
            // Check minimum reputation requirement
            if candidate.reputation_score.score < self.config.min_reputation {
                continue;
            }
            
            // Check if validator is in grace period
            if self.is_in_grace_period(&candidate.validator_info) {
                continue;
            }
            
            // Check if validator is currently slashed
            if candidate.validator_info.is_slashed {
                continue;
            }
            
            // Check maximum consecutive selections
            if candidate.consecutive_selections >= self.config.max_consecutive_selections {
                continue;
            }
            
            filtered.push(candidate.clone());
        }
        
        Ok(filtered)
    }
    
    /// Validate selection results
    fn validate_selection(&self, selected: &[SelectedValidator]) -> Result<()> {
        // Check minimum count
        if selected.len() < self.config.min_validators {
            return Err(ValidatorError::InsufficientValidators {
                available: selected.len(),
                required: self.config.min_validators,
            });
        }
        
        // Check stake distribution
        let total_stake: Stake = selected.iter().map(|v| v.stake).sum();
        let max_individual_stake = selected.iter().map(|v| v.stake).max().unwrap_or(0);
        
        if max_individual_stake > (total_stake * self.config.max_stake_concentration as u128) / 100 {
            return Err(ValidatorError::StakeConcentrationTooHigh);
        }
        
        // Check reputation distribution
        let avg_reputation: f64 = selected.iter()
            .map(|v| v.reputation_score.score as f64)
            .sum::<f64>() / selected.len() as f64;
        
        if avg_reputation < self.config.min_average_reputation as f64 {
            return Err(ValidatorError::AverageReputationTooLow);
        }
        
        Ok(())
    }
    
    /// Check if validator is in grace period
    fn is_in_grace_period(&self, validator: &ValidatorInfo) -> bool {
        let grace_period = chrono::Duration::seconds(self.config.grace_period_seconds as i64);
        Utc::now() - validator.joined_at < grace_period
    }
    
    /// Get last selected epoch for validator
    fn get_last_selected_epoch(&self, validator_id: &ValidatorId) -> Option<u64> {
        self.selection_history.iter()
            .rev()
            .find_map(|event| {
                // TODO: Check if validator was in this selection
                None
            })
    }
    
    /// Get consecutive selections count
    fn get_consecutive_selections(&self, _validator_id: &ValidatorId) -> u32 {
        // TODO: Implement consecutive selection tracking
        0
    }
    
    /// Calculate average reputation of selected validators
    fn calculate_average_reputation(&self, selected: &[SelectedValidator]) -> f64 {
        if selected.is_empty() {
            return 0.0;
        }
        
        selected.iter()
            .map(|v| v.reputation_score.score as f64)
            .sum::<f64>() / selected.len() as f64
    }
    
    /// Calculate average performance of selected validators
    fn calculate_average_performance(&self, selected: &[SelectedValidator]) -> f64 {
        if selected.is_empty() {
            return 0.0;
        }
        
        selected.iter()
            .map(|v| v.performance_metrics.overall_score)
            .sum::<f64>() / selected.len() as f64
    }
    
    /// Calculate total stake of selected validators
    fn calculate_total_stake(&self, selected: &[SelectedValidator]) -> Stake {
        selected.iter().map(|v| v.stake).sum()
    }
    
    /// Get selection statistics
    pub fn get_statistics(&self) -> SelectionStatistics {
        SelectionStatistics {
            total_selections: self.selection_history.len(),
            average_selection_time: self.metrics.get_average_selection_time(),
            average_validators_selected: self.calculate_average_validators_selected(),
            selection_strategy: self.selection_strategy.get_name().to_string(),
            last_selection: self.selection_history.last().map(|e| e.timestamp),
        }
    }
    
    /// Calculate average validators selected
    fn calculate_average_validators_selected(&self) -> f64 {
        if self.selection_history.is_empty() {
            return 0.0;
        }
        
        self.selection_history.iter()
            .map(|e| e.selected_count as f64)
            .sum::<f64>() / self.selection_history.len() as f64
    }
    
    /// Update validator performance
    pub async fn update_validator_performance(
        &mut self,
        validator_id: ValidatorId,
        performance_data: PerformanceData,
    ) -> Result<()> {
        self.performance_analyzer.update_performance(validator_id, performance_data).await?;
        self.metrics.increment_performance_updates();
        Ok(())
    }
    
    /// Update validator reputation
    pub async fn update_validator_reputation(
        &mut self,
        validator_id: ValidatorId,
        reputation_update: ReputationUpdate,
    ) -> Result<()> {
        self.reputation_engine.update_reputation(validator_id, reputation_update).await?;
        self.metrics.increment_reputation_updates();
        Ok(())
    }
}
