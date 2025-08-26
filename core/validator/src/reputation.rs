//! Reputation scoring engine for validators

use crate::{types::*, error::*};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Reputation calculation engine
pub struct ReputationEngine {
    config: ReputationConfig,
    reputation_cache: HashMap<ValidatorId, ReputationScore>,
    reputation_history: HashMap<ValidatorId, Vec<ReputationUpdate>>,
}

impl ReputationEngine {
    /// Create a new reputation engine
    pub fn new(config: ReputationConfig) -> Self {
        Self {
            config,
            reputation_cache: HashMap::new(),
            reputation_history: HashMap::new(),
        }
    }
    
    /// Calculate comprehensive reputation score for a validator
    pub async fn calculate_reputation(
        &mut self,
        validator_id: &ValidatorId,
        performance_history: &[PerformanceRecord],
        slashing_history: &[SlashingRecord],
    ) -> Result<ReputationScore> {
        // Check cache first
        if let Some(cached_score) = self.reputation_cache.get(validator_id) {
            if self.is_cache_valid(cached_score) {
                return Ok(cached_score.clone());
            }
        }
        
        // Calculate base reputation components
        let uptime_score = self.calculate_uptime_score(performance_history)?;
        let performance_score = self.calculate_performance_score(performance_history)?;
        let slashing_penalty = self.calculate_slashing_penalty(slashing_history)?;
        let governance_participation = self.calculate_governance_participation(validator_id).await?;
        
        // Apply weights and calculate final score
        let weighted_score = (uptime_score * self.config.uptime_weight) +
                           (performance_score * self.config.performance_weight) +
                           (governance_participation * self.config.governance_weight) -
                           (slashing_penalty * self.config.slashing_penalty_weight);
        
        // Apply decay for inactive periods
        let decayed_score = self.apply_reputation_decay(validator_id, weighted_score)?;
        
        // Clamp to valid range
        let final_score = (decayed_score * 100.0).clamp(0.0, self.config.max_score as f64) as u8;
        
        let reputation_score = ReputationScore {
            score: final_score,
            uptime_score,
            performance_score,
            slashing_penalty,
            governance_participation,
            last_updated: Utc::now(),
        };
        
        // Cache the result
        self.reputation_cache.insert(validator_id.clone(), reputation_score.clone());
        
        Ok(reputation_score)
    }
    
    /// Calculate uptime score from performance history
    fn calculate_uptime_score(&self, performance_history: &[PerformanceRecord]) -> Result<f64> {
        if performance_history.is_empty() {
            return Ok(0.5); // Neutral score for new validators
        }
        
        let mut total_uptime = 0.0;
        let mut total_weight = 0.0;
        let now = Utc::now();
        
        for record in performance_history {
            // Calculate age weight (more recent data has higher weight)
            let age_days = (now - record.timestamp).num_days() as f64;
            let weight = (-age_days * self.config.performance_decay_rate).exp();
            
            let uptime_ratio = if record.total_seconds > 0 {
                record.uptime_seconds as f64 / record.total_seconds as f64
            } else {
                0.0
            };
            
            total_uptime += uptime_ratio * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Ok(total_uptime / total_weight)
        } else {
            Ok(0.5)
        }
    }
    
    /// Calculate performance score from historical data
    fn calculate_performance_score(&self, performance_history: &[PerformanceRecord]) -> Result<f64> {
        if performance_history.is_empty() {
            return Ok(0.5); // Neutral score for new validators
        }
        
        let mut total_performance = 0.0;
        let mut total_weight = 0.0;
        let now = Utc::now();
        
        for record in performance_history {
            // Calculate age weight
            let age_days = (now - record.timestamp).num_days() as f64;
            let weight = (-age_days * self.config.performance_decay_rate).exp();
            
            // Calculate block production rate
            let block_production_rate = if record.blocks_expected > 0 {
                record.blocks_produced as f64 / record.blocks_expected as f64
            } else {
                1.0
            };
            
            // Calculate attestation rate
            let attestation_rate = if record.attestations_expected > 0 {
                record.attestations_made as f64 / record.attestations_expected as f64
            } else {
                1.0
            };
            
            // Calculate response time score (inverse relationship)
            let response_time_score = if record.average_response_time > 0.0 {
                (1000.0 / (record.average_response_time + 100.0)).min(1.0)
            } else {
                1.0
            };
            
            // Combine performance metrics
            let performance = (block_production_rate * 0.4) +
                            (attestation_rate * 0.4) +
                            (response_time_score * 0.2);
            
            total_performance += performance * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Ok(total_performance / total_weight)
        } else {
            Ok(0.5)
        }
    }
    
    /// Calculate slashing penalty impact on reputation
    fn calculate_slashing_penalty(&self, slashing_history: &[SlashingRecord]) -> Result<f64> {
        if slashing_history.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_penalty = 0.0;
        let now = Utc::now();
        
        for slashing in slashing_history {
            // Calculate time since slashing for recovery
            let days_since = (now - slashing.timestamp).num_days() as f64;
            let recovery_factor = if days_since > 0.0 {
                (-days_since / (self.config.min_recovery_epochs as f64 * 7.0)).exp()
            } else {
                1.0
            };
            
            // Calculate penalty based on slashing type
            let base_penalty = match slashing.slashing_type {
                SlashingType::Light => 0.1,
                SlashingType::Medium => 0.3,
                SlashingType::Severe => 0.8,
            };
            
            total_penalty += base_penalty * recovery_factor;
        }
        
        // Cap maximum penalty
        Ok(total_penalty.min(0.9))
    }
    
    /// Calculate governance participation score
    async fn calculate_governance_participation(&self, _validator_id: &ValidatorId) -> Result<f64> {
        // TODO: Implement governance participation tracking
        // For now, return neutral score
        Ok(0.5)
    }
    
    /// Apply reputation decay for inactive periods
    fn apply_reputation_decay(&self, validator_id: &ValidatorId, base_score: f64) -> Result<f64> {
        // Get last update time from cache
        let last_update = self.reputation_cache.get(validator_id)
            .map(|score| score.last_updated)
            .unwrap_or_else(Utc::now);
        
        let days_since_update = (Utc::now() - last_update).num_days() as f64;
        
        if days_since_update > 0.0 {
            let decay_factor = (-days_since_update * self.config.decay_rate).exp();
            Ok(base_score * decay_factor)
        } else {
            Ok(base_score)
        }
    }
    
    /// Check if cached reputation score is still valid
    fn is_cache_valid(&self, score: &ReputationScore) -> bool {
        let cache_duration = chrono::Duration::hours(1); // Cache for 1 hour
        Utc::now() - score.last_updated < cache_duration
    }
    
    /// Update validator reputation with specific event
    pub async fn update_reputation(
        &mut self,
        validator_id: ValidatorId,
        update: ReputationUpdate,
    ) -> Result<()> {
        // Get current reputation or create new one
        let mut current_score = self.reputation_cache.get(&validator_id)
            .cloned()
            .unwrap_or_else(|| ReputationScore {
                score: self.config.base_score,
                uptime_score: 0.5,
                performance_score: 0.5,
                slashing_penalty: 0.0,
                governance_participation: 0.5,
                last_updated: Utc::now(),
            });
        
        // Apply the update
        match update.update_type {
            ReputationUpdateType::PerformanceBonus => {
                current_score.performance_score = (current_score.performance_score + update.value * 0.1).min(1.0);
            }
            ReputationUpdateType::UptimeBonus => {
                current_score.uptime_score = (current_score.uptime_score + update.value * 0.1).min(1.0);
            }
            ReputationUpdateType::GovernanceParticipation => {
                current_score.governance_participation = (current_score.governance_participation + update.value * 0.1).min(1.0);
            }
            ReputationUpdateType::SlashingPenalty => {
                current_score.slashing_penalty = (current_score.slashing_penalty + update.value * 0.1).min(1.0);
            }
            ReputationUpdateType::MissedBlockPenalty => {
                current_score.performance_score = (current_score.performance_score - update.value * 0.05).max(0.0);
            }
            ReputationUpdateType::MissedAttestationPenalty => {
                current_score.performance_score = (current_score.performance_score - update.value * 0.02).max(0.0);
            }
            ReputationUpdateType::Recovery => {
                current_score.slashing_penalty = (current_score.slashing_penalty - update.value * self.config.recovery_rate).max(0.0);
            }
        }
        
        // Recalculate overall score
        let weighted_score = (current_score.uptime_score * self.config.uptime_weight) +
                           (current_score.performance_score * self.config.performance_weight) +
                           (current_score.governance_participation * self.config.governance_weight) -
                           (current_score.slashing_penalty * self.config.slashing_penalty_weight);
        
        current_score.score = (weighted_score * 100.0).clamp(0.0, self.config.max_score as f64) as u8;
        current_score.last_updated = Utc::now();
        
        // Update cache
        self.reputation_cache.insert(validator_id.clone(), current_score);
        
        // Record update in history
        self.reputation_history.entry(validator_id).or_insert_with(Vec::new).push(update);
        
        Ok(())
    }
    
    /// Get reputation history for a validator
    pub fn get_reputation_history(&self, validator_id: &ValidatorId) -> Option<&Vec<ReputationUpdate>> {
        self.reputation_history.get(validator_id)
    }
    
    /// Get current reputation score
    pub fn get_current_reputation(&self, validator_id: &ValidatorId) -> Option<&ReputationScore> {
        self.reputation_cache.get(validator_id)
    }
    
    /// Clear reputation cache
    pub fn clear_cache(&mut self) {
        self.reputation_cache.clear();
    }
    
    /// Get reputation statistics
    pub fn get_reputation_statistics(&self) -> ReputationStatistics {
        let total_validators = self.reputation_cache.len();
        
        if total_validators == 0 {
            return ReputationStatistics {
                total_validators: 0,
                average_reputation: 0.0,
                median_reputation: 0.0,
                high_reputation_count: 0,
                low_reputation_count: 0,
                reputation_distribution: HashMap::new(),
            };
        }
        
        let mut scores: Vec<u8> = self.reputation_cache.values().map(|s| s.score).collect();
        scores.sort();
        
        let average_reputation = scores.iter().map(|&s| s as f64).sum::<f64>() / scores.len() as f64;
        let median_reputation = scores[scores.len() / 2] as f64;
        
        let high_reputation_count = scores.iter().filter(|&&s| s >= 80).count();
        let low_reputation_count = scores.iter().filter(|&&s| s < 50).count();
        
        // Create distribution buckets
        let mut distribution = HashMap::new();
        for score in scores {
            let bucket = (score / 10) * 10; // Group by 10s
            *distribution.entry(bucket).or_insert(0) += 1;
        }
        
        ReputationStatistics {
            total_validators,
            average_reputation,
            median_reputation,
            high_reputation_count,
            low_reputation_count,
            reputation_distribution: distribution,
        }
    }
}

/// Reputation statistics
#[derive(Debug, Clone)]
pub struct ReputationStatistics {
    pub total_validators: usize,
    pub average_reputation: f64,
    pub median_reputation: f64,
    pub high_reputation_count: usize,
    pub low_reputation_count: usize,
    pub reputation_distribution: HashMap<u8, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    
    #[tokio::test]
    async fn test_reputation_calculation() {
        let config = ReputationConfig::default();
        let mut engine = ReputationEngine::new(config);
        
        let validator_id = "test_validator".to_string();
        
        // Create mock performance history
        let performance_history = vec![
            PerformanceRecord {
                epoch: 1,
                validator_id: validator_id.clone(),
                blocks_produced: 95,
                blocks_expected: 100,
                attestations_made: 98,
                attestations_expected: 100,
                uptime_seconds: 86000,
                total_seconds: 86400,
                average_response_time: 200.0,
                slashing_events: 0,
                timestamp: Utc::now() - chrono::Duration::days(1),
            }
        ];
        
        let slashing_history = vec![];
        
        let reputation = engine.calculate_reputation(
            &validator_id,
            &performance_history,
            &slashing_history,
        ).await.unwrap();
        
        assert!(reputation.score > 50); // Should have good reputation
        assert!(reputation.uptime_score > 0.9); // High uptime
        assert!(reputation.performance_score > 0.9); // Good performance
        assert_eq!(reputation.slashing_penalty, 0.0); // No slashing
    }
    
    #[tokio::test]
    async fn test_slashing_penalty() {
        let config = ReputationConfig::default();
        let mut engine = ReputationEngine::new(config);
        
        let validator_id = "slashed_validator".to_string();
        
        let performance_history = vec![];
        let slashing_history = vec![
            SlashingRecord {
                validator_id: validator_id.clone(),
                slashing_type: SlashingType::Medium,
                amount_slashed: 1000,
                reason: "Test slashing".to_string(),
                epoch: 1,
                timestamp: Utc::now() - chrono::Duration::days(1),
            }
        ];
        
        let reputation = engine.calculate_reputation(
            &validator_id,
            &performance_history,
            &slashing_history,
        ).await.unwrap();
        
        assert!(reputation.slashing_penalty > 0.0); // Should have penalty
        assert!(reputation.score < 50); // Should have low reputation due to slashing
    }
}
