//! Stake redistribution engine for slashed funds

use crate::{types::*, error::*};
use std::collections::HashMap;
use chrono::Utc;

/// Redistribution engine for slashed stake
pub struct RedistributionEngine {
    redistribution_history: Vec<RedistributionEvent>,
    community_pool: Stake,
    burned_amount: Stake,
}

/// Redistribution event record
#[derive(Debug, Clone)]
pub struct RedistributionEvent {
    pub id: uuid::Uuid,
    pub slashed_validator: ValidatorId,
    pub total_amount: Stake,
    pub redistributed_amount: Stake,
    pub burned_amount: Stake,
    pub community_pool_amount: Stake,
    pub recipients: Vec<RedistributionTarget>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RedistributionEngine {
    /// Create a new redistribution engine
    pub fn new() -> Self {
        Self {
            redistribution_history: Vec::new(),
            community_pool: 0,
            burned_amount: 0,
        }
    }
    
    /// Calculate redistribution of slashed stake
    pub async fn calculate_redistribution(
        &self,
        slashed_amount: Stake,
        slashed_validator: &ValidatorId,
        honest_validators: &[ValidatorId],
    ) -> Result<Vec<(ValidatorId, Stake)>> {
        if slashed_amount == 0 {
            return Ok(Vec::new());
        }
        
        // Get redistribution configuration
        let config = SlashingConfig::default(); // TODO: Pass config as parameter
        
        // Calculate amounts for different purposes
        let redistribution_amount = (slashed_amount * config.redistribution_percentage as u128) / 100;
        let burn_amount = (slashed_amount * config.burn_percentage as u128) / 100;
        let community_pool_amount = slashed_amount - redistribution_amount - burn_amount;
        
        // Calculate redistribution to honest validators
        let mut redistribution_targets = Vec::new();
        
        if !honest_validators.is_empty() && redistribution_amount > 0 {
            // Get validator stakes for proportional distribution
            let validator_stakes = self.get_validator_stakes(honest_validators).await?;
            let total_honest_stake: Stake = validator_stakes.values().sum();
            
            if total_honest_stake > 0 {
                for (validator_id, stake) in validator_stakes {
                    // Skip the slashed validator
                    if &validator_id == slashed_validator {
                        continue;
                    }
                    
                    // Calculate proportional share
                    let share = (redistribution_amount * stake) / total_honest_stake;
                    
                    if share > 0 {
                        redistribution_targets.push((validator_id, share));
                    }
                }
            }
        }
        
        // Add community pool allocation if any
        if community_pool_amount > 0 {
            redistribution_targets.push(("community_pool".to_string(), community_pool_amount));
        }
        
        Ok(redistribution_targets)
    }
    
    /// Execute redistribution of slashed stake
    pub async fn execute_redistribution(
        &mut self,
        slashed_amount: Stake,
        slashed_validator: ValidatorId,
        redistribution_targets: Vec<(ValidatorId, Stake)>,
    ) -> Result<RedistributionEvent> {
        let config = SlashingConfig::default();
        
        // Calculate amounts
        let redistributed_amount: Stake = redistribution_targets.iter()
            .filter(|(id, _)| id != "community_pool")
            .map(|(_, amount)| amount)
            .sum();
        
        let community_pool_amount: Stake = redistribution_targets.iter()
            .filter(|(id, _)| id == "community_pool")
            .map(|(_, amount)| amount)
            .sum();
        
        let burn_amount = (slashed_amount * config.burn_percentage as u128) / 100;
        
        // Update internal state
        self.community_pool += community_pool_amount;
        self.burned_amount += burn_amount;
        
        // Create redistribution targets with reasons
        let targets: Vec<RedistributionTarget> = redistribution_targets.into_iter()
            .map(|(validator_id, amount)| {
                let reason = if validator_id == "community_pool" {
                    RedistributionReason::CommunityPool
                } else {
                    RedistributionReason::HonestValidator
                };
                
                RedistributionTarget {
                    validator_id,
                    amount,
                    reason,
                }
            })
            .collect();
        
        // Add burn target if applicable
        let mut all_targets = targets;
        if burn_amount > 0 {
            all_targets.push(RedistributionTarget {
                validator_id: "burn".to_string(),
                amount: burn_amount,
                reason: RedistributionReason::Burn,
            });
        }
        
        // Create redistribution event
        let event = RedistributionEvent {
            id: uuid::Uuid::new_v4(),
            slashed_validator,
            total_amount: slashed_amount,
            redistributed_amount,
            burned_amount: burn_amount,
            community_pool_amount,
            recipients: all_targets,
            timestamp: Utc::now(),
        };
        
        // Record the event
        self.redistribution_history.push(event.clone());
        
        tracing::info!(
            "Executed redistribution: {} total, {} redistributed, {} burned, {} to community pool",
            slashed_amount, redistributed_amount, burn_amount, community_pool_amount
        );
        
        Ok(event)
    }
    
    /// Calculate reporter rewards for evidence submission
    pub fn calculate_reporter_rewards(
        &self,
        slashed_amount: Stake,
        reporters: &[ValidatorId],
    ) -> Vec<(ValidatorId, Stake)> {
        if reporters.is_empty() || slashed_amount == 0 {
            return Vec::new();
        }
        
        // Allocate 1% of slashed amount as reporter rewards
        let total_reward = slashed_amount / 100;
        let reward_per_reporter = total_reward / reporters.len() as u128;
        
        reporters.iter()
            .map(|reporter| (reporter.clone(), reward_per_reporter))
            .collect()
    }
    
    /// Get validator stakes for redistribution calculation
    async fn get_validator_stakes(&self, validators: &[ValidatorId]) -> Result<HashMap<ValidatorId, Stake>> {
        // TODO: Implement actual stake fetching from stake manager
        // For now, return equal stakes for all validators
        let mut stakes = HashMap::new();
        let default_stake = 1_000_000_000_000_000_000_000; // 1000 tokens
        
        for validator in validators {
            stakes.insert(validator.clone(), default_stake);
        }
        
        Ok(stakes)
    }
    
    /// Get redistribution statistics
    pub fn get_redistribution_statistics(&self) -> RedistributionStatistics {
        let total_redistributed: Stake = self.redistribution_history.iter()
            .map(|event| event.redistributed_amount)
            .sum();
        
        let total_burned: Stake = self.redistribution_history.iter()
            .map(|event| event.burned_amount)
            .sum();
        
        let total_to_community: Stake = self.redistribution_history.iter()
            .map(|event| event.community_pool_amount)
            .sum();
        
        let unique_recipients: std::collections::HashSet<_> = self.redistribution_history.iter()
            .flat_map(|event| event.recipients.iter().map(|r| &r.validator_id))
            .collect();
        
        RedistributionStatistics {
            total_events: self.redistribution_history.len(),
            total_redistributed_amount: total_redistributed,
            total_burned_amount: total_burned,
            total_community_pool_amount: total_to_community,
            current_community_pool: self.community_pool,
            total_burned: self.burned_amount,
            unique_recipients: unique_recipients.len(),
            average_redistribution_per_event: if self.redistribution_history.is_empty() {
                0.0
            } else {
                total_redistributed as f64 / self.redistribution_history.len() as f64
            },
        }
    }
    
    /// Get redistribution history for a validator
    pub fn get_validator_redistribution_history(&self, validator_id: &ValidatorId) -> Vec<&RedistributionEvent> {
        self.redistribution_history.iter()
            .filter(|event| {
                event.recipients.iter().any(|target| &target.validator_id == validator_id) ||
                &event.slashed_validator == validator_id
            })
            .collect()
    }
    
    /// Calculate redistribution efficiency
    pub fn calculate_redistribution_efficiency(&self) -> f64 {
        if self.redistribution_history.is_empty() {
            return 0.0;
        }
        
        let total_slashed: Stake = self.redistribution_history.iter()
            .map(|event| event.total_amount)
            .sum();
        
        let total_redistributed: Stake = self.redistribution_history.iter()
            .map(|event| event.redistributed_amount)
            .sum();
        
        if total_slashed == 0 {
            0.0
        } else {
            (total_redistributed as f64 / total_slashed as f64) * 100.0
        }
    }
    
    /// Get community pool balance
    pub fn get_community_pool_balance(&self) -> Stake {
        self.community_pool
    }
    
    /// Get total burned amount
    pub fn get_total_burned(&self) -> Stake {
        self.burned_amount
    }
    
    /// Withdraw from community pool (governance action)
    pub fn withdraw_from_community_pool(&mut self, amount: Stake, recipient: ValidatorId) -> Result<()> {
        if amount > self.community_pool {
            return Err(SlashingError::InsufficientStake {
                validator: "community_pool".to_string(),
                required: amount,
                available: self.community_pool,
            });
        }
        
        self.community_pool -= amount;
        
        tracing::info!(
            "Withdrew {} from community pool to {}. Remaining balance: {}",
            amount, recipient, self.community_pool
        );
        
        Ok(())
    }
}

/// Redistribution statistics
#[derive(Debug, Clone)]
pub struct RedistributionStatistics {
    pub total_events: usize,
    pub total_redistributed_amount: Stake,
    pub total_burned_amount: Stake,
    pub total_community_pool_amount: Stake,
    pub current_community_pool: Stake,
    pub total_burned: Stake,
    pub unique_recipients: usize,
    pub average_redistribution_per_event: f64,
}

impl Default for RedistributionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_redistribution_calculation() {
        let engine = RedistributionEngine::new();
        let slashed_amount = 1_000_000_000_000_000_000_000; // 1000 tokens
        let honest_validators = vec!["validator1".to_string(), "validator2".to_string()];
        
        let redistribution = engine.calculate_redistribution(
            slashed_amount,
            &"slashed_validator".to_string(),
            &honest_validators,
        ).await.unwrap();
        
        assert!(!redistribution.is_empty());
        
        let total_redistributed: Stake = redistribution.iter()
            .filter(|(id, _)| id != "community_pool")
            .map(|(_, amount)| amount)
            .sum();
        
        // Should redistribute 80% of slashed amount (default config)
        let expected_redistribution = (slashed_amount * 80) / 100;
        assert_eq!(total_redistributed, expected_redistribution);
    }
    
    #[test]
    fn test_reporter_rewards() {
        let engine = RedistributionEngine::new();
        let slashed_amount = 1_000_000_000_000_000_000_000; // 1000 tokens
        let reporters = vec!["reporter1".to_string(), "reporter2".to_string()];
        
        let rewards = engine.calculate_reporter_rewards(slashed_amount, &reporters);
        
        assert_eq!(rewards.len(), 2);
        
        let total_rewards: Stake = rewards.iter().map(|(_, amount)| amount).sum();
        let expected_total = slashed_amount / 100; // 1% of slashed amount
        
        assert_eq!(total_rewards, expected_total);
    }
}
