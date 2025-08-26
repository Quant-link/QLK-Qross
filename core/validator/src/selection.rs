//! Validator selection strategies

use crate::{types::*, error::*, SelectionStrategy};
use async_trait::async_trait;
use std::collections::HashMap;
use rand::seq::SliceRandom;

/// Stake-weighted selection strategy
pub struct StakeWeightedSelection;

#[async_trait]
impl SelectionStrategy for StakeWeightedSelection {
    async fn select_validators(
        &self,
        candidates: &[ValidatorCandidate],
        target_count: usize,
        _config: &SelectionConfig,
    ) -> Result<Vec<SelectedValidator>> {
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| b.stake.cmp(&a.stake));
        
        let selected = sorted_candidates.into_iter()
            .take(target_count)
            .map(|candidate| SelectedValidator {
                validator_id: candidate.validator_info.id,
                stake: candidate.stake,
                reputation_score: candidate.reputation_score,
                performance_metrics: candidate.performance_metrics,
                selection_score: candidate.selection_score,
                selection_reason: SelectionReason::HighStake,
            })
            .collect();
        
        Ok(selected)
    }
    
    fn get_name(&self) -> &'static str {
        "stake_weighted"
    }
    
    fn get_parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Reputation-based selection strategy
pub struct ReputationBasedSelection;

#[async_trait]
impl SelectionStrategy for ReputationBasedSelection {
    async fn select_validators(
        &self,
        candidates: &[ValidatorCandidate],
        target_count: usize,
        _config: &SelectionConfig,
    ) -> Result<Vec<SelectedValidator>> {
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| b.reputation_score.score.cmp(&a.reputation_score.score));
        
        let selected = sorted_candidates.into_iter()
            .take(target_count)
            .map(|candidate| SelectedValidator {
                validator_id: candidate.validator_info.id,
                stake: candidate.stake,
                reputation_score: candidate.reputation_score,
                performance_metrics: candidate.performance_metrics,
                selection_score: candidate.selection_score,
                selection_reason: SelectionReason::HighReputation,
            })
            .collect();
        
        Ok(selected)
    }
    
    fn get_name(&self) -> &'static str {
        "reputation_based"
    }
    
    fn get_parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Balanced selection strategy combining multiple factors
pub struct BalancedSelection;

#[async_trait]
impl SelectionStrategy for BalancedSelection {
    async fn select_validators(
        &self,
        candidates: &[ValidatorCandidate],
        target_count: usize,
        _config: &SelectionConfig,
    ) -> Result<Vec<SelectedValidator>> {
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|a, b| b.selection_score.partial_cmp(&a.selection_score).unwrap());
        
        let selected = sorted_candidates.into_iter()
            .take(target_count)
            .map(|candidate| SelectedValidator {
                validator_id: candidate.validator_info.id,
                stake: candidate.stake,
                reputation_score: candidate.reputation_score,
                performance_metrics: candidate.performance_metrics,
                selection_score: candidate.selection_score,
                selection_reason: SelectionReason::Balanced,
            })
            .collect();
        
        Ok(selected)
    }
    
    fn get_name(&self) -> &'static str {
        "balanced"
    }
    
    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("stake_weight".to_string(), 0.4);
        params.insert("reputation_weight".to_string(), 0.4);
        params.insert("performance_weight".to_string(), 0.2);
        params
    }
}

/// Diversity-focused selection strategy
pub struct DiversitySelection;

#[async_trait]
impl SelectionStrategy for DiversitySelection {
    async fn select_validators(
        &self,
        candidates: &[ValidatorCandidate],
        target_count: usize,
        _config: &SelectionConfig,
    ) -> Result<Vec<SelectedValidator>> {
        // Prioritize validators not recently selected
        let mut diversity_candidates = candidates.to_vec();
        diversity_candidates.sort_by_key(|c| c.consecutive_selections);
        
        let selected = diversity_candidates.into_iter()
            .take(target_count)
            .map(|candidate| SelectedValidator {
                validator_id: candidate.validator_info.id,
                stake: candidate.stake,
                reputation_score: candidate.reputation_score,
                performance_metrics: candidate.performance_metrics,
                selection_score: candidate.selection_score,
                selection_reason: SelectionReason::Diversity,
            })
            .collect();
        
        Ok(selected)
    }
    
    fn get_name(&self) -> &'static str {
        "diversity"
    }
    
    fn get_parameters(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
