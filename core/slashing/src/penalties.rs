//! Penalty calculation engine with three-tier slashing structure

use crate::{types::*, error::*};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Penalty calculation engine
pub struct PenaltyCalculator {
    config: SlashingConfig,
    slashing_history: HashMap<ValidatorId, Vec<SlashingEvent>>,
    cooldown_tracker: HashMap<ValidatorId, DateTime<Utc>>,
}

impl PenaltyCalculator {
    /// Create a new penalty calculator
    pub fn new(config: SlashingConfig) -> Self {
        Self {
            config,
            slashing_history: HashMap::new(),
            cooldown_tracker: HashMap::new(),
        }
    }
    
    /// Calculate penalty for validator misbehavior
    pub fn calculate_penalty(
        &self,
        misbehavior_type: &MisbehaviorType,
        current_stake: Stake,
        evidence: &[Evidence],
    ) -> Result<Penalty> {
        // Validate minimum stake requirement
        if current_stake < self.config.min_slashable_stake {
            return Ok(Penalty {
                penalty_type: PenaltyType::None,
                amount: 0,
                reason: "Stake below minimum slashable amount".to_string(),
                evidence_count: evidence.len(),
                calculated_at: Utc::now(),
            });
        }
        
        // Check evidence validity
        self.validate_evidence(evidence)?;
        
        // Determine base penalty type and percentage
        let (penalty_type, base_percentage) = self.get_base_penalty(misbehavior_type);
        
        // Calculate base penalty amount
        let base_amount = (current_stake * base_percentage as u128) / 100;
        
        // Apply modifiers based on evidence and history
        let modified_amount = self.apply_penalty_modifiers(
            base_amount,
            evidence,
            misbehavior_type,
            current_stake,
        )?;
        
        // Ensure penalty doesn't exceed maximum limits
        let final_amount = self.apply_penalty_limits(modified_amount, current_stake)?;
        
        Ok(Penalty {
            penalty_type,
            amount: final_amount,
            reason: self.generate_penalty_reason(misbehavior_type, evidence),
            evidence_count: evidence.len(),
            calculated_at: Utc::now(),
        })
    }
    
    /// Get base penalty type and percentage for misbehavior
    fn get_base_penalty(&self, misbehavior_type: &MisbehaviorType) -> (PenaltyType, u8) {
        match misbehavior_type {
            MisbehaviorType::IncorrectAttestation => {
                (PenaltyType::Light, self.config.light_slashing_percentage)
            }
            MisbehaviorType::ConflictingVotes => {
                (PenaltyType::Medium, self.config.medium_slashing_percentage)
            }
            MisbehaviorType::DoubleSigning => {
                (PenaltyType::Severe, self.config.severe_slashing_percentage)
            }
            MisbehaviorType::InvalidProposal => {
                (PenaltyType::Medium, self.config.medium_slashing_percentage)
            }
            MisbehaviorType::Unavailability => {
                (PenaltyType::None, 0) // Reputation penalty only
            }
        }
    }
    
    /// Apply penalty modifiers based on evidence quality and validator history
    fn apply_penalty_modifiers(
        &self,
        base_amount: Stake,
        evidence: &[Evidence],
        misbehavior_type: &MisbehaviorType,
        current_stake: Stake,
    ) -> Result<Stake> {
        let mut modified_amount = base_amount;
        
        // Evidence quality modifier
        let evidence_quality = self.assess_evidence_quality(evidence);
        let quality_modifier = match evidence_quality {
            EvidenceQuality::Weak => 0.7,      // Reduce penalty by 30%
            EvidenceQuality::Standard => 1.0,   // No change
            EvidenceQuality::Strong => 1.2,     // Increase penalty by 20%
            EvidenceQuality::Overwhelming => 1.5, // Increase penalty by 50%
        };
        
        modified_amount = (modified_amount as f64 * quality_modifier) as Stake;
        
        // Repeat offender modifier
        let repeat_modifier = self.calculate_repeat_offender_modifier(misbehavior_type);
        modified_amount = (modified_amount as f64 * repeat_modifier) as Stake;
        
        // Severity escalation for multiple concurrent violations
        if evidence.len() > 1 {
            let escalation_factor = 1.0 + (evidence.len() - 1) as f64 * 0.1; // 10% per additional evidence
            modified_amount = (modified_amount as f64 * escalation_factor.min(2.0)) as Stake; // Cap at 2x
        }
        
        // Stake-based modifier (larger stakes get proportionally higher penalties)
        if current_stake > self.config.min_slashable_stake * 10 {
            let stake_modifier = 1.0 + ((current_stake / self.config.min_slashable_stake) as f64).log10() * 0.1;
            modified_amount = (modified_amount as f64 * stake_modifier.min(1.5)) as Stake; // Cap at 1.5x
        }
        
        Ok(modified_amount)
    }
    
    /// Assess the quality of evidence
    fn assess_evidence_quality(&self, evidence: &[Evidence]) -> EvidenceQuality {
        let verified_count = evidence.iter().filter(|e| e.verified).count();
        let total_count = evidence.len();
        
        if verified_count == 0 {
            return EvidenceQuality::Weak;
        }
        
        let verification_ratio = verified_count as f64 / total_count as f64;
        let evidence_freshness = self.calculate_evidence_freshness(evidence);
        let reporter_diversity = self.calculate_reporter_diversity(evidence);
        
        let quality_score = verification_ratio * 0.5 + evidence_freshness * 0.3 + reporter_diversity * 0.2;
        
        match quality_score {
            score if score >= 0.9 => EvidenceQuality::Overwhelming,
            score if score >= 0.7 => EvidenceQuality::Strong,
            score if score >= 0.5 => EvidenceQuality::Standard,
            _ => EvidenceQuality::Weak,
        }
    }
    
    /// Calculate evidence freshness score
    fn calculate_evidence_freshness(&self, evidence: &[Evidence]) -> f64 {
        if evidence.is_empty() {
            return 0.0;
        }
        
        let now = Utc::now();
        let freshness_scores: Vec<f64> = evidence.iter()
            .map(|e| {
                let age_seconds = (now - e.timestamp).num_seconds() as f64;
                let max_age = self.config.max_evidence_age as f64 * 12.0; // Assuming 12 second blocks
                (1.0 - (age_seconds / max_age)).max(0.0)
            })
            .collect();
        
        freshness_scores.iter().sum::<f64>() / freshness_scores.len() as f64
    }
    
    /// Calculate reporter diversity score
    fn calculate_reporter_diversity(&self, evidence: &[Evidence]) -> f64 {
        if evidence.is_empty() {
            return 0.0;
        }
        
        let unique_reporters: std::collections::HashSet<_> = evidence.iter()
            .map(|e| &e.reporter)
            .collect();
        
        let diversity_ratio = unique_reporters.len() as f64 / evidence.len() as f64;
        diversity_ratio
    }
    
    /// Calculate repeat offender modifier
    fn calculate_repeat_offender_modifier(&self, _misbehavior_type: &MisbehaviorType) -> f64 {
        // TODO: Implement based on validator's slashing history
        // For now, return base modifier
        1.0
    }
    
    /// Apply penalty limits to prevent excessive slashing
    fn apply_penalty_limits(&self, amount: Stake, current_stake: Stake) -> Result<Stake> {
        // Ensure penalty doesn't exceed current stake
        let max_by_stake = current_stake;
        
        // Ensure penalty doesn't exceed per-epoch limit
        let max_by_epoch = self.config.max_slashing_per_epoch;
        
        // Apply the most restrictive limit
        let limited_amount = amount.min(max_by_stake).min(max_by_epoch);
        
        Ok(limited_amount)
    }
    
    /// Generate human-readable penalty reason
    fn generate_penalty_reason(&self, misbehavior_type: &MisbehaviorType, evidence: &[Evidence]) -> String {
        let base_reason = match misbehavior_type {
            MisbehaviorType::IncorrectAttestation => "Incorrect attestation submitted",
            MisbehaviorType::ConflictingVotes => "Conflicting votes detected in same round",
            MisbehaviorType::DoubleSigning => "Double signing detected",
            MisbehaviorType::InvalidProposal => "Invalid proposal submitted",
            MisbehaviorType::Unavailability => "Validator unavailability",
        };
        
        format!("{} (Evidence count: {})", base_reason, evidence.len())
    }
    
    /// Validate evidence before penalty calculation
    fn validate_evidence(&self, evidence: &[Evidence]) -> Result<()> {
        if evidence.is_empty() {
            return Err(SlashingError::InvalidEvidence("No evidence provided".to_string()));
        }
        
        let now = Utc::now();
        
        for evidence_item in evidence {
            // Check evidence age
            let age_seconds = (now - evidence_item.timestamp).num_seconds();
            let min_age = self.config.min_evidence_age as i64 * 12; // Assuming 12 second blocks
            let max_age = self.config.max_evidence_age as i64 * 12;
            
            if age_seconds < min_age {
                return Err(SlashingError::EvidenceTooRecent(
                    format!("Evidence is {} seconds old, minimum is {}", age_seconds, min_age)
                ));
            }
            
            if age_seconds > max_age {
                return Err(SlashingError::EvidenceTooOld(
                    format!("Evidence is {} seconds old, maximum is {}", age_seconds, max_age)
                ));
            }
            
            // Validate evidence data is not empty
            if evidence_item.evidence_data.is_empty() {
                return Err(SlashingError::InvalidEvidence("Empty evidence data".to_string()));
            }
        }
        
        Ok(())
    }
    
    /// Check if validator is in cooldown period
    pub fn is_in_cooldown(&self, validator_id: &ValidatorId) -> bool {
        if let Some(last_slashed) = self.cooldown_tracker.get(validator_id) {
            let cooldown_end = *last_slashed + chrono::Duration::seconds(self.config.slashing_cooldown as i64);
            Utc::now() < cooldown_end
        } else {
            false
        }
    }
    
    /// Record slashing event for cooldown tracking
    pub fn record_slashing(&mut self, validator_id: ValidatorId, event: SlashingEvent) {
        self.cooldown_tracker.insert(validator_id.clone(), event.timestamp);
        self.slashing_history.entry(validator_id).or_insert_with(Vec::new).push(event);
    }
    
    /// Get slashing history for validator
    pub fn get_validator_history(&self, validator_id: &ValidatorId) -> Option<&Vec<SlashingEvent>> {
        self.slashing_history.get(validator_id)
    }
    
    /// Calculate total slashed amount for validator
    pub fn get_total_slashed(&self, validator_id: &ValidatorId) -> Stake {
        self.slashing_history.get(validator_id)
            .map(|events| events.iter().map(|e| e.slashed_amount).sum())
            .unwrap_or(0)
    }
}

/// Evidence quality assessment
#[derive(Debug, Clone, PartialEq)]
enum EvidenceQuality {
    Weak,
    Standard,
    Strong,
    Overwhelming,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    
    #[test]
    fn test_light_slashing_calculation() {
        let config = SlashingConfig::default();
        let calculator = PenaltyCalculator::new(config);
        
        let evidence = vec![Evidence {
            id: Uuid::new_v4(),
            validator_id: "validator1".to_string(),
            misbehavior_type: MisbehaviorType::IncorrectAttestation,
            evidence_data: vec![1, 2, 3],
            reporter: "reporter1".to_string(),
            timestamp: Utc::now() - chrono::Duration::seconds(100),
            block_height: 1000,
            verified: true,
        }];
        
        let stake = 1_000_000_000_000_000_000_000; // 1000 tokens
        let penalty = calculator.calculate_penalty(
            &MisbehaviorType::IncorrectAttestation,
            stake,
            &evidence,
        ).unwrap();
        
        assert_eq!(penalty.penalty_type, PenaltyType::Light);
        assert_eq!(penalty.amount, stake * 5 / 100); // 5% slashing
    }
    
    #[test]
    fn test_severe_slashing_calculation() {
        let config = SlashingConfig::default();
        let calculator = PenaltyCalculator::new(config);
        
        let evidence = vec![Evidence {
            id: Uuid::new_v4(),
            validator_id: "validator1".to_string(),
            misbehavior_type: MisbehaviorType::DoubleSigning,
            evidence_data: vec![1, 2, 3],
            reporter: "reporter1".to_string(),
            timestamp: Utc::now() - chrono::Duration::seconds(100),
            block_height: 1000,
            verified: true,
        }];
        
        let stake = 1_000_000_000_000_000_000_000; // 1000 tokens
        let penalty = calculator.calculate_penalty(
            &MisbehaviorType::DoubleSigning,
            stake,
            &evidence,
        ).unwrap();
        
        assert_eq!(penalty.penalty_type, PenaltyType::Severe);
        assert_eq!(penalty.amount, stake * 50 / 100); // 50% slashing
    }
}
