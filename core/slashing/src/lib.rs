//! Quantlink Qross Slashing Engine
//! 
//! This module implements automated penalty calculation and stake redistribution mechanisms
//! with a three-tier slashing structure to enforce cryptoeconomic security in the consensus system.

pub mod engine;
pub mod evidence;
pub mod penalties;
pub mod redistribution;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub use error::{SlashingError, Result};
pub use types::*;

/// Trait for slashing evidence providers
#[async_trait]
pub trait EvidenceProvider: Send + Sync {
    /// Collect evidence of validator misbehavior
    async fn collect_evidence(&self, validator_id: &ValidatorId) -> Result<Vec<Evidence>>;
    
    /// Verify evidence authenticity
    async fn verify_evidence(&self, evidence: &Evidence) -> Result<bool>;
    
    /// Get evidence by ID
    async fn get_evidence(&self, evidence_id: &EvidenceId) -> Result<Option<Evidence>>;
}

/// Trait for stake management
#[async_trait]
pub trait StakeManager: Send + Sync {
    /// Get validator stake
    async fn get_stake(&self, validator_id: &ValidatorId) -> Result<Stake>;
    
    /// Slash validator stake
    async fn slash_stake(&mut self, validator_id: &ValidatorId, amount: Stake) -> Result<()>;
    
    /// Redistribute slashed stake
    async fn redistribute_stake(&mut self, recipients: &[(ValidatorId, Stake)]) -> Result<()>;
    
    /// Get total stake in the system
    async fn get_total_stake(&self) -> Result<Stake>;
    
    /// Lock stake for slashing investigation
    async fn lock_stake(&mut self, validator_id: &ValidatorId, amount: Stake) -> Result<()>;
    
    /// Unlock stake after investigation
    async fn unlock_stake(&mut self, validator_id: &ValidatorId, amount: Stake) -> Result<()>;
}

/// Main slashing engine
pub struct SlashingEngine {
    evidence_providers: Vec<Box<dyn EvidenceProvider>>,
    stake_manager: Box<dyn StakeManager>,
    penalty_calculator: penalties::PenaltyCalculator,
    redistribution_engine: redistribution::RedistributionEngine,
    config: SlashingConfig,
    metrics: metrics::SlashingMetrics,
    active_investigations: HashMap<ValidatorId, Investigation>,
    slashing_history: Vec<SlashingEvent>,
}

impl SlashingEngine {
    /// Create a new slashing engine
    pub fn new(
        stake_manager: Box<dyn StakeManager>,
        config: SlashingConfig,
    ) -> Self {
        Self {
            evidence_providers: Vec::new(),
            stake_manager,
            penalty_calculator: penalties::PenaltyCalculator::new(config.clone()),
            redistribution_engine: redistribution::RedistributionEngine::new(),
            config,
            metrics: metrics::SlashingMetrics::new(),
            active_investigations: HashMap::new(),
            slashing_history: Vec::new(),
        }
    }
    
    /// Add an evidence provider
    pub fn add_evidence_provider(&mut self, provider: Box<dyn EvidenceProvider>) {
        self.evidence_providers.push(provider);
    }
    
    /// Report potential misbehavior
    pub async fn report_misbehavior(
        &mut self,
        validator_id: ValidatorId,
        misbehavior_type: MisbehaviorType,
        reporter: ValidatorId,
        evidence_data: Vec<u8>,
    ) -> Result<InvestigationId> {
        let investigation_id = Uuid::new_v4();
        
        // Create evidence
        let evidence = Evidence {
            id: Uuid::new_v4(),
            validator_id: validator_id.clone(),
            misbehavior_type: misbehavior_type.clone(),
            evidence_data,
            reporter,
            timestamp: Utc::now(),
            block_height: 0, // TODO: Get current block height
            verified: false,
        };
        
        // Start investigation
        let investigation = Investigation {
            id: investigation_id,
            validator_id: validator_id.clone(),
            misbehavior_type,
            evidence: vec![evidence],
            status: InvestigationStatus::Pending,
            started_at: Utc::now(),
            deadline: Utc::now() + chrono::Duration::seconds(self.config.investigation_timeout as i64),
            locked_stake: 0,
        };
        
        self.active_investigations.insert(validator_id.clone(), investigation);
        
        // Lock stake during investigation
        let stake = self.stake_manager.get_stake(&validator_id).await?;
        let lock_amount = self.calculate_lock_amount(&misbehavior_type, stake);
        
        if lock_amount > 0 {
            self.stake_manager.lock_stake(&validator_id, lock_amount).await?;
            self.active_investigations.get_mut(&validator_id).unwrap().locked_stake = lock_amount;
        }
        
        self.metrics.increment_investigations();
        
        tracing::info!(
            "Started investigation {} for validator {} with misbehavior type {:?}",
            investigation_id, validator_id, misbehavior_type
        );
        
        Ok(investigation_id)
    }
    
    /// Process pending investigations
    pub async fn process_investigations(&mut self) -> Result<()> {
        let mut completed_investigations = Vec::new();
        
        for (validator_id, investigation) in &mut self.active_investigations {
            if investigation.status != InvestigationStatus::Pending {
                continue;
            }
            
            // Check if investigation has timed out
            if Utc::now() > investigation.deadline {
                investigation.status = InvestigationStatus::TimedOut;
                completed_investigations.push(validator_id.clone());
                continue;
            }
            
            // Collect additional evidence
            for provider in &self.evidence_providers {
                match provider.collect_evidence(validator_id).await {
                    Ok(evidence_list) => {
                        for evidence in evidence_list {
                            if provider.verify_evidence(&evidence).await.unwrap_or(false) {
                                investigation.evidence.push(evidence);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to collect evidence from provider: {}", e);
                    }
                }
            }
            
            // Evaluate evidence
            if self.has_sufficient_evidence(investigation) {
                investigation.status = InvestigationStatus::Proven;
                completed_investigations.push(validator_id.clone());
            } else if self.has_contradictory_evidence(investigation) {
                investigation.status = InvestigationStatus::Dismissed;
                completed_investigations.push(validator_id.clone());
            }
        }
        
        // Process completed investigations
        for validator_id in completed_investigations {
            self.complete_investigation(validator_id).await?;
        }
        
        Ok(())
    }
    
    /// Complete an investigation and apply penalties if necessary
    async fn complete_investigation(&mut self, validator_id: ValidatorId) -> Result<()> {
        let investigation = self.active_investigations.remove(&validator_id)
            .ok_or_else(|| SlashingError::InvestigationNotFound(validator_id.clone()))?;
        
        // Unlock stake
        if investigation.locked_stake > 0 {
            self.stake_manager.unlock_stake(&validator_id, investigation.locked_stake).await?;
        }
        
        match investigation.status {
            InvestigationStatus::Proven => {
                self.apply_slashing(&validator_id, &investigation).await?;
                self.metrics.increment_slashing_events();
            }
            InvestigationStatus::Dismissed => {
                tracing::info!("Investigation dismissed for validator {}", validator_id);
                self.metrics.increment_dismissed_investigations();
            }
            InvestigationStatus::TimedOut => {
                tracing::warn!("Investigation timed out for validator {}", validator_id);
                self.metrics.increment_timed_out_investigations();
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Apply slashing penalties
    async fn apply_slashing(&mut self, validator_id: &ValidatorId, investigation: &Investigation) -> Result<()> {
        let current_stake = self.stake_manager.get_stake(validator_id).await?;
        
        // Calculate penalty
        let penalty = self.penalty_calculator.calculate_penalty(
            &investigation.misbehavior_type,
            current_stake,
            &investigation.evidence,
        )?;
        
        // Apply slashing
        self.stake_manager.slash_stake(validator_id, penalty.amount).await?;
        
        // Redistribute slashed stake
        let redistribution = self.redistribution_engine.calculate_redistribution(
            penalty.amount,
            validator_id,
            &self.get_honest_validators().await?,
        ).await?;
        
        self.stake_manager.redistribute_stake(&redistribution).await?;
        
        // Record slashing event
        let slashing_event = SlashingEvent {
            id: Uuid::new_v4(),
            validator_id: validator_id.clone(),
            misbehavior_type: investigation.misbehavior_type.clone(),
            penalty_type: penalty.penalty_type,
            slashed_amount: penalty.amount,
            remaining_stake: current_stake - penalty.amount,
            timestamp: Utc::now(),
            evidence_count: investigation.evidence.len(),
            redistribution_recipients: redistribution.len(),
        };
        
        self.slashing_history.push(slashing_event.clone());
        
        // Update metrics
        self.metrics.record_slashing_amount(penalty.amount as f64);
        self.metrics.increment_penalty_type(&penalty.penalty_type);
        
        tracing::warn!(
            "Applied {} slashing to validator {}: {} stake slashed",
            penalty.penalty_type,
            validator_id,
            penalty.amount
        );
        
        Ok(())
    }
    
    /// Calculate amount to lock during investigation
    fn calculate_lock_amount(&self, misbehavior_type: &MisbehaviorType, stake: Stake) -> Stake {
        match misbehavior_type {
            MisbehaviorType::IncorrectAttestation => stake * self.config.light_slashing_percentage as u128 / 100,
            MisbehaviorType::ConflictingVotes => stake * self.config.medium_slashing_percentage as u128 / 100,
            MisbehaviorType::DoubleSigning => stake * self.config.severe_slashing_percentage as u128 / 100,
            MisbehaviorType::Unavailability => 0, // No lock for unavailability
            MisbehaviorType::InvalidProposal => stake * self.config.medium_slashing_percentage as u128 / 100,
        }
    }
    
    /// Check if investigation has sufficient evidence
    fn has_sufficient_evidence(&self, investigation: &Investigation) -> bool {
        let verified_evidence_count = investigation.evidence.iter()
            .filter(|e| e.verified)
            .count();
        
        match investigation.misbehavior_type {
            MisbehaviorType::DoubleSigning => verified_evidence_count >= 1, // One proof is sufficient
            MisbehaviorType::ConflictingVotes => verified_evidence_count >= 2, // Need multiple conflicting votes
            MisbehaviorType::IncorrectAttestation => verified_evidence_count >= 1,
            MisbehaviorType::Unavailability => verified_evidence_count >= 3, // Multiple missed blocks
            MisbehaviorType::InvalidProposal => verified_evidence_count >= 1,
        }
    }
    
    /// Check if investigation has contradictory evidence
    fn has_contradictory_evidence(&self, _investigation: &Investigation) -> bool {
        // TODO: Implement sophisticated evidence contradiction detection
        false
    }
    
    /// Get list of honest validators for redistribution
    async fn get_honest_validators(&self) -> Result<Vec<ValidatorId>> {
        // TODO: Implement logic to identify honest validators
        // For now, return empty list
        Ok(Vec::new())
    }
    
    /// Get slashing statistics
    pub fn get_statistics(&self) -> SlashingStatistics {
        let total_slashed = self.slashing_history.iter()
            .map(|event| event.slashed_amount)
            .sum();
        
        let penalty_counts = self.slashing_history.iter()
            .fold(HashMap::new(), |mut acc, event| {
                *acc.entry(event.penalty_type.clone()).or_insert(0) += 1;
                acc
            });
        
        SlashingStatistics {
            total_investigations: self.metrics.get_investigation_count(),
            active_investigations: self.active_investigations.len(),
            total_slashing_events: self.slashing_history.len(),
            total_slashed_amount: total_slashed,
            penalty_type_counts: penalty_counts,
            average_investigation_time: self.calculate_average_investigation_time(),
        }
    }
    
    /// Calculate average investigation time
    fn calculate_average_investigation_time(&self) -> f64 {
        if self.slashing_history.is_empty() {
            return 0.0;
        }
        
        // TODO: Track investigation start/end times for accurate calculation
        300.0 // Placeholder: 5 minutes average
    }
    
    /// Get slashing history for a validator
    pub fn get_validator_history(&self, validator_id: &ValidatorId) -> Vec<&SlashingEvent> {
        self.slashing_history.iter()
            .filter(|event| &event.validator_id == validator_id)
            .collect()
    }
    
    /// Check if validator is currently under investigation
    pub fn is_under_investigation(&self, validator_id: &ValidatorId) -> bool {
        self.active_investigations.contains_key(validator_id)
    }
    
    /// Get current investigation for validator
    pub fn get_investigation(&self, validator_id: &ValidatorId) -> Option<&Investigation> {
        self.active_investigations.get(validator_id)
    }
}
