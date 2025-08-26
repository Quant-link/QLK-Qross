//! Weighted voting mechanism with stake-based consensus

use crate::{
    types::*, error::*, ValidatorId, ProposalId, Stake, ReputationScore
};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Voting engine for stake-based consensus
pub struct VotingEngine {
    active_proposals: HashMap<ProposalId, ProposalVoting>,
    validator_stakes: HashMap<ValidatorId, Stake>,
    validator_reputations: HashMap<ValidatorId, ReputationScore>,
    config: VotingConfig,
    finalized_proposals: Vec<FinalizedProposal>,
}

/// Voting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingConfig {
    /// Minimum approval percentage required (0-100)
    pub min_approval_percentage: u8,
    
    /// Voting timeout in seconds
    pub voting_timeout: u64,
    
    /// Minimum stake weight for vote to count
    pub min_stake_weight: Stake,
    
    /// Reputation weight factor (0.0-1.0)
    pub reputation_weight_factor: f64,
    
    /// Stake weight factor (0.0-1.0)
    pub stake_weight_factor: f64,
}

/// Proposal voting state
#[derive(Debug, Clone)]
struct ProposalVoting {
    proposal: Proposal,
    votes: HashMap<ValidatorId, WeightedVote>,
    total_approve_weight: u128,
    total_reject_weight: u128,
    total_abstain_weight: u128,
    total_possible_weight: u128,
    started_at: DateTime<Utc>,
    timeout_at: DateTime<Utc>,
    is_finalized: bool,
}

impl VotingEngine {
    /// Create a new voting engine
    pub fn new() -> Self {
        Self {
            active_proposals: HashMap::new(),
            validator_stakes: HashMap::new(),
            validator_reputations: HashMap::new(),
            config: VotingConfig::default(),
            finalized_proposals: Vec::new(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: VotingConfig) -> Self {
        Self {
            active_proposals: HashMap::new(),
            validator_stakes: HashMap::new(),
            validator_reputations: HashMap::new(),
            config,
            finalized_proposals: Vec::new(),
        }
    }
    
    /// Update validator stake
    pub fn update_validator_stake(&mut self, validator_id: ValidatorId, stake: Stake) {
        self.validator_stakes.insert(validator_id, stake);
        self.recalculate_weights();
    }
    
    /// Update validator reputation
    pub fn update_validator_reputation(&mut self, validator_id: ValidatorId, reputation: ReputationScore) {
        self.validator_reputations.insert(validator_id, reputation);
        self.recalculate_weights();
    }
    
    /// Start voting on a proposal
    pub fn start_voting(&mut self, proposal: Proposal) -> Result<()> {
        let proposal_id = proposal.id;
        
        if self.active_proposals.contains_key(&proposal_id) {
            return Err(ConsensusError::InvalidProposal("Proposal already exists".to_string()));
        }
        
        let now = Utc::now();
        let timeout_at = now + chrono::Duration::seconds(self.config.voting_timeout as i64);
        
        let total_possible_weight = self.calculate_total_possible_weight();
        
        let proposal_voting = ProposalVoting {
            proposal,
            votes: HashMap::new(),
            total_approve_weight: 0,
            total_reject_weight: 0,
            total_abstain_weight: 0,
            total_possible_weight,
            started_at: now,
            timeout_at,
            is_finalized: false,
        };
        
        self.active_proposals.insert(proposal_id, proposal_voting);
        
        Ok(())
    }
    
    /// Process a vote message
    pub async fn process_vote(&mut self, message: ConsensusMessage) -> Result<()> {
        let vote = message.vote.ok_or_else(|| 
            ConsensusError::InvalidVote("Missing vote in message".to_string()))?;
        
        let proposal_id = vote.proposal_id;
        
        // Check if proposal exists and is active
        let proposal_voting = self.active_proposals.get_mut(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Proposal not found".to_string()))?;
        
        if proposal_voting.is_finalized {
            return Err(ConsensusError::InvalidVote("Proposal already finalized".to_string()));
        }
        
        // Check if voting has timed out
        if Utc::now() > proposal_voting.timeout_at {
            self.finalize_proposal_timeout(proposal_id)?;
            return Err(ConsensusError::ProposalTimeout("Voting period expired".to_string()));
        }
        
        // Check for duplicate vote
        if proposal_voting.votes.contains_key(&vote.voter) {
            return Err(ConsensusError::DuplicateVote(vote.voter));
        }
        
        // Calculate vote weight
        let weight = self.calculate_vote_weight(&vote.voter)?;
        
        // Check minimum stake requirement
        if weight < self.config.min_stake_weight {
            return Err(ConsensusError::InsufficientStake {
                required: self.config.min_stake_weight,
                actual: weight,
            });
        }
        
        // Create weighted vote
        let validator_info = self.get_validator_info(&vote.voter)?;
        let weighted_vote = WeightedVote {
            vote: vote.clone(),
            weight,
            validator_info,
        };
        
        // Update vote counts
        match vote.vote_type {
            VoteType::Approve => proposal_voting.total_approve_weight += weight,
            VoteType::Reject => proposal_voting.total_reject_weight += weight,
            VoteType::Abstain => proposal_voting.total_abstain_weight += weight,
        }
        
        // Store the vote
        proposal_voting.votes.insert(vote.voter, weighted_vote);
        
        // Check if consensus is reached
        if self.check_consensus_reached(&proposal_id)? {
            self.finalize_proposal(proposal_id)?;
        }
        
        Ok(())
    }
    
    /// Check if consensus is reached for a proposal
    fn check_consensus_reached(&self, proposal_id: &ProposalId) -> Result<bool> {
        let proposal_voting = self.active_proposals.get(proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Proposal not found".to_string()))?;
        
        let total_votes = proposal_voting.total_approve_weight + 
                         proposal_voting.total_reject_weight + 
                         proposal_voting.total_abstain_weight;
        
        if total_votes == 0 {
            return Ok(false);
        }
        
        let approval_percentage = (proposal_voting.total_approve_weight * 100) / total_votes;
        
        // Check if we have enough approval
        if approval_percentage >= self.config.min_approval_percentage as u128 {
            return Ok(true);
        }
        
        // Check if rejection is certain (even if all remaining votes are approve)
        let remaining_weight = proposal_voting.total_possible_weight - total_votes;
        let max_possible_approve = proposal_voting.total_approve_weight + remaining_weight;
        let max_approval_percentage = (max_possible_approve * 100) / proposal_voting.total_possible_weight;
        
        if max_approval_percentage < self.config.min_approval_percentage as u128 {
            return Ok(true); // Consensus reached (rejection)
        }
        
        Ok(false)
    }
    
    /// Finalize a proposal
    fn finalize_proposal(&mut self, proposal_id: ProposalId) -> Result<()> {
        let mut proposal_voting = self.active_proposals.remove(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Proposal not found".to_string()))?;
        
        proposal_voting.is_finalized = true;
        
        let total_votes = proposal_voting.total_approve_weight + 
                         proposal_voting.total_reject_weight + 
                         proposal_voting.total_abstain_weight;
        
        let result = if total_votes == 0 {
            ConsensusResult::Timeout
        } else {
            let approval_percentage = (proposal_voting.total_approve_weight * 100) / total_votes;
            if approval_percentage >= self.config.min_approval_percentage as u128 {
                ConsensusResult::Approved
            } else {
                ConsensusResult::Rejected
            }
        };
        
        let finalized = FinalizedProposal {
            proposal_id,
            proposal: proposal_voting.proposal,
            votes: proposal_voting.votes.into_values().map(|wv| wv.vote).collect(),
            finalized_at: Utc::now(),
            result,
        };
        
        self.finalized_proposals.push(finalized);
        
        Ok(())
    }
    
    /// Finalize a proposal due to timeout
    fn finalize_proposal_timeout(&mut self, proposal_id: ProposalId) -> Result<()> {
        let mut proposal_voting = self.active_proposals.remove(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Proposal not found".to_string()))?;
        
        proposal_voting.is_finalized = true;
        
        let finalized = FinalizedProposal {
            proposal_id,
            proposal: proposal_voting.proposal,
            votes: proposal_voting.votes.into_values().map(|wv| wv.vote).collect(),
            finalized_at: Utc::now(),
            result: ConsensusResult::Timeout,
        };
        
        self.finalized_proposals.push(finalized);
        
        Ok(())
    }
    
    /// Calculate vote weight based on stake and reputation
    fn calculate_vote_weight(&self, validator_id: &ValidatorId) -> Result<u128> {
        let stake = self.validator_stakes.get(validator_id)
            .copied()
            .unwrap_or(0);
        
        let reputation = self.validator_reputations.get(validator_id)
            .copied()
            .unwrap_or(50) as f64 / 100.0; // Default to 50% reputation
        
        let stake_weight = (stake as f64 * self.config.stake_weight_factor) as u128;
        let reputation_multiplier = 1.0 + (reputation - 0.5) * self.config.reputation_weight_factor;
        
        let final_weight = (stake_weight as f64 * reputation_multiplier) as u128;
        
        Ok(final_weight)
    }
    
    /// Calculate total possible weight from all validators
    fn calculate_total_possible_weight(&self) -> u128 {
        self.validator_stakes.iter()
            .map(|(validator_id, stake)| {
                self.calculate_vote_weight(validator_id).unwrap_or(0)
            })
            .sum()
    }
    
    /// Recalculate weights for all active proposals
    fn recalculate_weights(&mut self) {
        for proposal_voting in self.active_proposals.values_mut() {
            proposal_voting.total_possible_weight = self.calculate_total_possible_weight();
            
            // Recalculate vote weights
            let mut new_approve_weight = 0;
            let mut new_reject_weight = 0;
            let mut new_abstain_weight = 0;
            
            for weighted_vote in proposal_voting.votes.values_mut() {
                let new_weight = self.calculate_vote_weight(&weighted_vote.vote.voter).unwrap_or(0);
                weighted_vote.weight = new_weight;
                
                match weighted_vote.vote.vote_type {
                    VoteType::Approve => new_approve_weight += new_weight,
                    VoteType::Reject => new_reject_weight += new_weight,
                    VoteType::Abstain => new_abstain_weight += new_weight,
                }
            }
            
            proposal_voting.total_approve_weight = new_approve_weight;
            proposal_voting.total_reject_weight = new_reject_weight;
            proposal_voting.total_abstain_weight = new_abstain_weight;
        }
    }
    
    /// Get validator information
    fn get_validator_info(&self, validator_id: &ValidatorId) -> Result<ValidatorInfo> {
        let stake = self.validator_stakes.get(validator_id).copied().unwrap_or(0);
        let reputation = self.validator_reputations.get(validator_id).copied().unwrap_or(50);
        
        Ok(ValidatorInfo {
            id: validator_id.clone(),
            stake,
            reputation,
            public_key: vec![], // TODO: Implement public key storage
            network_address: String::new(), // TODO: Implement network address storage
            is_active: true,
            joined_at: Utc::now(), // TODO: Implement proper tracking
            last_seen: Utc::now(),
        })
    }
    
    /// Get finalized proposals and clear the list
    pub fn get_finalized_proposals(&mut self) -> Vec<FinalizedProposal> {
        std::mem::take(&mut self.finalized_proposals)
    }
    
    /// Get number of active proposals
    pub fn get_active_proposal_count(&self) -> usize {
        self.active_proposals.len()
    }
    
    /// Clean up expired proposals
    pub fn cleanup_expired_proposals(&mut self) -> Result<()> {
        let now = Utc::now();
        let expired_proposals: Vec<ProposalId> = self.active_proposals
            .iter()
            .filter(|(_, voting)| now > voting.timeout_at && !voting.is_finalized)
            .map(|(id, _)| *id)
            .collect();
        
        for proposal_id in expired_proposals {
            self.finalize_proposal_timeout(proposal_id)?;
        }
        
        Ok(())
    }
}

impl Default for VotingConfig {
    fn default() -> Self {
        Self {
            min_approval_percentage: 67,
            voting_timeout: 300, // 5 minutes
            min_stake_weight: 1_000_000_000_000_000_000, // 1 token
            reputation_weight_factor: 0.2,
            stake_weight_factor: 1.0,
        }
    }
}

impl Default for VotingEngine {
    fn default() -> Self {
        Self::new()
    }
}
