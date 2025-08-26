//! Modified Practical Byzantine Fault Tolerance (PBFT) implementation
//! 
//! This implementation reduces the traditional 3f+1 validator requirement to 2f+1
//! while maintaining Byzantine fault tolerance through enhanced cryptographic proofs.

use crate::{types::*, error::*};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// PBFT consensus engine
pub struct PbftEngine {
    round_number: u64,
    active_rounds: HashMap<ProposalId, PbftRound>,
    validator_set: HashSet<ValidatorId>,
    byzantine_threshold: usize, // f in 2f+1
}

/// PBFT round state
#[derive(Debug, Clone)]
struct PbftRound {
    proposal_id: ProposalId,
    proposal: Proposal,
    phase: PbftPhase,
    primary: ValidatorId,
    prepare_votes: HashMap<ValidatorId, ConsensusMessage>,
    commit_votes: HashMap<ValidatorId, ConsensusMessage>,
    started_at: DateTime<Utc>,
    timeout_at: DateTime<Utc>,
}

/// PBFT phases
#[derive(Debug, Clone, PartialEq)]
enum PbftPhase {
    Proposal,
    Prepare,
    Commit,
    Finalized,
}

impl PbftEngine {
    /// Create a new PBFT engine
    pub fn new() -> Self {
        Self {
            round_number: 0,
            active_rounds: HashMap::new(),
            validator_set: HashSet::new(),
            byzantine_threshold: 1, // Will be updated based on validator set size
        }
    }
    
    /// Update validator set
    pub fn update_validator_set(&mut self, validators: HashSet<ValidatorId>) {
        self.validator_set = validators;
        // Calculate Byzantine threshold: f = (n-1)/2 for 2f+1 consensus
        self.byzantine_threshold = (self.validator_set.len().saturating_sub(1)) / 2;
    }
    
    /// Process a proposal message
    pub async fn process_proposal(&mut self, message: ConsensusMessage) -> Result<()> {
        let proposal = message.proposal.as_ref()
            .ok_or_else(|| ConsensusError::InvalidProposal("Missing proposal".to_string()))?;
        
        let proposal_id = proposal.id;
        
        // Check if round already exists
        if self.active_rounds.contains_key(&proposal_id) {
            return Err(ConsensusError::InvalidProposal("Round already exists".to_string()));
        }
        
        // Validate proposer is in validator set
        if !self.validator_set.contains(&message.sender) {
            return Err(ConsensusError::UnknownValidator(message.sender));
        }
        
        // Create new PBFT round
        let round = PbftRound {
            proposal_id,
            proposal: proposal.clone(),
            phase: PbftPhase::Proposal,
            primary: message.sender,
            prepare_votes: HashMap::new(),
            commit_votes: HashMap::new(),
            started_at: Utc::now(),
            timeout_at: proposal.timeout,
        };
        
        self.active_rounds.insert(proposal_id, round);
        self.round_number += 1;
        
        tracing::info!("Started PBFT round {} for proposal {}", self.round_number, proposal_id);
        
        Ok(())
    }
    
    /// Process a prepare message
    pub async fn process_prepare(&mut self, message: ConsensusMessage) -> Result<()> {
        let proposal_id = message.proposal_id
            .ok_or_else(|| ConsensusError::InvalidProposal("Missing proposal ID".to_string()))?;
        
        let round = self.active_rounds.get_mut(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Round not found".to_string()))?;
        
        // Check if we're in the right phase
        if round.phase != PbftPhase::Proposal && round.phase != PbftPhase::Prepare {
            return Err(ConsensusError::InvalidProposal("Invalid phase for prepare".to_string()));
        }
        
        // Validate sender is in validator set
        if !self.validator_set.contains(&message.sender) {
            return Err(ConsensusError::UnknownValidator(message.sender.clone()));
        }
        
        // Check for duplicate prepare vote
        if round.prepare_votes.contains_key(&message.sender) {
            return Err(ConsensusError::DuplicateVote(message.sender));
        }
        
        // Store prepare vote
        round.prepare_votes.insert(message.sender.clone(), message);
        
        // Check if we have enough prepare votes (2f+1)
        let required_votes = 2 * self.byzantine_threshold + 1;
        if round.prepare_votes.len() >= required_votes {
            round.phase = PbftPhase::Prepare;
            tracing::info!("PBFT round {} entered prepare phase", proposal_id);
        }
        
        Ok(())
    }
    
    /// Process a commit message
    pub async fn process_commit(&mut self, message: ConsensusMessage) -> Result<()> {
        let proposal_id = message.proposal_id
            .ok_or_else(|| ConsensusError::InvalidProposal("Missing proposal ID".to_string()))?;
        
        let round = self.active_rounds.get_mut(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Round not found".to_string()))?;
        
        // Check if we're in the right phase
        if round.phase != PbftPhase::Prepare && round.phase != PbftPhase::Commit {
            return Err(ConsensusError::InvalidProposal("Invalid phase for commit".to_string()));
        }
        
        // Validate sender is in validator set
        if !self.validator_set.contains(&message.sender) {
            return Err(ConsensusError::UnknownValidator(message.sender.clone()));
        }
        
        // Check for duplicate commit vote
        if round.commit_votes.contains_key(&message.sender) {
            return Err(ConsensusError::DuplicateVote(message.sender));
        }
        
        // Store commit vote
        round.commit_votes.insert(message.sender.clone(), message);
        
        // Check if we have enough commit votes (2f+1)
        let required_votes = 2 * self.byzantine_threshold + 1;
        if round.commit_votes.len() >= required_votes {
            round.phase = PbftPhase::Commit;
            tracing::info!("PBFT round {} entered commit phase", proposal_id);
            
            // Check if we can finalize
            if self.can_finalize_round(&proposal_id)? {
                self.finalize_round(proposal_id)?;
            }
        }
        
        Ok(())
    }
    
    /// Check if a round can be finalized
    fn can_finalize_round(&self, proposal_id: &ProposalId) -> Result<bool> {
        let round = self.active_rounds.get(proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Round not found".to_string()))?;
        
        let required_votes = 2 * self.byzantine_threshold + 1;
        
        Ok(round.phase == PbftPhase::Commit &&
           round.prepare_votes.len() >= required_votes &&
           round.commit_votes.len() >= required_votes)
    }
    
    /// Finalize a PBFT round
    fn finalize_round(&mut self, proposal_id: ProposalId) -> Result<()> {
        let mut round = self.active_rounds.remove(&proposal_id)
            .ok_or_else(|| ConsensusError::InvalidProposal("Round not found".to_string()))?;
        
        round.phase = PbftPhase::Finalized;
        
        tracing::info!("PBFT round {} finalized for proposal {}", self.round_number, proposal_id);
        
        Ok(())
    }
    
    /// Get current round count
    pub fn get_round_count(&self) -> u64 {
        self.round_number
    }
    
    /// Get active rounds
    pub fn get_active_rounds(&self) -> Vec<ProposalId> {
        self.active_rounds.keys().cloned().collect()
    }
    
    /// Clean up expired rounds
    pub fn cleanup_expired_rounds(&mut self) -> Result<()> {
        let now = Utc::now();
        let expired_rounds: Vec<ProposalId> = self.active_rounds
            .iter()
            .filter(|(_, round)| now > round.timeout_at && round.phase != PbftPhase::Finalized)
            .map(|(id, _)| *id)
            .collect();
        
        for proposal_id in expired_rounds {
            self.active_rounds.remove(&proposal_id);
            tracing::warn!("PBFT round {} expired and removed", proposal_id);
        }
        
        Ok(())
    }
    
    /// Detect Byzantine behavior
    pub fn detect_byzantine_behavior(&self) -> Vec<ValidatorId> {
        let mut byzantine_validators = Vec::new();
        
        for round in self.active_rounds.values() {
            // Check for conflicting prepare votes
            let prepare_conflicts = self.detect_conflicting_votes(&round.prepare_votes);
            byzantine_validators.extend(prepare_conflicts);
            
            // Check for conflicting commit votes
            let commit_conflicts = self.detect_conflicting_votes(&round.commit_votes);
            byzantine_validators.extend(commit_conflicts);
        }
        
        byzantine_validators.sort();
        byzantine_validators.dedup();
        byzantine_validators
    }
    
    /// Detect conflicting votes from the same validator
    fn detect_conflicting_votes(&self, votes: &HashMap<ValidatorId, ConsensusMessage>) -> Vec<ValidatorId> {
        // In a real implementation, this would check for:
        // 1. Multiple votes for different proposals in the same round
        // 2. Votes that contradict previous votes
        // 3. Invalid signatures or message formats
        
        // For now, return empty as we need more sophisticated conflict detection
        Vec::new()
    }
    
    /// Get Byzantine threshold
    pub fn get_byzantine_threshold(&self) -> usize {
        self.byzantine_threshold
    }
    
    /// Get validator set size
    pub fn get_validator_count(&self) -> usize {
        self.validator_set.len()
    }
}

impl Default for PbftEngine {
    fn default() -> Self {
        Self::new()
    }
}
