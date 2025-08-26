//! Quantlink Qross Consensus Aggregation Engine
//! 
//! This module implements a heterogeneous consensus aggregation system that consolidates
//! finality mechanisms of different blockchains within a unified state machine using
//! a modified Practical Byzantine Fault Tolerance (PBFT) algorithm.

pub mod aggregator;
pub mod pbft;
pub mod voting;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub use error::{ConsensusError, Result};
pub use types::*;

/// Trait for consensus participants (validators)
#[async_trait]
pub trait ConsensusParticipant: Send + Sync {
    /// Get validator ID
    fn get_validator_id(&self) -> ValidatorId;
    
    /// Get validator stake
    fn get_stake(&self) -> Stake;
    
    /// Get validator reputation score
    fn get_reputation(&self) -> ReputationScore;
    
    /// Sign a consensus message
    async fn sign_message(&self, message: &ConsensusMessage) -> Result<Signature>;
    
    /// Verify a signature
    async fn verify_signature(&self, message: &ConsensusMessage, signature: &Signature, validator_id: &ValidatorId) -> Result<bool>;
}

/// Trait for consensus state machine
#[async_trait]
pub trait ConsensusStateMachine: Send + Sync {
    /// Process a consensus message
    async fn process_message(&mut self, message: ConsensusMessage) -> Result<ConsensusResponse>;
    
    /// Get current consensus state
    fn get_state(&self) -> ConsensusState;
    
    /// Check if consensus is reached for a proposal
    fn is_consensus_reached(&self, proposal_id: &ProposalId) -> bool;
    
    /// Get finalized proposals
    fn get_finalized_proposals(&self) -> Vec<FinalizedProposal>;
}

/// Main consensus aggregator
pub struct ConsensusAggregator {
    validator_id: ValidatorId,
    participants: HashMap<ValidatorId, Box<dyn ConsensusParticipant>>,
    state_machine: Box<dyn ConsensusStateMachine>,
    voting_engine: voting::VotingEngine,
    pbft_engine: pbft::PbftEngine,
    metrics: metrics::ConsensusMetrics,
}

impl ConsensusAggregator {
    /// Create a new consensus aggregator
    pub fn new(
        validator_id: ValidatorId,
        state_machine: Box<dyn ConsensusStateMachine>,
    ) -> Self {
        Self {
            validator_id,
            participants: HashMap::new(),
            state_machine,
            voting_engine: voting::VotingEngine::new(),
            pbft_engine: pbft::PbftEngine::new(),
            metrics: metrics::ConsensusMetrics::new(),
        }
    }
    
    /// Add a validator to the consensus
    pub fn add_validator(&mut self, validator: Box<dyn ConsensusParticipant>) {
        let validator_id = validator.get_validator_id();
        self.participants.insert(validator_id, validator);
    }
    
    /// Remove a validator from the consensus
    pub fn remove_validator(&mut self, validator_id: &ValidatorId) {
        self.participants.remove(validator_id);
    }
    
    /// Propose a new consensus item
    pub async fn propose(&mut self, proposal: Proposal) -> Result<ProposalId> {
        let proposal_id = Uuid::new_v4();
        
        // Create consensus message
        let message = ConsensusMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::Proposal,
            proposal_id: Some(proposal_id),
            proposal: Some(proposal),
            vote: None,
            sender: self.validator_id.clone(),
            timestamp: Utc::now(),
            signature: None,
        };
        
        // Sign the message
        let signature = self.sign_message(&message).await?;
        let mut signed_message = message;
        signed_message.signature = Some(signature);
        
        // Process through PBFT
        self.pbft_engine.process_proposal(signed_message).await?;
        
        // Update metrics
        self.metrics.increment_proposals();
        
        Ok(proposal_id)
    }
    
    /// Vote on a proposal
    pub async fn vote(&mut self, proposal_id: ProposalId, vote_type: VoteType) -> Result<()> {
        let vote = Vote {
            proposal_id,
            vote_type,
            voter: self.validator_id.clone(),
            timestamp: Utc::now(),
        };
        
        let message = ConsensusMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::Vote,
            proposal_id: Some(proposal_id),
            proposal: None,
            vote: Some(vote),
            sender: self.validator_id.clone(),
            timestamp: Utc::now(),
            signature: None,
        };
        
        // Sign the message
        let signature = self.sign_message(&message).await?;
        let mut signed_message = message;
        signed_message.signature = Some(signature);
        
        // Process through voting engine
        self.voting_engine.process_vote(signed_message).await?;
        
        // Update metrics
        self.metrics.increment_votes();
        
        Ok(())
    }
    
    /// Process incoming consensus message
    pub async fn process_message(&mut self, message: ConsensusMessage) -> Result<()> {
        // Verify signature
        if let Some(signature) = &message.signature {
            if let Some(participant) = self.participants.get(&message.sender) {
                let is_valid = participant.verify_signature(&message, signature, &message.sender).await?;
                if !is_valid {
                    return Err(ConsensusError::InvalidSignature);
                }
            } else {
                return Err(ConsensusError::UnknownValidator(message.sender));
            }
        } else {
            return Err(ConsensusError::MissingSignature);
        }
        
        // Process based on message type
        match message.message_type {
            MessageType::Proposal => {
                self.pbft_engine.process_proposal(message).await?;
            }
            MessageType::Vote => {
                self.voting_engine.process_vote(message).await?;
            }
            MessageType::Prepare => {
                self.pbft_engine.process_prepare(message).await?;
            }
            MessageType::Commit => {
                self.pbft_engine.process_commit(message).await?;
            }
        }
        
        // Check for consensus
        self.check_consensus().await?;
        
        Ok(())
    }
    
    /// Check if consensus is reached and finalize proposals
    async fn check_consensus(&mut self) -> Result<()> {
        let finalized = self.voting_engine.get_finalized_proposals();
        
        for proposal in finalized {
            // Process through state machine
            let response = self.state_machine.process_message(ConsensusMessage {
                id: Uuid::new_v4(),
                message_type: MessageType::Commit,
                proposal_id: Some(proposal.proposal_id),
                proposal: Some(proposal.proposal),
                vote: None,
                sender: self.validator_id.clone(),
                timestamp: Utc::now(),
                signature: None,
            }).await?;
            
            // Update metrics
            self.metrics.increment_finalized();
            
            tracing::info!("Proposal {} finalized with response: {:?}", proposal.proposal_id, response);
        }
        
        Ok(())
    }
    
    /// Get current consensus statistics
    pub fn get_statistics(&self) -> ConsensusStatistics {
        ConsensusStatistics {
            total_validators: self.participants.len(),
            active_proposals: self.voting_engine.get_active_proposal_count(),
            finalized_proposals: self.metrics.get_finalized_count(),
            total_votes: self.metrics.get_vote_count(),
            consensus_rounds: self.pbft_engine.get_round_count(),
        }
    }
    
    /// Sign a consensus message
    async fn sign_message(&self, message: &ConsensusMessage) -> Result<Signature> {
        // In a real implementation, this would use the validator's private key
        // For now, we'll create a placeholder signature
        Ok(Signature {
            r: vec![0u8; 32],
            s: vec![0u8; 32],
            v: 0,
        })
    }
}

/// Consensus statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStatistics {
    pub total_validators: usize,
    pub active_proposals: usize,
    pub finalized_proposals: u64,
    pub total_votes: u64,
    pub consensus_rounds: u64,
}
