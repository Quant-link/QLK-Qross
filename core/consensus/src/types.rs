//! Core types for the consensus module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Validator identifier
pub type ValidatorId = String;

/// Proposal identifier
pub type ProposalId = Uuid;

/// Stake amount (in smallest unit)
pub type Stake = u128;

/// Reputation score (0-100)
pub type ReputationScore = u8;

/// Consensus message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMessage {
    pub id: Uuid,
    pub message_type: MessageType,
    pub proposal_id: Option<ProposalId>,
    pub proposal: Option<Proposal>,
    pub vote: Option<Vote>,
    pub sender: ValidatorId,
    pub timestamp: DateTime<Utc>,
    pub signature: Option<Signature>,
}

/// Message type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    Proposal,
    Vote,
    Prepare,
    Commit,
}

/// Proposal for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: ProposalId,
    pub proposal_type: ProposalType,
    pub data: Vec<u8>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub proposer: ValidatorId,
    pub timestamp: DateTime<Utc>,
    pub timeout: DateTime<Utc>,
}

/// Proposal type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalType {
    CrossChainTransfer,
    StateUpdate,
    ValidatorChange,
    ParameterUpdate,
    Emergency,
    Custom(String),
}

/// Vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub proposal_id: ProposalId,
    pub vote_type: VoteType,
    pub voter: ValidatorId,
    pub timestamp: DateTime<Utc>,
}

/// Vote type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoteType {
    Approve,
    Reject,
    Abstain,
}

/// Cryptographic signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub r: Vec<u8>,
    pub s: Vec<u8>,
    pub v: u8,
}

/// Consensus state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusState {
    Idle,
    Proposing,
    Voting,
    Preparing,
    Committing,
    Finalizing,
}

/// Consensus response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResponse {
    pub success: bool,
    pub message: String,
    pub data: Option<Vec<u8>>,
}

/// Finalized proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizedProposal {
    pub proposal_id: ProposalId,
    pub proposal: Proposal,
    pub votes: Vec<Vote>,
    pub finalized_at: DateTime<Utc>,
    pub result: ConsensusResult,
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusResult {
    Approved,
    Rejected,
    Timeout,
}

/// Validator information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub id: ValidatorId,
    pub stake: Stake,
    pub reputation: ReputationScore,
    pub public_key: Vec<u8>,
    pub network_address: String,
    pub is_active: bool,
    pub joined_at: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

/// Consensus configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Minimum stake required to participate
    pub min_stake: Stake,
    
    /// Maximum number of validators
    pub max_validators: usize,
    
    /// Voting timeout in seconds
    pub voting_timeout: u64,
    
    /// Minimum approval percentage (0-100)
    pub min_approval_percentage: u8,
    
    /// Byzantine fault tolerance threshold (f in 2f+1)
    pub byzantine_threshold: usize,
    
    /// Reputation decay rate per epoch
    pub reputation_decay_rate: f64,
    
    /// Slashing parameters
    pub slashing_config: SlashingConfig,
}

/// Slashing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingConfig {
    /// Light slashing percentage for incorrect attestation
    pub light_slashing_percentage: u8,
    
    /// Medium slashing percentage for conflicting votes
    pub medium_slashing_percentage: u8,
    
    /// Severe slashing percentage for double signing
    pub severe_slashing_percentage: u8,
    
    /// Minimum evidence age in blocks
    pub min_evidence_age: u64,
    
    /// Maximum evidence age in blocks
    pub max_evidence_age: u64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_stake: 1_000_000_000_000_000_000, // 1 token with 18 decimals
            max_validators: 100,
            voting_timeout: 30,
            min_approval_percentage: 67,
            byzantine_threshold: 33,
            reputation_decay_rate: 0.01,
            slashing_config: SlashingConfig::default(),
        }
    }
}

impl Default for SlashingConfig {
    fn default() -> Self {
        Self {
            light_slashing_percentage: 5,
            medium_slashing_percentage: 15,
            severe_slashing_percentage: 50,
            min_evidence_age: 1,
            max_evidence_age: 10000,
        }
    }
}

/// Weighted vote for stake-based consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedVote {
    pub vote: Vote,
    pub weight: u128, // Based on stake and reputation
    pub validator_info: ValidatorInfo,
}

/// Consensus round information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRound {
    pub round_number: u64,
    pub proposal_id: ProposalId,
    pub phase: ConsensusPhase,
    pub started_at: DateTime<Utc>,
    pub timeout_at: DateTime<Utc>,
    pub participants: Vec<ValidatorId>,
}

/// Consensus phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusPhase {
    Proposal,
    Voting,
    Prepare,
    Commit,
    Finalize,
}

/// Cross-chain state proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainProof {
    pub chain_id: String,
    pub block_number: u64,
    pub state_root: Vec<u8>,
    pub proof: Vec<u8>,
    pub validators: Vec<ValidatorId>,
    pub signatures: Vec<Signature>,
}
