//! Consensus metrics collection and monitoring

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge, 
    register_counter, register_gauge, register_histogram, 
    register_int_counter, register_int_gauge
};
use std::sync::Arc;

/// Consensus metrics collector
pub struct ConsensusMetrics {
    // Proposal metrics
    proposals_total: IntCounter,
    proposals_finalized: IntCounter,
    proposals_rejected: IntCounter,
    proposals_timeout: IntCounter,
    
    // Vote metrics
    votes_total: IntCounter,
    votes_approve: IntCounter,
    votes_reject: IntCounter,
    votes_abstain: IntCounter,
    
    // Round metrics
    rounds_total: IntCounter,
    rounds_active: IntGauge,
    round_duration: Histogram,
    
    // Validator metrics
    validators_active: IntGauge,
    validators_byzantine: IntCounter,
    
    // Performance metrics
    consensus_latency: Histogram,
    message_processing_time: Histogram,
    
    // Network metrics
    messages_sent: IntCounter,
    messages_received: IntCounter,
    messages_invalid: IntCounter,
    
    // Stake metrics
    total_stake: Gauge,
    average_stake: Gauge,
    
    // Reputation metrics
    average_reputation: Gauge,
    reputation_updates: IntCounter,
}

impl ConsensusMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            proposals_total: register_int_counter!(
                "consensus_proposals_total",
                "Total number of proposals submitted"
            ).unwrap(),
            
            proposals_finalized: register_int_counter!(
                "consensus_proposals_finalized_total",
                "Total number of finalized proposals"
            ).unwrap(),
            
            proposals_rejected: register_int_counter!(
                "consensus_proposals_rejected_total",
                "Total number of rejected proposals"
            ).unwrap(),
            
            proposals_timeout: register_int_counter!(
                "consensus_proposals_timeout_total",
                "Total number of timed out proposals"
            ).unwrap(),
            
            votes_total: register_int_counter!(
                "consensus_votes_total",
                "Total number of votes cast"
            ).unwrap(),
            
            votes_approve: register_int_counter!(
                "consensus_votes_approve_total",
                "Total number of approve votes"
            ).unwrap(),
            
            votes_reject: register_int_counter!(
                "consensus_votes_reject_total",
                "Total number of reject votes"
            ).unwrap(),
            
            votes_abstain: register_int_counter!(
                "consensus_votes_abstain_total",
                "Total number of abstain votes"
            ).unwrap(),
            
            rounds_total: register_int_counter!(
                "consensus_rounds_total",
                "Total number of consensus rounds"
            ).unwrap(),
            
            rounds_active: register_int_gauge!(
                "consensus_rounds_active",
                "Number of currently active consensus rounds"
            ).unwrap(),
            
            round_duration: register_histogram!(
                "consensus_round_duration_seconds",
                "Duration of consensus rounds in seconds",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0]
            ).unwrap(),
            
            validators_active: register_int_gauge!(
                "consensus_validators_active",
                "Number of active validators"
            ).unwrap(),
            
            validators_byzantine: register_int_counter!(
                "consensus_validators_byzantine_total",
                "Total number of detected Byzantine validators"
            ).unwrap(),
            
            consensus_latency: register_histogram!(
                "consensus_latency_seconds",
                "Time from proposal to finalization",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0]
            ).unwrap(),
            
            message_processing_time: register_histogram!(
                "consensus_message_processing_seconds",
                "Time to process consensus messages",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            ).unwrap(),
            
            messages_sent: register_int_counter!(
                "consensus_messages_sent_total",
                "Total number of consensus messages sent"
            ).unwrap(),
            
            messages_received: register_int_counter!(
                "consensus_messages_received_total",
                "Total number of consensus messages received"
            ).unwrap(),
            
            messages_invalid: register_int_counter!(
                "consensus_messages_invalid_total",
                "Total number of invalid consensus messages"
            ).unwrap(),
            
            total_stake: register_gauge!(
                "consensus_total_stake",
                "Total stake in the consensus"
            ).unwrap(),
            
            average_stake: register_gauge!(
                "consensus_average_stake",
                "Average stake per validator"
            ).unwrap(),
            
            average_reputation: register_gauge!(
                "consensus_average_reputation",
                "Average reputation score"
            ).unwrap(),
            
            reputation_updates: register_int_counter!(
                "consensus_reputation_updates_total",
                "Total number of reputation updates"
            ).unwrap(),
        }
    }
    
    /// Increment proposal count
    pub fn increment_proposals(&self) {
        self.proposals_total.inc();
    }
    
    /// Increment finalized proposal count
    pub fn increment_finalized(&self) {
        self.proposals_finalized.inc();
    }
    
    /// Increment rejected proposal count
    pub fn increment_rejected(&self) {
        self.proposals_rejected.inc();
    }
    
    /// Increment timeout proposal count
    pub fn increment_timeout(&self) {
        self.proposals_timeout.inc();
    }
    
    /// Increment vote count
    pub fn increment_votes(&self) {
        self.votes_total.inc();
    }
    
    /// Increment approve vote count
    pub fn increment_approve_votes(&self) {
        self.votes_approve.inc();
    }
    
    /// Increment reject vote count
    pub fn increment_reject_votes(&self) {
        self.votes_reject.inc();
    }
    
    /// Increment abstain vote count
    pub fn increment_abstain_votes(&self) {
        self.votes_abstain.inc();
    }
    
    /// Increment round count
    pub fn increment_rounds(&self) {
        self.rounds_total.inc();
    }
    
    /// Set active rounds count
    pub fn set_active_rounds(&self, count: i64) {
        self.rounds_active.set(count);
    }
    
    /// Record round duration
    pub fn record_round_duration(&self, duration_seconds: f64) {
        self.round_duration.observe(duration_seconds);
    }
    
    /// Set active validators count
    pub fn set_active_validators(&self, count: i64) {
        self.validators_active.set(count);
    }
    
    /// Increment Byzantine validator count
    pub fn increment_byzantine_validators(&self) {
        self.validators_byzantine.inc();
    }
    
    /// Record consensus latency
    pub fn record_consensus_latency(&self, latency_seconds: f64) {
        self.consensus_latency.observe(latency_seconds);
    }
    
    /// Record message processing time
    pub fn record_message_processing_time(&self, time_seconds: f64) {
        self.message_processing_time.observe(time_seconds);
    }
    
    /// Increment sent messages count
    pub fn increment_messages_sent(&self) {
        self.messages_sent.inc();
    }
    
    /// Increment received messages count
    pub fn increment_messages_received(&self) {
        self.messages_received.inc();
    }
    
    /// Increment invalid messages count
    pub fn increment_invalid_messages(&self) {
        self.messages_invalid.inc();
    }
    
    /// Set total stake
    pub fn set_total_stake(&self, stake: f64) {
        self.total_stake.set(stake);
    }
    
    /// Set average stake
    pub fn set_average_stake(&self, stake: f64) {
        self.average_stake.set(stake);
    }
    
    /// Set average reputation
    pub fn set_average_reputation(&self, reputation: f64) {
        self.average_reputation.set(reputation);
    }
    
    /// Increment reputation updates count
    pub fn increment_reputation_updates(&self) {
        self.reputation_updates.inc();
    }
    
    /// Get finalized count
    pub fn get_finalized_count(&self) -> u64 {
        self.proposals_finalized.get() as u64
    }
    
    /// Get vote count
    pub fn get_vote_count(&self) -> u64 {
        self.votes_total.get() as u64
    }
    
    /// Get round count
    pub fn get_round_count(&self) -> u64 {
        self.rounds_total.get() as u64
    }
}

impl Default for ConsensusMetrics {
    fn default() -> Self {
        Self::new()
    }
}
