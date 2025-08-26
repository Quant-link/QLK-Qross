//! Slashing metrics collection and monitoring

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram,
    register_int_counter, register_int_gauge
};
use crate::types::PenaltyType;
use std::collections::HashMap;

/// Slashing metrics collector
pub struct SlashingMetrics {
    // Investigation metrics
    investigations_total: IntCounter,
    investigations_active: IntGauge,
    investigations_completed: IntCounter,
    investigations_dismissed: IntCounter,
    investigations_timed_out: IntCounter,
    investigation_duration: Histogram,
    
    // Slashing event metrics
    slashing_events_total: IntCounter,
    slashed_amount_total: Counter,
    slashing_by_type: HashMap<PenaltyType, IntCounter>,
    
    // Evidence metrics
    evidence_submitted: IntCounter,
    evidence_verified: IntCounter,
    evidence_rejected: IntCounter,
    evidence_verification_time: Histogram,
    
    // Redistribution metrics
    redistribution_events: IntCounter,
    redistributed_amount: Counter,
    burned_amount: Counter,
    community_pool_amount: Counter,
    
    // Validator metrics
    validators_slashed: IntCounter,
    validators_under_investigation: IntGauge,
    average_stake_slashed: Gauge,
    
    // Network health metrics
    network_health_score: Gauge,
    byzantine_behavior_detected: IntCounter,
    emergency_halts: IntCounter,
    
    // Performance metrics
    slashing_processing_time: Histogram,
    penalty_calculation_time: Histogram,
    
    // Error metrics
    slashing_errors: IntCounter,
    evidence_errors: IntCounter,
    redistribution_errors: IntCounter,
}

impl SlashingMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let mut slashing_by_type = HashMap::new();
        
        // Register penalty type counters
        for penalty_type in [PenaltyType::Light, PenaltyType::Medium, PenaltyType::Severe, PenaltyType::None] {
            let counter = register_int_counter!(
                format!("slashing_events_{}_total", penalty_type.to_string()),
                format!("Total number of {} slashing events", penalty_type)
            ).unwrap();
            slashing_by_type.insert(penalty_type, counter);
        }
        
        Self {
            investigations_total: register_int_counter!(
                "slashing_investigations_total",
                "Total number of slashing investigations started"
            ).unwrap(),
            
            investigations_active: register_int_gauge!(
                "slashing_investigations_active",
                "Number of currently active investigations"
            ).unwrap(),
            
            investigations_completed: register_int_counter!(
                "slashing_investigations_completed_total",
                "Total number of completed investigations"
            ).unwrap(),
            
            investigations_dismissed: register_int_counter!(
                "slashing_investigations_dismissed_total",
                "Total number of dismissed investigations"
            ).unwrap(),
            
            investigations_timed_out: register_int_counter!(
                "slashing_investigations_timed_out_total",
                "Total number of timed out investigations"
            ).unwrap(),
            
            investigation_duration: register_histogram!(
                "slashing_investigation_duration_seconds",
                "Duration of slashing investigations",
                vec![60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0]
            ).unwrap(),
            
            slashing_events_total: register_int_counter!(
                "slashing_events_total",
                "Total number of slashing events"
            ).unwrap(),
            
            slashed_amount_total: register_counter!(
                "slashing_amount_total",
                "Total amount of stake slashed"
            ).unwrap(),
            
            slashing_by_type,
            
            evidence_submitted: register_int_counter!(
                "slashing_evidence_submitted_total",
                "Total number of evidence submissions"
            ).unwrap(),
            
            evidence_verified: register_int_counter!(
                "slashing_evidence_verified_total",
                "Total number of verified evidence pieces"
            ).unwrap(),
            
            evidence_rejected: register_int_counter!(
                "slashing_evidence_rejected_total",
                "Total number of rejected evidence pieces"
            ).unwrap(),
            
            evidence_verification_time: register_histogram!(
                "slashing_evidence_verification_seconds",
                "Time to verify evidence",
                vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
            ).unwrap(),
            
            redistribution_events: register_int_counter!(
                "slashing_redistribution_events_total",
                "Total number of redistribution events"
            ).unwrap(),
            
            redistributed_amount: register_counter!(
                "slashing_redistributed_amount_total",
                "Total amount redistributed to honest validators"
            ).unwrap(),
            
            burned_amount: register_counter!(
                "slashing_burned_amount_total",
                "Total amount burned"
            ).unwrap(),
            
            community_pool_amount: register_counter!(
                "slashing_community_pool_amount_total",
                "Total amount added to community pool"
            ).unwrap(),
            
            validators_slashed: register_int_counter!(
                "slashing_validators_slashed_total",
                "Total number of validators slashed"
            ).unwrap(),
            
            validators_under_investigation: register_int_gauge!(
                "slashing_validators_under_investigation",
                "Number of validators currently under investigation"
            ).unwrap(),
            
            average_stake_slashed: register_gauge!(
                "slashing_average_stake_slashed",
                "Average amount of stake slashed per event"
            ).unwrap(),
            
            network_health_score: register_gauge!(
                "slashing_network_health_score",
                "Network health score (0-100)"
            ).unwrap(),
            
            byzantine_behavior_detected: register_int_counter!(
                "slashing_byzantine_behavior_detected_total",
                "Total number of Byzantine behavior detections"
            ).unwrap(),
            
            emergency_halts: register_int_counter!(
                "slashing_emergency_halts_total",
                "Total number of emergency halts triggered"
            ).unwrap(),
            
            slashing_processing_time: register_histogram!(
                "slashing_processing_time_seconds",
                "Time to process slashing events",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            ).unwrap(),
            
            penalty_calculation_time: register_histogram!(
                "slashing_penalty_calculation_seconds",
                "Time to calculate penalties",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            ).unwrap(),
            
            slashing_errors: register_int_counter!(
                "slashing_errors_total",
                "Total number of slashing errors"
            ).unwrap(),
            
            evidence_errors: register_int_counter!(
                "slashing_evidence_errors_total",
                "Total number of evidence processing errors"
            ).unwrap(),
            
            redistribution_errors: register_int_counter!(
                "slashing_redistribution_errors_total",
                "Total number of redistribution errors"
            ).unwrap(),
        }
    }
    
    /// Increment investigation count
    pub fn increment_investigations(&self) {
        self.investigations_total.inc();
    }
    
    /// Set active investigations count
    pub fn set_active_investigations(&self, count: i64) {
        self.investigations_active.set(count);
    }
    
    /// Increment completed investigations
    pub fn increment_completed_investigations(&self) {
        self.investigations_completed.inc();
    }
    
    /// Increment dismissed investigations
    pub fn increment_dismissed_investigations(&self) {
        self.investigations_dismissed.inc();
    }
    
    /// Increment timed out investigations
    pub fn increment_timed_out_investigations(&self) {
        self.investigations_timed_out.inc();
    }
    
    /// Record investigation duration
    pub fn record_investigation_duration(&self, duration_seconds: f64) {
        self.investigation_duration.observe(duration_seconds);
    }
    
    /// Increment slashing events
    pub fn increment_slashing_events(&self) {
        self.slashing_events_total.inc();
    }
    
    /// Record slashed amount
    pub fn record_slashing_amount(&self, amount: f64) {
        self.slashed_amount_total.inc_by(amount);
    }
    
    /// Increment penalty type counter
    pub fn increment_penalty_type(&self, penalty_type: &PenaltyType) {
        if let Some(counter) = self.slashing_by_type.get(penalty_type) {
            counter.inc();
        }
    }
    
    /// Increment evidence submitted
    pub fn increment_evidence_submitted(&self) {
        self.evidence_submitted.inc();
    }
    
    /// Increment evidence verified
    pub fn increment_evidence_verified(&self) {
        self.evidence_verified.inc();
    }
    
    /// Increment evidence rejected
    pub fn increment_evidence_rejected(&self) {
        self.evidence_rejected.inc();
    }
    
    /// Record evidence verification time
    pub fn record_evidence_verification_time(&self, time_seconds: f64) {
        self.evidence_verification_time.observe(time_seconds);
    }
    
    /// Increment redistribution events
    pub fn increment_redistribution_events(&self) {
        self.redistribution_events.inc();
    }
    
    /// Record redistributed amount
    pub fn record_redistributed_amount(&self, amount: f64) {
        self.redistributed_amount.inc_by(amount);
    }
    
    /// Record burned amount
    pub fn record_burned_amount(&self, amount: f64) {
        self.burned_amount.inc_by(amount);
    }
    
    /// Record community pool amount
    pub fn record_community_pool_amount(&self, amount: f64) {
        self.community_pool_amount.inc_by(amount);
    }
    
    /// Increment validators slashed
    pub fn increment_validators_slashed(&self) {
        self.validators_slashed.inc();
    }
    
    /// Set validators under investigation
    pub fn set_validators_under_investigation(&self, count: i64) {
        self.validators_under_investigation.set(count);
    }
    
    /// Set average stake slashed
    pub fn set_average_stake_slashed(&self, amount: f64) {
        self.average_stake_slashed.set(amount);
    }
    
    /// Set network health score
    pub fn set_network_health_score(&self, score: f64) {
        self.network_health_score.set(score);
    }
    
    /// Increment Byzantine behavior detected
    pub fn increment_byzantine_behavior(&self) {
        self.byzantine_behavior_detected.inc();
    }
    
    /// Increment emergency halts
    pub fn increment_emergency_halts(&self) {
        self.emergency_halts.inc();
    }
    
    /// Record slashing processing time
    pub fn record_slashing_processing_time(&self, time_seconds: f64) {
        self.slashing_processing_time.observe(time_seconds);
    }
    
    /// Record penalty calculation time
    pub fn record_penalty_calculation_time(&self, time_seconds: f64) {
        self.penalty_calculation_time.observe(time_seconds);
    }
    
    /// Increment slashing errors
    pub fn increment_slashing_errors(&self) {
        self.slashing_errors.inc();
    }
    
    /// Increment evidence errors
    pub fn increment_evidence_errors(&self) {
        self.evidence_errors.inc();
    }
    
    /// Increment redistribution errors
    pub fn increment_redistribution_errors(&self) {
        self.redistribution_errors.inc();
    }
    
    /// Get investigation count
    pub fn get_investigation_count(&self) -> u64 {
        self.investigations_total.get() as u64
    }
    
    /// Get slashing events count
    pub fn get_slashing_events_count(&self) -> u64 {
        self.slashing_events_total.get() as u64
    }
    
    /// Get total slashed amount
    pub fn get_total_slashed_amount(&self) -> f64 {
        self.slashed_amount_total.get()
    }
    
    /// Get evidence verification rate
    pub fn get_evidence_verification_rate(&self) -> f64 {
        let submitted = self.evidence_submitted.get() as f64;
        let verified = self.evidence_verified.get() as f64;
        
        if submitted > 0.0 {
            (verified / submitted) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for SlashingMetrics {
    fn default() -> Self {
        Self::new()
    }
}
