//! Validator selection metrics

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram,
    register_int_counter, register_int_gauge
};

/// Validator selection metrics collector
pub struct SelectionMetrics {
    selections_total: IntCounter,
    selection_time: Histogram,
    selected_validators: IntGauge,
    total_candidates: IntGauge,
    candidate_build_failures: IntCounter,
    performance_updates: IntCounter,
    reputation_updates: IntCounter,
    average_reputation: Gauge,
    average_performance: Gauge,
    stake_concentration: Gauge,
}

impl SelectionMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            selections_total: register_int_counter!(
                "validator_selections_total",
                "Total number of validator selections performed"
            ).unwrap(),
            
            selection_time: register_histogram!(
                "validator_selection_duration_seconds",
                "Time taken to perform validator selection",
                vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            ).unwrap(),
            
            selected_validators: register_int_gauge!(
                "validator_selected_count",
                "Number of validators selected in last selection"
            ).unwrap(),
            
            total_candidates: register_int_gauge!(
                "validator_candidates_total",
                "Total number of validator candidates"
            ).unwrap(),
            
            candidate_build_failures: register_int_counter!(
                "validator_candidate_build_failures_total",
                "Total number of candidate build failures"
            ).unwrap(),
            
            performance_updates: register_int_counter!(
                "validator_performance_updates_total",
                "Total number of performance updates"
            ).unwrap(),
            
            reputation_updates: register_int_counter!(
                "validator_reputation_updates_total",
                "Total number of reputation updates"
            ).unwrap(),
            
            average_reputation: register_gauge!(
                "validator_average_reputation",
                "Average reputation score of selected validators"
            ).unwrap(),
            
            average_performance: register_gauge!(
                "validator_average_performance",
                "Average performance score of selected validators"
            ).unwrap(),
            
            stake_concentration: register_gauge!(
                "validator_stake_concentration",
                "Stake concentration ratio of selected validators"
            ).unwrap(),
        }
    }
    
    /// Record selection time
    pub fn record_selection_time(&self, duration: f64) {
        self.selection_time.observe(duration);
    }
    
    /// Set selected validators count
    pub fn set_selected_validators(&self, count: i64) {
        self.selected_validators.set(count);
    }
    
    /// Set total candidates count
    pub fn set_total_candidates(&self, count: i64) {
        self.total_candidates.set(count);
    }
    
    /// Increment selections
    pub fn increment_selections(&self) {
        self.selections_total.inc();
    }
    
    /// Increment candidate build failures
    pub fn increment_candidate_build_failures(&self) {
        self.candidate_build_failures.inc();
    }
    
    /// Increment performance updates
    pub fn increment_performance_updates(&self) {
        self.performance_updates.inc();
    }
    
    /// Increment reputation updates
    pub fn increment_reputation_updates(&self) {
        self.reputation_updates.inc();
    }
    
    /// Set average reputation
    pub fn set_average_reputation(&self, reputation: f64) {
        self.average_reputation.set(reputation);
    }
    
    /// Set average performance
    pub fn set_average_performance(&self, performance: f64) {
        self.average_performance.set(performance);
    }
    
    /// Set stake concentration
    pub fn set_stake_concentration(&self, concentration: f64) {
        self.stake_concentration.set(concentration);
    }
    
    /// Get average selection time
    pub fn get_average_selection_time(&self) -> f64 {
        // TODO: Calculate from histogram
        0.0
    }
}

impl Default for SelectionMetrics {
    fn default() -> Self {
        Self::new()
    }
}
