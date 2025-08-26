//! Metrics collection for trusted setup ceremonies

use prometheus::{Counter, Histogram, IntCounter, IntGauge, register_counter, register_histogram, register_int_counter, register_int_gauge};

/// Ceremony metrics collector
pub struct CeremonyMetrics {
    ceremonies_initiated: IntCounter,
    ceremonies_completed: IntCounter,
    ceremonies_failed: IntCounter,
    ceremony_timeouts: IntCounter,
    ceremony_duration: Histogram,
    beacon_generation_time: Histogram,
    parameter_generation_time: Histogram,
    verification_time: Histogram,
    active_ceremonies: IntGauge,
}

impl CeremonyMetrics {
    pub fn new() -> Self {
        Self {
            ceremonies_initiated: register_int_counter!(
                "trusted_setup_ceremonies_initiated_total",
                "Total number of ceremonies initiated"
            ).unwrap(),
            
            ceremonies_completed: register_int_counter!(
                "trusted_setup_ceremonies_completed_total", 
                "Total number of ceremonies completed"
            ).unwrap(),
            
            ceremonies_failed: register_int_counter!(
                "trusted_setup_ceremonies_failed_total",
                "Total number of ceremonies failed"
            ).unwrap(),
            
            ceremony_timeouts: register_int_counter!(
                "trusted_setup_ceremony_timeouts_total",
                "Total number of ceremony timeouts"
            ).unwrap(),
            
            ceremony_duration: register_histogram!(
                "trusted_setup_ceremony_duration_seconds",
                "Duration of ceremony execution",
                vec![60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0]
            ).unwrap(),
            
            beacon_generation_time: register_histogram!(
                "trusted_setup_beacon_generation_seconds",
                "Time to generate random beacons",
                vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
            ).unwrap(),
            
            parameter_generation_time: register_histogram!(
                "trusted_setup_parameter_generation_seconds", 
                "Time to generate ceremony parameters",
                vec![1.0, 10.0, 60.0, 300.0, 600.0, 1800.0]
            ).unwrap(),
            
            verification_time: register_histogram!(
                "trusted_setup_verification_seconds",
                "Time to verify contributions and parameters",
                vec![0.1, 1.0, 10.0, 60.0, 300.0]
            ).unwrap(),
            
            active_ceremonies: register_int_gauge!(
                "trusted_setup_active_ceremonies",
                "Number of currently active ceremonies"
            ).unwrap(),
        }
    }
    
    pub fn increment_ceremonies_initiated(&self) {
        self.ceremonies_initiated.inc();
    }
    
    pub fn increment_ceremonies_completed(&self) {
        self.ceremonies_completed.inc();
    }
    
    pub fn increment_ceremonies_failed(&self) {
        self.ceremonies_failed.inc();
    }
    
    pub fn increment_ceremony_timeouts(&self) {
        self.ceremony_timeouts.inc();
    }
    
    pub fn record_ceremony_initiation_time(&self, duration: f64) {
        // Record in beacon generation time for now
        self.beacon_generation_time.observe(duration);
    }
    
    pub fn get_total_ceremonies(&self) -> u64 {
        self.ceremonies_initiated.get() as u64
    }
    
    pub fn get_success_rate(&self) -> f64 {
        let completed = self.ceremonies_completed.get() as f64;
        let failed = self.ceremonies_failed.get() as f64;
        let total = completed + failed;
        
        if total > 0.0 {
            completed / total
        } else {
            0.0
        }
    }
    
    pub fn get_average_ceremony_duration(&self) -> f64 {
        // TODO: Calculate from histogram
        0.0
    }
}

impl Default for CeremonyMetrics {
    fn default() -> Self {
        Self::new()
    }
}
