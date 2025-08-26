//! Metrics collection for proof aggregation protocol

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram,
    register_int_counter, register_int_gauge
};

/// Aggregation metrics collector
pub struct AggregationMetrics {
    // Aggregation request metrics
    aggregation_requests: IntCounter,
    active_aggregations: IntGauge,
    completed_aggregations: IntCounter,
    failed_aggregations: IntCounter,
    
    // Timing metrics
    submission_time: Histogram,
    aggregation_time: Histogram,
    dependency_resolution_time: Histogram,
    batch_processing_time: Histogram,
    finality_time: Histogram,
    
    // Dependency metrics
    dependency_graph_size: Histogram,
    dependency_depth: Histogram,
    cyclic_dependencies: IntCounter,
    dependency_timeouts: IntCounter,
    
    // Batch processing metrics
    batch_size: Histogram,
    batch_verification_time: Histogram,
    batch_failures: IntCounter,
    
    // Validator allocation metrics
    validator_allocations: IntCounter,
    allocation_time: Histogram,
    allocation_failures: IntCounter,
    
    // Finality metrics
    finality_submissions: IntCounter,
    finality_successes: IntCounter,
    finality_rejections: IntCounter,
    finality_timeouts: IntCounter,
    
    // Emergency halt metrics
    emergency_halts: IntCounter,
    network_partitions: IntCounter,
    performance_degradations: IntCounter,
    
    // Proof composition metrics
    composition_depth: Histogram,
    compression_ratio: Histogram,
    recursive_compositions: IntCounter,
    
    // Resource utilization metrics
    memory_usage: Gauge,
    cpu_utilization: Gauge,
    network_bandwidth: Gauge,
    
    // Cache metrics
    cache_hits: IntCounter,
    cache_misses: IntCounter,
    cache_size: IntGauge,
    
    // Error metrics
    aggregation_errors: IntCounter,
    consensus_errors: IntCounter,
    state_sync_errors: IntCounter,
}

impl AggregationMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            aggregation_requests: register_int_counter!(
                "proof_aggregation_requests_total",
                "Total number of proof aggregation requests"
            ).unwrap(),
            
            active_aggregations: register_int_gauge!(
                "proof_aggregation_active",
                "Number of currently active aggregations"
            ).unwrap(),
            
            completed_aggregations: register_int_counter!(
                "proof_aggregation_completed_total",
                "Total number of completed aggregations"
            ).unwrap(),
            
            failed_aggregations: register_int_counter!(
                "proof_aggregation_failed_total",
                "Total number of failed aggregations"
            ).unwrap(),
            
            submission_time: register_histogram!(
                "proof_aggregation_submission_duration_seconds",
                "Time taken to submit proofs for aggregation",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            ).unwrap(),
            
            aggregation_time: register_histogram!(
                "proof_aggregation_duration_seconds",
                "Time taken to complete proof aggregation",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0]
            ).unwrap(),
            
            dependency_resolution_time: register_histogram!(
                "proof_aggregation_dependency_resolution_seconds",
                "Time taken to resolve dependencies",
                vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
            ).unwrap(),
            
            batch_processing_time: register_histogram!(
                "proof_aggregation_batch_processing_seconds",
                "Time taken to process proof batches",
                vec![0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
            ).unwrap(),
            
            finality_time: register_histogram!(
                "proof_aggregation_finality_duration_seconds",
                "Time taken for finality determination",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
            ).unwrap(),
            
            dependency_graph_size: register_histogram!(
                "proof_aggregation_dependency_graph_size",
                "Size of dependency graphs",
                vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
            ).unwrap(),
            
            dependency_depth: register_histogram!(
                "proof_aggregation_dependency_depth",
                "Depth of dependency graphs",
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            ).unwrap(),
            
            cyclic_dependencies: register_int_counter!(
                "proof_aggregation_cyclic_dependencies_total",
                "Total number of cyclic dependencies detected"
            ).unwrap(),
            
            dependency_timeouts: register_int_counter!(
                "proof_aggregation_dependency_timeouts_total",
                "Total number of dependency resolution timeouts"
            ).unwrap(),
            
            batch_size: register_histogram!(
                "proof_aggregation_batch_size",
                "Size of proof batches",
                vec![1.0, 5.0, 10.0, 20.0, 32.0, 50.0, 64.0, 100.0]
            ).unwrap(),
            
            batch_verification_time: register_histogram!(
                "proof_aggregation_batch_verification_seconds",
                "Time taken to verify proof batches",
                vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
            ).unwrap(),
            
            batch_failures: register_int_counter!(
                "proof_aggregation_batch_failures_total",
                "Total number of batch processing failures"
            ).unwrap(),
            
            validator_allocations: register_int_counter!(
                "proof_aggregation_validator_allocations_total",
                "Total number of validator allocations"
            ).unwrap(),
            
            allocation_time: register_histogram!(
                "proof_aggregation_allocation_duration_seconds",
                "Time taken for validator allocation",
                vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
            ).unwrap(),
            
            allocation_failures: register_int_counter!(
                "proof_aggregation_allocation_failures_total",
                "Total number of validator allocation failures"
            ).unwrap(),
            
            finality_submissions: register_int_counter!(
                "proof_aggregation_finality_submissions_total",
                "Total number of finality submissions"
            ).unwrap(),
            
            finality_successes: register_int_counter!(
                "proof_aggregation_finality_successes_total",
                "Total number of successful finality determinations"
            ).unwrap(),
            
            finality_rejections: register_int_counter!(
                "proof_aggregation_finality_rejections_total",
                "Total number of finality rejections"
            ).unwrap(),
            
            finality_timeouts: register_int_counter!(
                "proof_aggregation_finality_timeouts_total",
                "Total number of finality timeouts"
            ).unwrap(),
            
            emergency_halts: register_int_counter!(
                "proof_aggregation_emergency_halts_total",
                "Total number of emergency halts triggered"
            ).unwrap(),
            
            network_partitions: register_int_counter!(
                "proof_aggregation_network_partitions_total",
                "Total number of network partitions detected"
            ).unwrap(),
            
            performance_degradations: register_int_counter!(
                "proof_aggregation_performance_degradations_total",
                "Total number of performance degradations detected"
            ).unwrap(),
            
            composition_depth: register_histogram!(
                "proof_aggregation_composition_depth",
                "Depth of recursive proof composition",
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            ).unwrap(),
            
            compression_ratio: register_histogram!(
                "proof_aggregation_compression_ratio",
                "Compression ratio achieved by aggregation",
                vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
            ).unwrap(),
            
            recursive_compositions: register_int_counter!(
                "proof_aggregation_recursive_compositions_total",
                "Total number of recursive proof compositions"
            ).unwrap(),
            
            memory_usage: register_gauge!(
                "proof_aggregation_memory_usage_bytes",
                "Current memory usage for aggregation operations"
            ).unwrap(),
            
            cpu_utilization: register_gauge!(
                "proof_aggregation_cpu_utilization_percent",
                "Current CPU utilization for aggregation operations"
            ).unwrap(),
            
            network_bandwidth: register_gauge!(
                "proof_aggregation_network_bandwidth_mbps",
                "Current network bandwidth usage"
            ).unwrap(),
            
            cache_hits: register_int_counter!(
                "proof_aggregation_cache_hits_total",
                "Total number of cache hits"
            ).unwrap(),
            
            cache_misses: register_int_counter!(
                "proof_aggregation_cache_misses_total",
                "Total number of cache misses"
            ).unwrap(),
            
            cache_size: register_int_gauge!(
                "proof_aggregation_cache_size",
                "Current cache size"
            ).unwrap(),
            
            aggregation_errors: register_int_counter!(
                "proof_aggregation_errors_total",
                "Total number of aggregation errors"
            ).unwrap(),
            
            consensus_errors: register_int_counter!(
                "proof_aggregation_consensus_errors_total",
                "Total number of consensus integration errors"
            ).unwrap(),
            
            state_sync_errors: register_int_counter!(
                "proof_aggregation_state_sync_errors_total",
                "Total number of state synchronization errors"
            ).unwrap(),
        }
    }
    
    /// Increment aggregation requests
    pub fn increment_aggregation_requests(&self) {
        self.aggregation_requests.inc();
    }
    
    /// Set active aggregations count
    pub fn set_active_aggregations(&self, count: i64) {
        self.active_aggregations.set(count);
    }
    
    /// Increment completed aggregations
    pub fn increment_completed_aggregations(&self) {
        self.completed_aggregations.inc();
    }
    
    /// Increment failed aggregations
    pub fn increment_failed_aggregations(&self) {
        self.failed_aggregations.inc();
    }
    
    /// Record submission time
    pub fn record_submission_time(&self, duration: f64) {
        self.submission_time.observe(duration);
    }
    
    /// Record aggregation time
    pub fn record_aggregation_time(&self, duration: f64) {
        self.aggregation_time.observe(duration);
    }
    
    /// Record dependency resolution time
    pub fn record_dependency_resolution_time(&self, duration: f64) {
        self.dependency_resolution_time.observe(duration);
    }
    
    /// Record batch processing time
    pub fn record_batch_processing_time(&self, duration: f64) {
        self.batch_processing_time.observe(duration);
    }
    
    /// Record finality time
    pub fn record_finality_time(&self, duration: f64) {
        self.finality_time.observe(duration);
    }
    
    /// Record dependency graph size
    pub fn record_dependency_graph_size(&self, size: f64) {
        self.dependency_graph_size.observe(size);
    }
    
    /// Record dependency depth
    pub fn record_dependency_depth(&self, depth: f64) {
        self.dependency_depth.observe(depth);
    }
    
    /// Increment cyclic dependencies
    pub fn increment_cyclic_dependencies(&self) {
        self.cyclic_dependencies.inc();
    }
    
    /// Increment dependency timeouts
    pub fn increment_dependency_timeouts(&self) {
        self.dependency_timeouts.inc();
    }
    
    /// Record batch size
    pub fn record_batch_size(&self, size: f64) {
        self.batch_size.observe(size);
    }
    
    /// Record batch verification time
    pub fn record_batch_verification_time(&self, duration: f64) {
        self.batch_verification_time.observe(duration);
    }
    
    /// Increment batch failures
    pub fn increment_batch_failures(&self) {
        self.batch_failures.inc();
    }
    
    /// Increment validator allocations
    pub fn increment_validator_allocations(&self) {
        self.validator_allocations.inc();
    }
    
    /// Record allocation time
    pub fn record_allocation_time(&self, duration: f64) {
        self.allocation_time.observe(duration);
    }
    
    /// Increment allocation failures
    pub fn increment_allocation_failures(&self) {
        self.allocation_failures.inc();
    }
    
    /// Increment finality submissions
    pub fn increment_finality_submissions(&self) {
        self.finality_submissions.inc();
    }
    
    /// Increment finality successes
    pub fn increment_finality_successes(&self) {
        self.finality_successes.inc();
    }
    
    /// Increment finality rejections
    pub fn increment_finality_rejections(&self) {
        self.finality_rejections.inc();
    }
    
    /// Increment finality timeouts
    pub fn increment_finality_timeouts(&self) {
        self.finality_timeouts.inc();
    }
    
    /// Increment emergency halts
    pub fn increment_emergency_halts(&self) {
        self.emergency_halts.inc();
    }
    
    /// Increment network partitions
    pub fn increment_network_partitions(&self) {
        self.network_partitions.inc();
    }
    
    /// Increment performance degradations
    pub fn increment_performance_degradations(&self) {
        self.performance_degradations.inc();
    }
    
    /// Record composition depth
    pub fn record_composition_depth(&self, depth: f64) {
        self.composition_depth.observe(depth);
    }
    
    /// Record compression ratio
    pub fn record_compression_ratio(&self, ratio: f64) {
        self.compression_ratio.observe(ratio);
    }
    
    /// Increment recursive compositions
    pub fn increment_recursive_compositions(&self) {
        self.recursive_compositions.inc();
    }
    
    /// Set memory usage
    pub fn set_memory_usage(&self, bytes: f64) {
        self.memory_usage.set(bytes);
    }
    
    /// Set CPU utilization
    pub fn set_cpu_utilization(&self, percent: f64) {
        self.cpu_utilization.set(percent);
    }
    
    /// Set network bandwidth
    pub fn set_network_bandwidth(&self, mbps: f64) {
        self.network_bandwidth.set(mbps);
    }
    
    /// Increment cache hits
    pub fn increment_cache_hits(&self) {
        self.cache_hits.inc();
    }
    
    /// Increment cache misses
    pub fn increment_cache_misses(&self) {
        self.cache_misses.inc();
    }
    
    /// Set cache size
    pub fn set_cache_size(&self, size: i64) {
        self.cache_size.set(size);
    }
    
    /// Increment aggregation errors
    pub fn increment_aggregation_errors(&self) {
        self.aggregation_errors.inc();
    }
    
    /// Increment consensus errors
    pub fn increment_consensus_errors(&self) {
        self.consensus_errors.inc();
    }
    
    /// Increment state sync errors
    pub fn increment_state_sync_errors(&self) {
        self.state_sync_errors.inc();
    }
    
    /// Get total aggregations
    pub fn get_total_aggregations(&self) -> u64 {
        self.aggregation_requests.get() as u64
    }
    
    /// Get average aggregation time
    pub fn get_average_aggregation_time(&self) -> f64 {
        // TODO: Calculate from histogram
        0.0
    }
    
    /// Get success rate
    pub fn get_success_rate(&self) -> f64 {
        let completed = self.completed_aggregations.get() as f64;
        let failed = self.failed_aggregations.get() as f64;
        let total = completed + failed;
        
        if total > 0.0 {
            completed / total
        } else {
            0.0
        }
    }
    
    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.get() as f64;
        let misses = self.cache_misses.get() as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

impl Default for AggregationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
