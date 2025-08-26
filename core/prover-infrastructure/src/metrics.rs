//! Metrics collection for prover infrastructure

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram,
    register_int_counter, register_int_gauge
};

/// Prover infrastructure metrics collector
pub struct ProverMetrics {
    // Job metrics
    jobs_submitted: IntCounter,
    jobs_assigned: IntCounter,
    jobs_completed: IntCounter,
    jobs_failed: IntCounter,
    jobs_cancelled: IntCounter,
    
    // Timing metrics
    job_submission_time: Histogram,
    proof_generation_time: Histogram,
    job_queue_time: Histogram,
    scheduling_time: Histogram,
    
    // Resource metrics
    active_provers: IntGauge,
    total_cpu_cores: IntGauge,
    total_memory_gb: IntGauge,
    total_gpu_memory_gb: IntGauge,
    
    // Utilization metrics
    cpu_utilization: Gauge,
    memory_utilization: Gauge,
    gpu_utilization: Gauge,
    
    // Queue metrics
    queue_length: IntGauge,
    queue_wait_time: Histogram,
    
    // Scaling metrics
    scaling_events: IntCounter,
    scale_up_events: IntCounter,
    scale_down_events: IntCounter,
    
    // Error metrics
    prover_errors: IntCounter,
    gpu_errors: IntCounter,
    kubernetes_errors: IntCounter,
    scheduling_errors: IntCounter,
    
    // Performance metrics
    throughput: Gauge,
    success_rate: Gauge,
    average_job_duration: Gauge,
    
    // Kubernetes metrics
    pod_restarts: IntCounter,
    pod_failures: IntCounter,
    deployment_updates: IntCounter,
}

impl ProverMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            jobs_submitted: register_int_counter!(
                "prover_jobs_submitted_total",
                "Total number of proof jobs submitted"
            ).unwrap(),
            
            jobs_assigned: register_int_counter!(
                "prover_jobs_assigned_total",
                "Total number of proof jobs assigned to provers"
            ).unwrap(),
            
            jobs_completed: register_int_counter!(
                "prover_jobs_completed_total",
                "Total number of proof jobs completed successfully"
            ).unwrap(),
            
            jobs_failed: register_int_counter!(
                "prover_jobs_failed_total",
                "Total number of proof jobs that failed"
            ).unwrap(),
            
            jobs_cancelled: register_int_counter!(
                "prover_jobs_cancelled_total",
                "Total number of proof jobs cancelled"
            ).unwrap(),
            
            job_submission_time: register_histogram!(
                "prover_job_submission_duration_seconds",
                "Time taken to submit proof jobs",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            ).unwrap(),
            
            proof_generation_time: register_histogram!(
                "prover_proof_generation_duration_seconds",
                "Time taken to generate proofs",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
            ).unwrap(),
            
            job_queue_time: register_histogram!(
                "prover_job_queue_duration_seconds",
                "Time jobs spend in queue before processing",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
            ).unwrap(),
            
            scheduling_time: register_histogram!(
                "prover_scheduling_duration_seconds",
                "Time taken for job scheduling decisions",
                vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            ).unwrap(),
            
            active_provers: register_int_gauge!(
                "prover_active_provers",
                "Number of currently active prover nodes"
            ).unwrap(),
            
            total_cpu_cores: register_int_gauge!(
                "prover_total_cpu_cores",
                "Total CPU cores available across all provers"
            ).unwrap(),
            
            total_memory_gb: register_int_gauge!(
                "prover_total_memory_gb",
                "Total memory in GB available across all provers"
            ).unwrap(),
            
            total_gpu_memory_gb: register_int_gauge!(
                "prover_total_gpu_memory_gb",
                "Total GPU memory in GB available across all provers"
            ).unwrap(),
            
            cpu_utilization: register_gauge!(
                "prover_cpu_utilization_percent",
                "Current CPU utilization percentage across all provers"
            ).unwrap(),
            
            memory_utilization: register_gauge!(
                "prover_memory_utilization_percent",
                "Current memory utilization percentage across all provers"
            ).unwrap(),
            
            gpu_utilization: register_gauge!(
                "prover_gpu_utilization_percent",
                "Current GPU utilization percentage across all provers"
            ).unwrap(),
            
            queue_length: register_int_gauge!(
                "prover_queue_length",
                "Current number of jobs in the proof generation queue"
            ).unwrap(),
            
            queue_wait_time: register_histogram!(
                "prover_queue_wait_duration_seconds",
                "Time jobs wait in queue before being assigned",
                vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
            ).unwrap(),
            
            scaling_events: register_int_counter!(
                "prover_scaling_events_total",
                "Total number of scaling events"
            ).unwrap(),
            
            scale_up_events: register_int_counter!(
                "prover_scale_up_events_total",
                "Total number of scale up events"
            ).unwrap(),
            
            scale_down_events: register_int_counter!(
                "prover_scale_down_events_total",
                "Total number of scale down events"
            ).unwrap(),
            
            prover_errors: register_int_counter!(
                "prover_errors_total",
                "Total number of prover-related errors"
            ).unwrap(),
            
            gpu_errors: register_int_counter!(
                "prover_gpu_errors_total",
                "Total number of GPU-related errors"
            ).unwrap(),
            
            kubernetes_errors: register_int_counter!(
                "prover_kubernetes_errors_total",
                "Total number of Kubernetes-related errors"
            ).unwrap(),
            
            scheduling_errors: register_int_counter!(
                "prover_scheduling_errors_total",
                "Total number of scheduling errors"
            ).unwrap(),
            
            throughput: register_gauge!(
                "prover_throughput_jobs_per_second",
                "Current proof generation throughput in jobs per second"
            ).unwrap(),
            
            success_rate: register_gauge!(
                "prover_success_rate_percent",
                "Current success rate percentage for proof generation"
            ).unwrap(),
            
            average_job_duration: register_gauge!(
                "prover_average_job_duration_seconds",
                "Current average job duration in seconds"
            ).unwrap(),
            
            pod_restarts: register_int_counter!(
                "prover_pod_restarts_total",
                "Total number of prover pod restarts"
            ).unwrap(),
            
            pod_failures: register_int_counter!(
                "prover_pod_failures_total",
                "Total number of prover pod failures"
            ).unwrap(),
            
            deployment_updates: register_int_counter!(
                "prover_deployment_updates_total",
                "Total number of deployment updates"
            ).unwrap(),
        }
    }
    
    // Job metrics
    pub fn increment_jobs_submitted(&self) {
        self.jobs_submitted.inc();
    }
    
    pub fn increment_jobs_assigned(&self) {
        self.jobs_assigned.inc();
    }
    
    pub fn increment_jobs_completed(&self) {
        self.jobs_completed.inc();
    }
    
    pub fn increment_jobs_failed(&self) {
        self.jobs_failed.inc();
    }
    
    pub fn increment_jobs_cancelled(&self) {
        self.jobs_cancelled.inc();
    }
    
    // Timing metrics
    pub fn record_job_submission_time(&self, duration: f64) {
        self.job_submission_time.observe(duration);
    }
    
    pub fn record_proof_generation_time(&self, duration: f64) {
        self.proof_generation_time.observe(duration);
    }
    
    pub fn record_job_queue_time(&self, duration: f64) {
        self.job_queue_time.observe(duration);
    }
    
    pub fn record_scheduling_time(&self, duration: f64) {
        self.scheduling_time.observe(duration);
    }
    
    // Resource metrics
    pub fn set_active_provers(&self, count: i64) {
        self.active_provers.set(count);
    }
    
    pub fn set_total_cpu_cores(&self, cores: i64) {
        self.total_cpu_cores.set(cores);
    }
    
    pub fn set_total_memory_gb(&self, memory: i64) {
        self.total_memory_gb.set(memory);
    }
    
    pub fn set_total_gpu_memory_gb(&self, memory: i64) {
        self.total_gpu_memory_gb.set(memory);
    }
    
    // Utilization metrics
    pub fn set_cpu_utilization(&self, utilization: f64) {
        self.cpu_utilization.set(utilization * 100.0);
    }
    
    pub fn set_memory_utilization(&self, utilization: f64) {
        self.memory_utilization.set(utilization * 100.0);
    }
    
    pub fn set_gpu_utilization(&self, utilization: f64) {
        self.gpu_utilization.set(utilization * 100.0);
    }
    
    // Queue metrics
    pub fn set_queue_length(&self, length: i64) {
        self.queue_length.set(length);
    }
    
    pub fn record_queue_wait_time(&self, duration: f64) {
        self.queue_wait_time.observe(duration);
    }
    
    // Scaling metrics
    pub fn increment_scaling_events(&self) {
        self.scaling_events.inc();
    }
    
    pub fn increment_scale_up_events(&self) {
        self.scale_up_events.inc();
    }
    
    pub fn increment_scale_down_events(&self) {
        self.scale_down_events.inc();
    }
    
    // Error metrics
    pub fn increment_prover_errors(&self) {
        self.prover_errors.inc();
    }
    
    pub fn increment_gpu_errors(&self) {
        self.gpu_errors.inc();
    }
    
    pub fn increment_kubernetes_errors(&self) {
        self.kubernetes_errors.inc();
    }
    
    pub fn increment_scheduling_errors(&self) {
        self.scheduling_errors.inc();
    }
    
    // Performance metrics
    pub fn set_throughput(&self, throughput: f64) {
        self.throughput.set(throughput);
    }
    
    pub fn set_success_rate(&self, rate: f64) {
        self.success_rate.set(rate * 100.0);
    }
    
    pub fn set_average_job_duration(&self, duration: f64) {
        self.average_job_duration.set(duration);
    }
    
    // Kubernetes metrics
    pub fn increment_pod_restarts(&self) {
        self.pod_restarts.inc();
    }
    
    pub fn increment_pod_failures(&self) {
        self.pod_failures.inc();
    }
    
    pub fn increment_deployment_updates(&self) {
        self.deployment_updates.inc();
    }
    
    // Calculated metrics
    pub fn get_total_jobs(&self) -> u64 {
        self.jobs_submitted.get() as u64
    }
    
    pub fn get_success_rate(&self) -> f64 {
        let completed = self.jobs_completed.get() as f64;
        let failed = self.jobs_failed.get() as f64;
        let total = completed + failed;
        
        if total > 0.0 {
            completed / total
        } else {
            0.0
        }
    }
    
    pub fn get_average_proof_generation_time(&self) -> f64 {
        // TODO: Calculate from histogram
        0.0
    }
    
    pub fn get_current_throughput(&self) -> f64 {
        self.throughput.get()
    }
    
    /// Update all calculated metrics
    pub fn update_calculated_metrics(&self) {
        let success_rate = self.get_success_rate();
        self.set_success_rate(success_rate);
        
        // TODO: Calculate and update other derived metrics
    }
}

impl Default for ProverMetrics {
    fn default() -> Self {
        Self::new()
    }
}
