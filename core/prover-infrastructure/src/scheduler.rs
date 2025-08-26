//! Proof job scheduling and prioritization

use crate::{types::*, error::*};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

/// Proof scheduler for job prioritization and deadline management
pub struct ProofScheduler {
    config: SchedulerConfig,
    priority_queue: BinaryHeap<ScheduledJob>,
    deadline_queue: BinaryHeap<DeadlineJob>,
    job_metadata: HashMap<ProofJobId, JobMetadata>,
    scheduling_history: Vec<SchedulingDecision>,
}

/// Scheduled job with priority
#[derive(Debug, Clone)]
pub struct ScheduledJob {
    pub job_id: ProofJobId,
    pub priority: ProofPriority,
    pub estimated_duration: std::time::Duration,
    pub resource_requirements: ResourceRequirements,
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

/// Job with deadline tracking
#[derive(Debug, Clone)]
pub struct DeadlineJob {
    pub job_id: ProofJobId,
    pub deadline: chrono::DateTime<chrono::Utc>,
    pub slack_time: std::time::Duration,
}

/// Job metadata for scheduling decisions
#[derive(Debug, Clone)]
pub struct JobMetadata {
    pub circuit_complexity: f64,
    pub estimated_memory_usage: u64,
    pub requires_gpu: bool,
    pub batch_compatible: bool,
    pub retry_count: u32,
}

/// Scheduling decision record
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    pub decision_id: uuid::Uuid,
    pub job_id: ProofJobId,
    pub algorithm_used: SchedulingAlgorithm,
    pub decision_time: chrono::DateTime<chrono::Utc>,
    pub factors_considered: Vec<SchedulingFactor>,
    pub expected_completion: chrono::DateTime<chrono::Utc>,
}

/// Factors considered in scheduling
#[derive(Debug, Clone)]
pub enum SchedulingFactor {
    Priority(ProofPriority),
    Deadline(chrono::DateTime<chrono::Utc>),
    ResourceAvailability(f64),
    BatchOptimization(bool),
    CircuitSpecialization(qross_zk_circuits::CircuitId),
}

impl ProofScheduler {
    /// Create a new proof scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            priority_queue: BinaryHeap::new(),
            deadline_queue: BinaryHeap::new(),
            job_metadata: HashMap::new(),
            scheduling_history: Vec::new(),
        }
    }
    
    /// Schedule a proof job
    pub async fn schedule_proof_job(&mut self, job_id: ProofJobId) -> Result<()> {
        // TODO: Get job details from job queue
        // For now, create placeholder job
        let scheduled_job = ScheduledJob {
            job_id,
            priority: ProofPriority::Normal,
            estimated_duration: std::time::Duration::from_secs(300),
            resource_requirements: ResourceRequirements {
                estimated_cpu_time: std::time::Duration::from_secs(300),
                estimated_memory_gb: 4,
                estimated_gpu_memory_gb: 2,
                requires_gpu: true,
                parallel_threads: 4,
                estimated_duration: std::time::Duration::from_secs(300),
            },
            submitted_at: chrono::Utc::now(),
            deadline: None,
        };
        
        // Add to priority queue
        self.priority_queue.push(scheduled_job.clone());
        
        // Add to deadline queue if deadline exists
        if let Some(deadline) = scheduled_job.deadline {
            let slack_time = deadline.signed_duration_since(chrono::Utc::now())
                .to_std()
                .unwrap_or(std::time::Duration::ZERO);
            
            let deadline_job = DeadlineJob {
                job_id,
                deadline,
                slack_time,
            };
            
            self.deadline_queue.push(deadline_job);
        }
        
        // Record metadata
        let metadata = JobMetadata {
            circuit_complexity: 1.0, // Placeholder
            estimated_memory_usage: scheduled_job.resource_requirements.estimated_memory_gb as u64 * 1024 * 1024 * 1024,
            requires_gpu: scheduled_job.resource_requirements.requires_gpu,
            batch_compatible: true,
            retry_count: 0,
        };
        
        self.job_metadata.insert(job_id, metadata);
        
        // Record scheduling decision
        let decision = SchedulingDecision {
            decision_id: uuid::Uuid::new_v4(),
            job_id,
            algorithm_used: self.config.scheduling_algorithm.clone(),
            decision_time: chrono::Utc::now(),
            factors_considered: vec![
                SchedulingFactor::Priority(scheduled_job.priority),
                SchedulingFactor::ResourceAvailability(0.8),
            ],
            expected_completion: chrono::Utc::now() + chrono::Duration::from_std(scheduled_job.estimated_duration).unwrap(),
        };
        
        self.scheduling_history.push(decision);
        
        tracing::info!("Scheduled proof job {} with priority {:?}", job_id, scheduled_job.priority);
        
        Ok(())
    }
    
    /// Get next job to process based on scheduling algorithm
    pub async fn get_next_job(&mut self) -> Result<Option<ProofJobId>> {
        match self.config.scheduling_algorithm {
            SchedulingAlgorithm::FIFO => self.get_fifo_job().await,
            SchedulingAlgorithm::PriorityBased => self.get_priority_job().await,
            SchedulingAlgorithm::DeadlineAware => self.get_deadline_aware_job().await,
            SchedulingAlgorithm::ResourceOptimized => self.get_resource_optimized_job().await,
        }
    }
    
    /// FIFO scheduling
    async fn get_fifo_job(&mut self) -> Result<Option<ProofJobId>> {
        // Get oldest job by submission time
        let mut oldest_job = None;
        let mut oldest_time = chrono::Utc::now();
        
        for job in &self.priority_queue {
            if job.submitted_at < oldest_time {
                oldest_time = job.submitted_at;
                oldest_job = Some(job.job_id);
            }
        }
        
        if let Some(job_id) = oldest_job {
            self.remove_job_from_queues(job_id);
        }
        
        Ok(oldest_job)
    }
    
    /// Priority-based scheduling
    async fn get_priority_job(&mut self) -> Result<Option<ProofJobId>> {
        if let Some(job) = self.priority_queue.pop() {
            self.remove_job_from_deadline_queue(job.job_id);
            Ok(Some(job.job_id))
        } else {
            Ok(None)
        }
    }
    
    /// Deadline-aware scheduling
    async fn get_deadline_aware_job(&mut self) -> Result<Option<ProofJobId>> {
        // Check for urgent deadlines first
        while let Some(deadline_job) = self.deadline_queue.peek() {
            let now = chrono::Utc::now();
            let time_to_deadline = deadline_job.deadline.signed_duration_since(now);
            
            // If deadline is within 10 minutes, prioritize
            if time_to_deadline <= chrono::Duration::minutes(10) {
                let job_id = deadline_job.job_id;
                self.deadline_queue.pop();
                self.remove_job_from_priority_queue(job_id);
                return Ok(Some(job_id));
            } else {
                break;
            }
        }
        
        // Fall back to priority-based scheduling
        self.get_priority_job().await
    }
    
    /// Resource-optimized scheduling
    async fn get_resource_optimized_job(&mut self) -> Result<Option<ProofJobId>> {
        // TODO: Implement resource optimization logic
        // This would consider current resource availability and job requirements
        self.get_priority_job().await
    }
    
    /// Remove job from all queues
    fn remove_job_from_queues(&mut self, job_id: ProofJobId) {
        self.remove_job_from_priority_queue(job_id);
        self.remove_job_from_deadline_queue(job_id);
        self.job_metadata.remove(&job_id);
    }
    
    /// Remove job from priority queue
    fn remove_job_from_priority_queue(&mut self, job_id: ProofJobId) {
        let mut temp_queue = BinaryHeap::new();
        
        while let Some(job) = self.priority_queue.pop() {
            if job.job_id != job_id {
                temp_queue.push(job);
            }
        }
        
        self.priority_queue = temp_queue;
    }
    
    /// Remove job from deadline queue
    fn remove_job_from_deadline_queue(&mut self, job_id: ProofJobId) {
        let mut temp_queue = BinaryHeap::new();
        
        while let Some(job) = self.deadline_queue.pop() {
            if job.job_id != job_id {
                temp_queue.push(job);
            }
        }
        
        self.deadline_queue = temp_queue;
    }
    
    /// Get scheduling statistics
    pub fn get_scheduling_statistics(&self) -> SchedulingStatistics {
        let queued_jobs = self.priority_queue.len();
        let jobs_with_deadlines = self.deadline_queue.len();
        let total_decisions = self.scheduling_history.len();
        
        let average_decision_time = if !self.scheduling_history.is_empty() {
            // Calculate average time between submission and scheduling
            0.0 // Placeholder
        } else {
            0.0
        };
        
        SchedulingStatistics {
            queued_jobs,
            jobs_with_deadlines,
            total_decisions,
            average_decision_time,
            algorithm_used: self.config.scheduling_algorithm.clone(),
        }
    }
    
    /// Optimize scheduling based on performance data
    pub fn optimize_scheduling(&mut self, performance_data: &[SchedulingPerformanceData]) {
        // Analyze performance data and adjust scheduling parameters
        if performance_data.is_empty() {
            return;
        }
        
        let average_completion_time: f64 = performance_data.iter()
            .map(|data| data.actual_completion_time.as_secs_f64())
            .sum::<f64>() / performance_data.len() as f64;
        
        let deadline_miss_rate: f64 = performance_data.iter()
            .filter(|data| data.missed_deadline)
            .count() as f64 / performance_data.len() as f64;
        
        // Adjust algorithm if deadline miss rate is high
        if deadline_miss_rate > 0.1 && self.config.deadline_awareness {
            tracing::info!("High deadline miss rate ({:.2}%), optimizing for deadline awareness", deadline_miss_rate);
            // Could switch to more aggressive deadline-aware scheduling
        }
        
        tracing::debug!(
            "Scheduling optimization: avg_completion={:.2}s, deadline_miss_rate={:.2}%",
            average_completion_time,
            deadline_miss_rate * 100.0
        );
    }
}

/// Implement ordering for scheduled jobs (priority queue)
impl Ord for ScheduledJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier submission time
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.submitted_at.cmp(&self.submitted_at),
            other => other,
        }
    }
}

impl PartialOrd for ScheduledJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScheduledJob {
    fn eq(&self, other: &Self) -> bool {
        self.job_id == other.job_id
    }
}

impl Eq for ScheduledJob {}

/// Implement ordering for deadline jobs
impl Ord for DeadlineJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Earlier deadline first
        other.deadline.cmp(&self.deadline)
    }
}

impl PartialOrd for DeadlineJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DeadlineJob {
    fn eq(&self, other: &Self) -> bool {
        self.job_id == other.job_id
    }
}

impl Eq for DeadlineJob {}

/// Scheduling statistics
#[derive(Debug, Clone)]
pub struct SchedulingStatistics {
    pub queued_jobs: usize,
    pub jobs_with_deadlines: usize,
    pub total_decisions: usize,
    pub average_decision_time: f64,
    pub algorithm_used: SchedulingAlgorithm,
}

/// Performance data for scheduling optimization
#[derive(Debug, Clone)]
pub struct SchedulingPerformanceData {
    pub job_id: ProofJobId,
    pub estimated_completion_time: std::time::Duration,
    pub actual_completion_time: std::time::Duration,
    pub missed_deadline: bool,
    pub resource_efficiency: f64,
}
