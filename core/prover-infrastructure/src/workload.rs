//! Workload balancing for distributed proof generation

use crate::{types::*, error::*};
use std::collections::HashMap;
use consistent_hash::ConsistentHashRing;

/// Workload balancer for optimal prover allocation
pub struct WorkloadBalancer {
    config: WorkloadConfig,
    load_tracker: LoadTracker,
    performance_analyzer: PerformanceAnalyzer,
    consistent_hash_ring: ConsistentHashRing<ProverId>,
}

/// Load tracking for provers
pub struct LoadTracker {
    current_loads: HashMap<ProverId, ProverLoad>,
    load_history: HashMap<ProverId, Vec<LoadSnapshot>>,
    utilization_threshold: f64,
}

/// Current load information for a prover
#[derive(Debug, Clone)]
pub struct ProverLoad {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub active_jobs: usize,
    pub queue_length: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Load snapshot for historical analysis
#[derive(Debug, Clone)]
pub struct LoadSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub throughput: f64,
}

/// Performance analysis for workload optimization
pub struct PerformanceAnalyzer {
    circuit_specialization_map: HashMap<qross_zk_circuits::CircuitId, Vec<ProverId>>,
    performance_history: HashMap<ProverId, Vec<PerformanceRecord>>,
    optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub target_prover: ProverId,
    pub expected_improvement: f64,
    pub confidence: f64,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    IncreaseResources,
    DecreaseResources,
    SpecializeCircuit,
    RebalanceLoad,
    UpgradeHardware,
}

impl WorkloadBalancer {
    /// Create a new workload balancer
    pub fn new(config: WorkloadConfig) -> Self {
        Self {
            load_tracker: LoadTracker::new(config.resource_utilization_threshold),
            performance_analyzer: PerformanceAnalyzer::new(),
            consistent_hash_ring: ConsistentHashRing::new(),
            config,
        }
    }
    
    /// Find optimal prover for job requirements
    pub async fn find_optimal_prover(
        &mut self,
        requirements: &ResourceRequirements,
        available_provers: &[ProverId],
    ) -> Result<Option<ProverId>> {
        if available_provers.is_empty() {
            return Ok(None);
        }
        
        // Update load information
        self.update_load_information(available_provers).await?;
        
        // Apply balancing algorithm
        let optimal_prover = match self.config.balancing_algorithm {
            BalancingAlgorithm::WeightedRoundRobin => {
                self.weighted_round_robin_selection(requirements, available_provers).await?
            }
            BalancingAlgorithm::LeastConnections => {
                self.least_connections_selection(available_provers).await?
            }
            BalancingAlgorithm::ResourceAware => {
                self.resource_aware_selection(requirements, available_provers).await?
            }
            BalancingAlgorithm::ConsistentHashing => {
                self.consistent_hash_selection(requirements, available_provers).await?
            }
        };
        
        // Update load tracking
        if let Some(prover_id) = optimal_prover {
            self.load_tracker.record_job_assignment(prover_id, requirements)?;
        }
        
        Ok(optimal_prover)
    }
    
    /// Update load information for provers
    async fn update_load_information(&mut self, prover_ids: &[ProverId]) -> Result<()> {
        for &prover_id in prover_ids {
            // TODO: Get actual load information from prover nodes
            let load = ProverLoad {
                cpu_utilization: 0.5, // Placeholder
                memory_utilization: 0.4,
                gpu_utilization: 0.3,
                active_jobs: 2,
                queue_length: 1,
                last_updated: chrono::Utc::now(),
            };
            
            self.load_tracker.update_load(prover_id, load);
        }
        
        Ok(())
    }
    
    /// Weighted round-robin selection
    async fn weighted_round_robin_selection(
        &self,
        _requirements: &ResourceRequirements,
        available_provers: &[ProverId],
    ) -> Result<Option<ProverId>> {
        // Calculate weights based on performance and current load
        let mut weighted_provers: Vec<(ProverId, f64)> = Vec::new();
        
        for &prover_id in available_provers {
            let weight = self.calculate_prover_weight(prover_id)?;
            weighted_provers.push((prover_id, weight));
        }
        
        // Sort by weight (highest first)
        weighted_provers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Select based on weighted probability
        if let Some((prover_id, _)) = weighted_provers.first() {
            Ok(Some(*prover_id))
        } else {
            Ok(None)
        }
    }
    
    /// Least connections selection
    async fn least_connections_selection(&self, available_provers: &[ProverId]) -> Result<Option<ProverId>> {
        let mut min_connections = usize::MAX;
        let mut selected_prover = None;
        
        for &prover_id in available_provers {
            if let Some(load) = self.load_tracker.current_loads.get(&prover_id) {
                let total_connections = load.active_jobs + load.queue_length;
                if total_connections < min_connections {
                    min_connections = total_connections;
                    selected_prover = Some(prover_id);
                }
            }
        }
        
        Ok(selected_prover)
    }
    
    /// Resource-aware selection
    async fn resource_aware_selection(
        &self,
        requirements: &ResourceRequirements,
        available_provers: &[ProverId],
    ) -> Result<Option<ProverId>> {
        let mut best_fit_score = f64::MAX;
        let mut selected_prover = None;
        
        for &prover_id in available_provers {
            if let Some(load) = self.load_tracker.current_loads.get(&prover_id) {
                // Check if prover can handle the requirements
                if !self.can_handle_requirements(prover_id, requirements, load)? {
                    continue;
                }
                
                // Calculate fit score (lower is better)
                let fit_score = self.calculate_resource_fit_score(requirements, load)?;
                
                if fit_score < best_fit_score {
                    best_fit_score = fit_score;
                    selected_prover = Some(prover_id);
                }
            }
        }
        
        Ok(selected_prover)
    }
    
    /// Consistent hash selection
    async fn consistent_hash_selection(
        &self,
        requirements: &ResourceRequirements,
        available_provers: &[ProverId],
    ) -> Result<Option<ProverId>> {
        // Create hash key from requirements
        let hash_key = format!("{}_{}_{}",
            requirements.estimated_cpu_time.as_secs(),
            requirements.estimated_memory_gb,
            requirements.requires_gpu
        );
        
        // Find prover using consistent hashing
        for &prover_id in available_provers {
            if let Some(selected) = self.consistent_hash_ring.get(&hash_key) {
                if selected == &prover_id {
                    return Ok(Some(prover_id));
                }
            }
        }
        
        // Fallback to first available prover
        Ok(available_provers.first().copied())
    }
    
    /// Calculate prover weight for selection
    fn calculate_prover_weight(&self, prover_id: ProverId) -> Result<f64> {
        let base_weight = 1.0;
        
        // Adjust based on current load
        if let Some(load) = self.load_tracker.current_loads.get(&prover_id) {
            let load_factor = 1.0 - (load.cpu_utilization + load.memory_utilization + load.gpu_utilization) / 3.0;
            let queue_factor = 1.0 / (1.0 + load.queue_length as f64);
            
            Ok(base_weight * load_factor * queue_factor)
        } else {
            Ok(base_weight)
        }
    }
    
    /// Check if prover can handle requirements
    fn can_handle_requirements(
        &self,
        _prover_id: ProverId,
        requirements: &ResourceRequirements,
        load: &ProverLoad,
    ) -> Result<bool> {
        // Check utilization thresholds
        let cpu_available = 1.0 - load.cpu_utilization;
        let memory_available = 1.0 - load.memory_utilization;
        let gpu_available = 1.0 - load.gpu_utilization;
        
        // Estimate resource requirements as percentages
        let cpu_needed = 0.2; // Placeholder: 20% CPU
        let memory_needed = requirements.estimated_memory_gb as f64 / 32.0; // Assume 32GB total
        let gpu_needed = if requirements.requires_gpu { 0.3 } else { 0.0 }; // 30% GPU if needed
        
        Ok(cpu_available >= cpu_needed && 
           memory_available >= memory_needed && 
           gpu_available >= gpu_needed)
    }
    
    /// Calculate resource fit score
    fn calculate_resource_fit_score(&self, requirements: &ResourceRequirements, load: &ProverLoad) -> Result<f64> {
        // Lower score is better fit
        let cpu_score = load.cpu_utilization;
        let memory_score = load.memory_utilization;
        let gpu_score = if requirements.requires_gpu { load.gpu_utilization } else { 0.0 };
        let queue_score = load.queue_length as f64 / 10.0; // Normalize queue length
        
        Ok(cpu_score + memory_score + gpu_score + queue_score)
    }
    
    /// Add prover to load balancer
    pub fn add_prover(&mut self, prover_id: ProverId) {
        self.consistent_hash_ring.add(&prover_id);
        self.load_tracker.add_prover(prover_id);
        
        tracing::info!("Added prover {} to workload balancer", prover_id);
    }
    
    /// Remove prover from load balancer
    pub fn remove_prover(&mut self, prover_id: ProverId) {
        self.consistent_hash_ring.remove(&prover_id);
        self.load_tracker.remove_prover(prover_id);
        
        tracing::info!("Removed prover {} from workload balancer", prover_id);
    }
    
    /// Get load balancing statistics
    pub fn get_load_balancing_statistics(&self) -> LoadBalancingStatistics {
        let total_provers = self.load_tracker.current_loads.len();
        let overloaded_provers = self.load_tracker.current_loads.values()
            .filter(|load| {
                (load.cpu_utilization + load.memory_utilization + load.gpu_utilization) / 3.0 > self.config.resource_utilization_threshold
            })
            .count();
        
        let average_utilization = if total_provers > 0 {
            self.load_tracker.current_loads.values()
                .map(|load| (load.cpu_utilization + load.memory_utilization + load.gpu_utilization) / 3.0)
                .sum::<f64>() / total_provers as f64
        } else {
            0.0
        };
        
        LoadBalancingStatistics {
            total_provers,
            overloaded_provers,
            average_utilization,
            utilization_threshold: self.config.resource_utilization_threshold,
        }
    }
    
    /// Generate optimization recommendations
    pub fn generate_optimization_recommendations(&mut self) -> Vec<OptimizationRecommendation> {
        self.performance_analyzer.analyze_and_recommend(&self.load_tracker)
    }
}

impl LoadTracker {
    fn new(utilization_threshold: f64) -> Self {
        Self {
            current_loads: HashMap::new(),
            load_history: HashMap::new(),
            utilization_threshold,
        }
    }
    
    fn update_load(&mut self, prover_id: ProverId, load: ProverLoad) {
        // Update current load
        self.current_loads.insert(prover_id, load.clone());
        
        // Add to history
        let snapshot = LoadSnapshot {
            timestamp: load.last_updated,
            cpu_utilization: load.cpu_utilization,
            memory_utilization: load.memory_utilization,
            gpu_utilization: load.gpu_utilization,
            throughput: 0.0, // TODO: Calculate actual throughput
        };
        
        self.load_history.entry(prover_id)
            .or_insert_with(Vec::new)
            .push(snapshot);
        
        // Keep only recent history (last 100 snapshots)
        if let Some(history) = self.load_history.get_mut(&prover_id) {
            if history.len() > 100 {
                history.drain(0..history.len() - 100);
            }
        }
    }
    
    fn record_job_assignment(&mut self, prover_id: ProverId, _requirements: &ResourceRequirements) -> Result<()> {
        if let Some(load) = self.current_loads.get_mut(&prover_id) {
            load.queue_length += 1;
        }
        
        Ok(())
    }
    
    fn add_prover(&mut self, prover_id: ProverId) {
        let initial_load = ProverLoad {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            gpu_utilization: 0.0,
            active_jobs: 0,
            queue_length: 0,
            last_updated: chrono::Utc::now(),
        };
        
        self.current_loads.insert(prover_id, initial_load);
        self.load_history.insert(prover_id, Vec::new());
    }
    
    fn remove_prover(&mut self, prover_id: ProverId) {
        self.current_loads.remove(&prover_id);
        self.load_history.remove(&prover_id);
    }
}

impl PerformanceAnalyzer {
    fn new() -> Self {
        Self {
            circuit_specialization_map: HashMap::new(),
            performance_history: HashMap::new(),
            optimization_recommendations: Vec::new(),
        }
    }
    
    fn analyze_and_recommend(&mut self, load_tracker: &LoadTracker) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Analyze each prover's performance
        for (prover_id, load) in &load_tracker.current_loads {
            // Check for overutilization
            let avg_utilization = (load.cpu_utilization + load.memory_utilization + load.gpu_utilization) / 3.0;
            
            if avg_utilization > 0.9 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::IncreaseResources,
                    target_prover: *prover_id,
                    expected_improvement: 0.3,
                    confidence: 0.8,
                    generated_at: chrono::Utc::now(),
                });
            } else if avg_utilization < 0.2 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::DecreaseResources,
                    target_prover: *prover_id,
                    expected_improvement: 0.2,
                    confidence: 0.7,
                    generated_at: chrono::Utc::now(),
                });
            }
        }
        
        self.optimization_recommendations = recommendations.clone();
        recommendations
    }
}

/// Load balancing statistics
#[derive(Debug, Clone)]
pub struct LoadBalancingStatistics {
    pub total_provers: usize,
    pub overloaded_provers: usize,
    pub average_utilization: f64,
    pub utilization_threshold: f64,
}
