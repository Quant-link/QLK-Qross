//! Resource management and estimation

use crate::{types::*, error::*};
use qross_zk_circuits::{CircuitId, CircuitInputs};
use std::collections::HashMap;

/// Resource manager for capacity planning and allocation
pub struct ResourceManager {
    config: ResourceConfig,
    resource_estimator: ResourceEstimator,
    capacity_planner: CapacityPlanner,
    monitoring_agent: MonitoringAgent,
}

/// Resource estimation engine
pub struct ResourceEstimator {
    circuit_profiles: HashMap<CircuitId, CircuitResourceProfile>,
    estimation_models: HashMap<String, EstimationModel>,
    historical_data: Vec<ResourceUsageRecord>,
}

/// Circuit resource profile
#[derive(Debug, Clone)]
pub struct CircuitResourceProfile {
    pub circuit_id: CircuitId,
    pub base_cpu_time: std::time::Duration,
    pub base_memory_gb: u32,
    pub base_gpu_memory_gb: u32,
    pub scaling_factors: ScalingFactors,
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Scaling factors for resource estimation
#[derive(Debug, Clone)]
pub struct ScalingFactors {
    pub input_size_factor: f64,
    pub complexity_factor: f64,
    pub parallelization_factor: f64,
    pub gpu_acceleration_factor: f64,
}

/// Optimization hints for resource allocation
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    PreferGpu,
    PreferCpu,
    MemoryIntensive,
    ComputeIntensive,
    BatchFriendly,
}

/// Resource estimation model
#[derive(Debug, Clone)]
pub struct EstimationModel {
    pub model_type: ModelType,
    pub parameters: Vec<f64>,
    pub accuracy_score: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Types of estimation models
#[derive(Debug, Clone)]
pub enum ModelType {
    Linear,
    Polynomial,
    Exponential,
    MachineLearning,
}

/// Historical resource usage record
#[derive(Debug, Clone)]
pub struct ResourceUsageRecord {
    pub circuit_id: CircuitId,
    pub input_size: usize,
    pub actual_cpu_time: std::time::Duration,
    pub actual_memory_gb: u32,
    pub actual_gpu_memory_gb: u32,
    pub completion_time: std::time::Duration,
    pub success: bool,
    pub recorded_at: chrono::DateTime<chrono::Utc>,
}

/// Capacity planning engine
pub struct CapacityPlanner {
    current_capacity: TotalCapacity,
    allocated_capacity: TotalCapacity,
    capacity_forecasts: Vec<CapacityForecast>,
    scaling_policies: Vec<ScalingPolicy>,
}

/// Total system capacity
#[derive(Debug, Clone)]
pub struct TotalCapacity {
    pub total_cpu_cores: u32,
    pub total_memory_gb: u32,
    pub total_gpu_memory_gb: u32,
    pub total_storage_gb: u32,
    pub total_network_bandwidth_mbps: u32,
}

/// Capacity forecast
#[derive(Debug, Clone)]
pub struct CapacityForecast {
    pub forecast_time: chrono::DateTime<chrono::Utc>,
    pub predicted_demand: ResourceDemand,
    pub confidence_interval: (f64, f64),
    pub recommended_capacity: TotalCapacity,
}

/// Resource demand prediction
#[derive(Debug, Clone)]
pub struct ResourceDemand {
    pub cpu_demand: f64,
    pub memory_demand: f64,
    pub gpu_demand: f64,
    pub peak_demand_time: chrono::DateTime<chrono::Utc>,
}

/// Scaling policy
#[derive(Debug, Clone)]
pub struct ScalingPolicy {
    pub policy_name: String,
    pub trigger_conditions: Vec<ScalingTrigger>,
    pub scaling_action: ScalingAction,
    pub cooldown_period: std::time::Duration,
}

/// Scaling trigger conditions
#[derive(Debug, Clone)]
pub enum ScalingTrigger {
    CpuUtilization(f64),
    MemoryUtilization(f64),
    GpuUtilization(f64),
    QueueLength(usize),
    ResponseTime(std::time::Duration),
}

/// Scaling actions
#[derive(Debug, Clone)]
pub enum ScalingAction {
    ScaleUp(u32),
    ScaleDown(u32),
    AddGpuNodes(u32),
    RemoveGpuNodes(u32),
}

/// Resource monitoring agent
pub struct MonitoringAgent {
    monitoring_interval: std::time::Duration,
    metrics_collector: MetricsCollector,
    alert_manager: AlertManager,
}

/// Metrics collection system
pub struct MetricsCollector {
    system_metrics: HashMap<String, f64>,
    application_metrics: HashMap<String, f64>,
    custom_metrics: HashMap<String, f64>,
}

/// Alert management system
pub struct AlertManager {
    alert_rules: Vec<AlertRule>,
    active_alerts: Vec<Alert>,
    notification_channels: Vec<NotificationChannel>,
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub rule_name: String,
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub duration: std::time::Duration,
    pub severity: AlertSeverity,
}

/// Comparison operators for alerts
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_id: uuid::Uuid,
    pub rule_name: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub triggered_at: chrono::DateTime<chrono::Utc>,
    pub resolved_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Notification channel
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Email(String),
    Slack(String),
    Webhook(String),
    PagerDuty(String),
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            resource_estimator: ResourceEstimator::new(),
            capacity_planner: CapacityPlanner::new(),
            monitoring_agent: MonitoringAgent::new(
                std::time::Duration::from_secs(config.resource_monitoring_interval)
            ),
            config,
        }
    }
    
    /// Estimate resource requirements for proof generation
    pub async fn estimate_proof_requirements(
        &self,
        circuit_id: CircuitId,
        inputs: &CircuitInputs,
    ) -> Result<ResourceRequirements> {
        self.resource_estimator.estimate_requirements(circuit_id, inputs).await
    }
    
    /// Check resource availability
    pub async fn check_resource_availability(&self, requirements: &ResourceRequirements) -> Result<bool> {
        let available_capacity = self.capacity_planner.get_available_capacity().await?;
        
        let cpu_available = available_capacity.total_cpu_cores as f64 * (1.0 - self.config.cpu_oversubscription_ratio);
        let memory_available = available_capacity.total_memory_gb as f64 * (1.0 - self.config.memory_safety_margin);
        let gpu_memory_available = available_capacity.total_gpu_memory_gb as f64 * (1.0 - self.config.gpu_memory_reservation);
        
        let cpu_needed = requirements.estimated_cpu_time.as_secs_f64() / 3600.0; // Convert to hours
        let memory_needed = requirements.estimated_memory_gb as f64;
        let gpu_memory_needed = requirements.estimated_gpu_memory_gb as f64;
        
        Ok(cpu_needed <= cpu_available && 
           memory_needed <= memory_available && 
           gpu_memory_needed <= gpu_memory_available)
    }
    
    /// Record actual resource usage
    pub async fn record_resource_usage(&mut self, record: ResourceUsageRecord) -> Result<()> {
        self.resource_estimator.add_usage_record(record).await
    }
    
    /// Get resource utilization statistics
    pub async fn get_resource_utilization(&self) -> Result<ResourceUtilizationStats> {
        let current_capacity = self.capacity_planner.get_current_capacity().await?;
        let allocated_capacity = self.capacity_planner.get_allocated_capacity().await?;
        
        let cpu_utilization = allocated_capacity.total_cpu_cores as f64 / current_capacity.total_cpu_cores as f64;
        let memory_utilization = allocated_capacity.total_memory_gb as f64 / current_capacity.total_memory_gb as f64;
        let gpu_utilization = allocated_capacity.total_gpu_memory_gb as f64 / current_capacity.total_gpu_memory_gb as f64;
        
        Ok(ResourceUtilizationStats {
            cpu_utilization,
            memory_utilization,
            gpu_utilization,
            total_capacity: current_capacity,
            allocated_capacity,
        })
    }
    
    /// Generate capacity forecast
    pub async fn generate_capacity_forecast(&mut self, forecast_horizon: std::time::Duration) -> Result<CapacityForecast> {
        self.capacity_planner.generate_forecast(forecast_horizon).await
    }
}

impl ResourceEstimator {
    fn new() -> Self {
        Self {
            circuit_profiles: HashMap::new(),
            estimation_models: HashMap::new(),
            historical_data: Vec::new(),
        }
    }
    
    async fn estimate_requirements(&self, circuit_id: CircuitId, inputs: &CircuitInputs) -> Result<ResourceRequirements> {
        // Get circuit profile or create default
        let profile = self.circuit_profiles.get(&circuit_id)
            .cloned()
            .unwrap_or_else(|| self.create_default_profile(circuit_id));
        
        // Calculate input size factor
        let input_size = inputs.public_inputs.len() + inputs.private_inputs.len();
        let size_factor = (input_size as f64).sqrt() * profile.scaling_factors.input_size_factor;
        
        // Estimate resources
        let estimated_cpu_time = profile.base_cpu_time.mul_f64(size_factor);
        let estimated_memory_gb = (profile.base_memory_gb as f64 * size_factor) as u32;
        let estimated_gpu_memory_gb = (profile.base_gpu_memory_gb as f64 * size_factor) as u32;
        
        Ok(ResourceRequirements {
            estimated_cpu_time,
            estimated_memory_gb,
            estimated_gpu_memory_gb,
            requires_gpu: estimated_gpu_memory_gb > 0,
            parallel_threads: 4, // Default
            estimated_duration: estimated_cpu_time,
        })
    }
    
    fn create_default_profile(&self, circuit_id: CircuitId) -> CircuitResourceProfile {
        CircuitResourceProfile {
            circuit_id,
            base_cpu_time: std::time::Duration::from_secs(300), // 5 minutes
            base_memory_gb: 4,
            base_gpu_memory_gb: 2,
            scaling_factors: ScalingFactors {
                input_size_factor: 1.2,
                complexity_factor: 1.0,
                parallelization_factor: 0.8,
                gpu_acceleration_factor: 2.0,
            },
            optimization_hints: vec![OptimizationHint::PreferGpu],
        }
    }
    
    async fn add_usage_record(&mut self, record: ResourceUsageRecord) -> Result<()> {
        self.historical_data.push(record.clone());
        
        // Update circuit profile based on actual usage
        self.update_circuit_profile(&record).await?;
        
        // Keep only recent history (last 1000 records)
        if self.historical_data.len() > 1000 {
            self.historical_data.drain(0..self.historical_data.len() - 1000);
        }
        
        Ok(())
    }
    
    async fn update_circuit_profile(&mut self, record: &ResourceUsageRecord) -> Result<()> {
        // TODO: Implement machine learning-based profile updates
        // This would analyze historical data and adjust circuit profiles
        Ok(())
    }
}

impl CapacityPlanner {
    fn new() -> Self {
        Self {
            current_capacity: TotalCapacity {
                total_cpu_cores: 0,
                total_memory_gb: 0,
                total_gpu_memory_gb: 0,
                total_storage_gb: 0,
                total_network_bandwidth_mbps: 0,
            },
            allocated_capacity: TotalCapacity {
                total_cpu_cores: 0,
                total_memory_gb: 0,
                total_gpu_memory_gb: 0,
                total_storage_gb: 0,
                total_network_bandwidth_mbps: 0,
            },
            capacity_forecasts: Vec::new(),
            scaling_policies: Vec::new(),
        }
    }
    
    async fn get_current_capacity(&self) -> Result<TotalCapacity> {
        Ok(self.current_capacity.clone())
    }
    
    async fn get_allocated_capacity(&self) -> Result<TotalCapacity> {
        Ok(self.allocated_capacity.clone())
    }
    
    async fn get_available_capacity(&self) -> Result<TotalCapacity> {
        Ok(TotalCapacity {
            total_cpu_cores: self.current_capacity.total_cpu_cores - self.allocated_capacity.total_cpu_cores,
            total_memory_gb: self.current_capacity.total_memory_gb - self.allocated_capacity.total_memory_gb,
            total_gpu_memory_gb: self.current_capacity.total_gpu_memory_gb - self.allocated_capacity.total_gpu_memory_gb,
            total_storage_gb: self.current_capacity.total_storage_gb - self.allocated_capacity.total_storage_gb,
            total_network_bandwidth_mbps: self.current_capacity.total_network_bandwidth_mbps - self.allocated_capacity.total_network_bandwidth_mbps,
        })
    }
    
    async fn generate_forecast(&mut self, _forecast_horizon: std::time::Duration) -> Result<CapacityForecast> {
        // TODO: Implement actual forecasting algorithm
        let forecast = CapacityForecast {
            forecast_time: chrono::Utc::now() + chrono::Duration::hours(24),
            predicted_demand: ResourceDemand {
                cpu_demand: 0.8,
                memory_demand: 0.7,
                gpu_demand: 0.9,
                peak_demand_time: chrono::Utc::now() + chrono::Duration::hours(12),
            },
            confidence_interval: (0.7, 0.9),
            recommended_capacity: self.current_capacity.clone(),
        };
        
        self.capacity_forecasts.push(forecast.clone());
        Ok(forecast)
    }
}

impl MonitoringAgent {
    fn new(monitoring_interval: std::time::Duration) -> Self {
        Self {
            monitoring_interval,
            metrics_collector: MetricsCollector::new(),
            alert_manager: AlertManager::new(),
        }
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            system_metrics: HashMap::new(),
            application_metrics: HashMap::new(),
            custom_metrics: HashMap::new(),
        }
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            alert_rules: Vec::new(),
            active_alerts: Vec::new(),
            notification_channels: Vec::new(),
        }
    }
}

/// Resource utilization statistics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationStats {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub total_capacity: TotalCapacity,
    pub allocated_capacity: TotalCapacity,
}
