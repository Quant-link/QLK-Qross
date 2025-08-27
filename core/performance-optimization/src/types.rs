//! Types for performance optimization engine

use rust_decimal::Decimal;
use std::collections::HashMap;

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub fee_config: FeeOptimizationConfig,
    pub batch_config: BatchProcessingConfig,
    pub monitoring_config: PerformanceMonitoringConfig,
    pub algorithm_config: AdaptiveAlgorithmConfig,
    pub cost_config: CostModelingConfig,
    pub priority_config: PriorityManagementConfig,
    pub gas_config: GasPredictionConfig,
    pub metrics_config: OptimizationMetricsConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            fee_config: FeeOptimizationConfig::default(),
            batch_config: BatchProcessingConfig::default(),
            monitoring_config: PerformanceMonitoringConfig::default(),
            algorithm_config: AdaptiveAlgorithmConfig::default(),
            cost_config: CostModelingConfig::default(),
            priority_config: PriorityManagementConfig::default(),
            gas_config: GasPredictionConfig::default(),
            metrics_config: OptimizationMetricsConfig::default(),
        }
    }
}

/// Fee optimization configuration
#[derive(Debug, Clone)]
pub struct FeeOptimizationConfig {
    pub max_fee_multiplier: Decimal,
    pub min_fee_threshold: Decimal,
    pub congestion_sensitivity: Decimal,
    pub priority_weight: Decimal,
    pub cross_chain_optimization: bool,
    pub amm_integration: bool,
}

impl Default for FeeOptimizationConfig {
    fn default() -> Self {
        Self {
            max_fee_multiplier: Decimal::from(5),
            min_fee_threshold: Decimal::from_f64(0.001).unwrap(),
            congestion_sensitivity: Decimal::from_f64(1.5).unwrap(),
            priority_weight: Decimal::from_f64(0.3).unwrap(),
            cross_chain_optimization: true,
            amm_integration: true,
        }
    }
}

/// Transaction representation
#[derive(Debug, Clone)]
pub struct Transaction {
    pub transaction_id: TransactionId,
    pub transaction_type: TransactionType,
    pub source_network: NetworkId,
    pub target_network: NetworkId,
    pub amount: Decimal,
    pub fee: Decimal,
    pub execution_urgency: ExecutionUrgency,
    pub user_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Transaction {
    pub fn is_cross_chain(&self) -> bool {
        self.source_network != self.target_network
    }
    
    pub fn involves_amm(&self) -> bool {
        matches!(self.transaction_type, TransactionType::AMMSwap | TransactionType::LiquidityProvision)
    }
}

/// Transaction identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransactionId(pub uuid::Uuid);

impl TransactionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Network identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NetworkId {
    Ethereum,
    Polygon,
    Arbitrum,
    Optimism,
    BSC,
    Avalanche,
    Solana,
    Cosmos,
}

/// Transaction types
#[derive(Debug, Clone)]
pub enum TransactionType {
    Transfer,
    AMMSwap,
    LiquidityProvision,
    CrossChainBridge,
    GovernanceVote,
    StakingOperation,
    ProofSubmission,
}

/// Execution urgency levels
#[derive(Debug, Clone, Copy)]
pub enum ExecutionUrgency {
    Immediate,
    Fast,
    Standard,
    Economy,
}

/// Optimized transaction
#[derive(Debug, Clone)]
pub struct OptimizedTransaction {
    pub transaction_id: TransactionId,
    pub original_transaction: Transaction,
    pub optimal_fee: OptimalFee,
    pub priority: Priority,
    pub cross_chain_optimization: Option<CrossChainOptimization>,
    pub amm_optimization: Option<AMMOptimization>,
    pub estimated_execution_time: std::time::Duration,
    pub cost_savings: Decimal,
    pub optimization_confidence: Decimal,
    pub optimization_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Optimal fee calculation
#[derive(Debug, Clone)]
pub struct OptimalFee {
    pub base_fee: Decimal,
    pub congestion_multiplier: Decimal,
    pub priority_premium: Decimal,
    pub predicted_gas_price: Decimal,
    pub market_adjustment: Decimal,
    pub total_fee: Decimal,
    pub fee_breakdown: FeeBreakdown,
    pub optimization_strategy: OptimizationStrategy,
}

/// Fee breakdown
#[derive(Debug, Clone)]
pub struct FeeBreakdown {
    pub network_fee: Decimal,
    pub protocol_fee: Decimal,
    pub priority_fee: Decimal,
    pub cross_chain_fee: Decimal,
    pub amm_fee: Decimal,
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    StandardOptimization,
    CrossChainOptimization,
    AMMOptimization,
    HybridOptimization,
    BatchOptimization,
}

/// Transaction priority
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Priority {
    pub score: Decimal,
    pub factors: PriorityFactors,
    pub level: PriorityLevel,
}

/// Priority factors
#[derive(Debug, Clone)]
pub struct PriorityFactors {
    pub fee_premium: Decimal,
    pub user_tier: UserTier,
    pub transaction_value: Decimal,
    pub network_congestion: Decimal,
    pub time_sensitivity: Decimal,
}

/// Priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityLevel {
    Critical,
    High,
    Medium,
    Low,
}

/// User tiers for priority calculation
#[derive(Debug, Clone)]
pub enum UserTier {
    Premium,
    Standard,
    Basic,
}

/// Cross-chain optimization
#[derive(Debug, Clone)]
pub struct CrossChainOptimization {
    pub bridge_cost: BridgeCost,
    pub optimal_route: CrossChainRoute,
    pub arbitrage_opportunity: Option<ArbitrageOpportunity>,
    pub estimated_time: std::time::Duration,
    pub cost_savings: Decimal,
}

/// Bridge cost calculation
#[derive(Debug, Clone)]
pub struct BridgeCost {
    pub base_cost: Decimal,
    pub gas_cost: Decimal,
    pub protocol_fee: Decimal,
    pub total_cost: Decimal,
}

/// Cross-chain route
#[derive(Debug, Clone)]
pub struct CrossChainRoute {
    pub route_id: uuid::Uuid,
    pub source_network: NetworkId,
    pub target_network: NetworkId,
    pub intermediate_networks: Vec<NetworkId>,
    pub bridges: Vec<BridgeInfo>,
    pub total_cost: Decimal,
    pub estimated_time: std::time::Duration,
}

/// Bridge information
#[derive(Debug, Clone)]
pub struct BridgeInfo {
    pub bridge_id: String,
    pub bridge_type: BridgeType,
    pub source_network: NetworkId,
    pub target_network: NetworkId,
    pub fee_rate: Decimal,
    pub security_level: SecurityLevel,
}

/// Bridge types
#[derive(Debug, Clone)]
pub enum BridgeType {
    Native,
    Wrapped,
    Liquidity,
    Validator,
}

/// Security levels
#[derive(Debug, Clone)]
pub enum SecurityLevel {
    High,
    Medium,
    Low,
}

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub opportunity_id: uuid::Uuid,
    pub profit_potential: Decimal,
    pub execution_cost: Decimal,
    pub net_profit: Decimal,
    pub confidence_score: Decimal,
    pub time_window: std::time::Duration,
}

/// AMM optimization
#[derive(Debug, Clone)]
pub struct AMMOptimization {
    pub bonding_curve_impact: BondingCurveImpact,
    pub liquidity_impact: LiquidityImpact,
    pub predicted_slippage: Decimal,
    pub mev_protection_cost: MEVProtectionCost,
    pub optimized_pool_fee: Decimal,
    pub total_amm_cost: Decimal,
}

/// Bonding curve impact
#[derive(Debug, Clone)]
pub struct BondingCurveImpact {
    pub price_impact: Decimal,
    pub curve_shift: Decimal,
    pub optimal_trade_size: Decimal,
}

/// Liquidity impact
#[derive(Debug, Clone)]
pub struct LiquidityImpact {
    pub liquidity_utilization: Decimal,
    pub pool_depth_impact: Decimal,
    pub rebalancing_cost: Decimal,
}

/// MEV protection cost
#[derive(Debug, Clone)]
pub struct MEVProtectionCost {
    pub protection_fee: Decimal,
    pub expected_mev_loss: Decimal,
    pub net_protection_value: Decimal,
}

/// Fee optimization metrics
#[derive(Debug, Clone)]
pub struct FeeOptimizationMetrics {
    pub total_transactions_optimized: u64,
    pub average_cost_savings: Decimal,
    pub optimization_success_rate: Decimal,
    pub average_execution_time: std::time::Duration,
    pub queue_utilization: Decimal,
    pub gas_prediction_accuracy: Decimal,
    pub cross_chain_optimization_rate: Decimal,
    pub amm_optimization_rate: Decimal,
}

/// Fee optimization event
#[derive(Debug, Clone)]
pub struct FeeOptimizationEvent {
    pub event_id: uuid::Uuid,
    pub transaction_id: TransactionId,
    pub optimization_type: OptimizationType,
    pub original_fee: Decimal,
    pub optimized_fee: Decimal,
    pub cost_savings: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Optimization types
#[derive(Debug, Clone)]
pub enum OptimizationType {
    FeeOptimization,
    BatchOptimization,
    CrossChainOptimization,
    AMMOptimization,
    PriorityOptimization,
}

/// Optimization identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OptimizationId(pub uuid::Uuid);

impl OptimizationId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Active optimization tracking
#[derive(Debug, Clone)]
pub struct ActiveOptimization {
    pub optimization_id: OptimizationId,
    pub transaction_id: TransactionId,
    pub optimization_type: OptimizationType,
    pub status: OptimizationStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
}

/// Optimization status
#[derive(Debug, Clone)]
pub enum OptimizationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Fee model for networks
#[derive(Debug, Clone)]
pub struct FeeModel {
    pub network_id: NetworkId,
    pub base_fee_formula: String,
    pub congestion_model: String,
    pub gas_price_model: String,
    pub priority_model: String,
    pub update_frequency: std::time::Duration,
}

/// Prediction models for gas prices
#[derive(Debug, Clone)]
pub enum PredictionModel {
    ARIMA,
    LSTM,
    LinearRegression,
}

// Placeholder configurations for other modules
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    pub max_batch_size: usize,
    pub batch_timeout: std::time::Duration,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_timeout: std::time::Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMonitoringConfig {
    pub monitoring_interval: std::time::Duration,
    pub metrics_retention: std::time::Duration,
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: std::time::Duration::from_secs(10),
            metrics_retention: std::time::Duration::from_secs(86400), // 24 hours
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveAlgorithmConfig {
    pub learning_rate: Decimal,
    pub adaptation_threshold: Decimal,
}

impl Default for AdaptiveAlgorithmConfig {
    fn default() -> Self {
        Self {
            learning_rate: Decimal::from_f64(0.01).unwrap(),
            adaptation_threshold: Decimal::from_f64(0.1).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CostModelingConfig {
    pub model_update_frequency: std::time::Duration,
    pub historical_data_window: std::time::Duration,
}

impl Default for CostModelingConfig {
    fn default() -> Self {
        Self {
            model_update_frequency: std::time::Duration::from_secs(300), // 5 minutes
            historical_data_window: std::time::Duration::from_secs(604800), // 7 days
        }
    }
}

#[derive(Debug, Clone)]
pub struct PriorityManagementConfig {
    pub max_queue_size: usize,
    pub fairness_threshold: Decimal,
}

impl Default for PriorityManagementConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            fairness_threshold: Decimal::from_f64(0.8).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GasPredictionConfig {
    pub prediction_window: std::time::Duration,
    pub model_ensemble_size: usize,
}

impl Default for GasPredictionConfig {
    fn default() -> Self {
        Self {
            prediction_window: std::time::Duration::from_secs(3600), // 1 hour
            model_ensemble_size: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationMetricsConfig {
    pub metrics_collection_interval: std::time::Duration,
    pub metrics_aggregation_window: std::time::Duration,
}

impl Default for OptimizationMetricsConfig {
    fn default() -> Self {
        Self {
            metrics_collection_interval: std::time::Duration::from_secs(60), // 1 minute
            metrics_aggregation_window: std::time::Duration::from_secs(3600), // 1 hour
        }
    }
}

// Additional types for batch processing optimization

/// Execution requirements for transactions
#[derive(Debug, Clone)]
pub struct ExecutionRequirements {
    pub estimated_duration: std::time::Duration,
    pub priority_level: PriorityLevel,
    pub resource_intensity: ResourceIntensity,
    pub atomicity_requirements: AtomicityRequirements,
}

/// Resource needs for transactions
#[derive(Debug, Clone)]
pub struct ResourceNeeds {
    pub cpu_requirement: rust_decimal::Decimal,
    pub memory_requirement: rust_decimal::Decimal,
    pub network_requirement: rust_decimal::Decimal,
    pub storage_requirement: rust_decimal::Decimal,
}

/// Resource intensity levels
#[derive(Debug, Clone)]
pub enum ResourceIntensity {
    Low,
    Medium,
    High,
    Critical,
}

/// Atomicity requirements
#[derive(Debug, Clone)]
pub enum AtomicityRequirements {
    None,
    Partial,
    Full,
    CrossChain,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    Serialize,
    Parallelize,
    Optimize,
    Defer,
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct TaskExecutionResult {
    pub task_id: uuid::Uuid,
    pub transaction_id: TransactionId,
    pub execution_success: bool,
    pub execution_time: std::time::Duration,
    pub resource_usage: ResourceUsage,
    pub error_message: Option<String>,
}

/// Synchronized results
#[derive(Debug, Clone)]
pub struct SynchronizedResults {
    pub results: Vec<TaskExecutionResult>,
    pub synchronization_time: std::time::Duration,
    pub consistency_validated: bool,
}

/// Batch execution result
#[derive(Debug, Clone)]
pub struct BatchExecutionResult {
    pub batch_id: crate::batch_processing::BatchId,
    pub execution_success: bool,
    pub executed_transactions: usize,
    pub execution_time: std::time::Duration,
    pub cost_savings: rust_decimal::Decimal,
    pub efficiency_metrics: BatchEfficiencyMetrics,
}

/// Batch efficiency metrics
#[derive(Debug, Clone)]
pub struct BatchEfficiencyMetrics {
    pub throughput: rust_decimal::Decimal,
    pub resource_utilization: rust_decimal::Decimal,
    pub parallelization_efficiency: rust_decimal::Decimal,
}

/// Dependency cache for optimization
#[derive(Debug, Clone)]
pub struct DependencyCache {
    pub cache: std::collections::HashMap<TransactionId, Vec<TransactionId>>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl DependencyCache {
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            last_updated: chrono::Utc::now(),
        }
    }
}

/// Dependency metrics for analysis
#[derive(Debug, Clone)]
pub struct DependencyMetrics {
    pub total_dependencies: usize,
    pub average_dependency_depth: rust_decimal::Decimal,
    pub critical_path_length: usize,
    pub parallelization_potential: rust_decimal::Decimal,
}
