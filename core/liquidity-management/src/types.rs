//! Type definitions for liquidity management system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rust_decimal::Decimal;

/// Unique identifier for liquidity pools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolId(pub uuid::Uuid);

impl PoolId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for PoolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for assets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetId(pub uuid::Uuid);

impl AssetId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for AssetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for liquidity providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LiquidityProvider(pub uuid::Uuid);

impl LiquidityProvider {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Unique identifier for traders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraderId(pub uuid::Uuid);

impl TraderId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Unique identifier for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub uuid::Uuid);

impl OperationId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Asset amount with identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetAmount {
    pub asset_id: AssetId,
    pub amount: Decimal,
}

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfiguration {
    pub pool_id: PoolId,
    pub assets: Vec<AssetId>,
    pub bonding_curve: BondingCurve,
    pub fee_rate: Decimal,
    pub is_cross_chain: bool,
    pub supported_chains: Vec<ChainId>,
    pub risk_parameters: RiskParameters,
}

/// Bonding curve types and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BondingCurve {
    ConstantProduct {
        k: Decimal,
    },
    ConstantSum {
        k: Decimal,
    },
    ConstantMean {
        weights: HashMap<AssetId, Decimal>,
    },
    Stable {
        amplification_factor: Decimal,
    },
    Custom {
        curve_type: BondingCurveType,
        parameters: BondingCurveParameters,
    },
}

impl BondingCurve {
    pub fn get_parameters(&self) -> BondingCurveParameters {
        match self {
            BondingCurve::ConstantProduct { k } => {
                let mut params = BondingCurveParameters::new();
                params.set_parameter("k".to_string(), *k);
                params
            }
            BondingCurve::ConstantSum { k } => {
                let mut params = BondingCurveParameters::new();
                params.set_parameter("k".to_string(), *k);
                params
            }
            BondingCurve::ConstantMean { weights } => {
                let mut params = BondingCurveParameters::new();
                for (asset_id, weight) in weights {
                    params.set_weight(format!("{}", asset_id), *weight);
                }
                params
            }
            BondingCurve::Stable { amplification_factor } => {
                let mut params = BondingCurveParameters::new();
                params.set_amplification_factor(*amplification_factor);
                params
            }
            BondingCurve::Custom { parameters, .. } => parameters.clone(),
        }
    }
}

/// Bonding curve type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BondingCurveType {
    ConstantProduct,
    ConstantSum,
    ConstantMean,
    Stable,
    Custom,
}

/// Bonding curve parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondingCurveParameters {
    parameters: HashMap<String, Decimal>,
    weights: HashMap<String, Decimal>,
    amplification_factor: Option<Decimal>,
}

impl BondingCurveParameters {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            weights: HashMap::new(),
            amplification_factor: None,
        }
    }
    
    pub fn set_parameter(&mut self, key: String, value: Decimal) {
        self.parameters.insert(key, value);
    }
    
    pub fn get_parameter(&self, key: &str) -> Option<Decimal> {
        self.parameters.get(key).copied()
    }
    
    pub fn set_weight(&mut self, key: String, value: Decimal) {
        self.weights.insert(key, value);
    }
    
    pub fn get_weights(&self) -> crate::error::Result<&HashMap<String, Decimal>> {
        Ok(&self.weights)
    }
    
    pub fn set_amplification_factor(&mut self, factor: Decimal) {
        self.amplification_factor = Some(factor);
    }
    
    pub fn get_amplification_factor(&self) -> Option<Decimal> {
        self.amplification_factor
    }
}

/// Chain identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainId(pub u64);

/// Risk parameters for pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    pub max_slippage: Decimal,
    pub max_price_impact: Decimal,
    pub liquidity_threshold: Decimal,
    pub volatility_threshold: Decimal,
    pub correlation_threshold: Decimal,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_slippage: Decimal::from(5), // 5%
            max_price_impact: Decimal::from(10), // 10%
            liquidity_threshold: Decimal::from(1000000), // $1M
            volatility_threshold: Decimal::from(50), // 50%
            correlation_threshold: Decimal::from(80), // 80%
        }
    }
}

/// Liquidity position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityPosition {
    pub provider: LiquidityProvider,
    pub pool_id: PoolId,
    pub liquidity_amount: Decimal,
    pub assets_provided: Vec<AssetAmount>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_update: chrono::DateTime<chrono::Utc>,
    pub fees_earned: HashMap<AssetId, Decimal>,
    pub impermanent_loss: Decimal,
}

/// Swap parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapParameters {
    pub pool_id: PoolId,
    pub input_asset: AssetAmount,
    pub output_asset_id: AssetId,
    pub min_output_amount: Decimal,
    pub max_slippage: Decimal,
}

/// Swap result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapResult {
    pub pool_id: PoolId,
    pub trader: TraderId,
    pub input_asset: AssetId,
    pub output_asset: AssetId,
    pub input_amount: Decimal,
    pub output_amount: Decimal,
    pub fee_amount: Decimal,
    pub price_impact: Decimal,
    pub executed_at: chrono::DateTime<chrono::Utc>,
}

/// Swap quote for price estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapQuote {
    pub pool_id: PoolId,
    pub input_asset: AssetId,
    pub output_asset: AssetId,
    pub input_amount: Decimal,
    pub output_amount: Decimal,
    pub fee_amount: Decimal,
    pub price_impact: Decimal,
    pub valid_until: chrono::DateTime<chrono::Utc>,
}

/// Pool information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfo {
    pub pool_id: PoolId,
    pub configuration: PoolConfiguration,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub total_liquidity: Decimal,
    pub active_providers: u64,
    pub volume_24h: Decimal,
    pub fees_collected: Decimal,
}

/// Pending operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOperation {
    pub operation_id: OperationId,
    pub operation_type: OperationType,
    pub pool_id: PoolId,
    pub initiated_at: chrono::DateTime<chrono::Utc>,
    pub expected_completion: chrono::DateTime<chrono::Utc>,
    pub status: OperationStatus,
}

/// Operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationType {
    PoolCreation,
    LiquidityAddition,
    LiquidityRemoval,
    Swap,
    CrossChainTransfer,
    Rebalancing,
}

/// Operation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Cross-chain state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainState {
    pub chain_states: HashMap<ChainId, ChainState>,
    pub last_sync: chrono::DateTime<chrono::Utc>,
    pub sync_proof: Option<qross_zk_verification::ProofId>,
}

/// Chain-specific state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainState {
    pub chain_id: ChainId,
    pub reserves: HashMap<AssetId, Decimal>,
    pub last_block: u64,
    pub state_root: Vec<u8>,
}

/// Price information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceInfo {
    pub asset_id: AssetId,
    pub price: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: PriceSource,
}

/// Price source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriceSource {
    Pool,
    Oracle,
    Aggregated,
}

/// Arbitrage opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageOpportunity {
    pub opportunity_id: uuid::Uuid,
    pub source_pool: PoolId,
    pub target_pool: PoolId,
    pub asset_pair: (AssetId, AssetId),
    pub profit_potential: Decimal,
    pub required_capital: Decimal,
    pub execution_cost: Decimal,
    pub detected_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// Pool risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolRiskMetrics {
    pub pool_id: PoolId,
    pub liquidity_risk: Decimal,
    pub volatility_risk: Decimal,
    pub correlation_risk: Decimal,
    pub impermanent_loss_risk: Decimal,
    pub overall_risk_score: Decimal,
    pub calculated_at: chrono::DateTime<chrono::Utc>,
}

/// Yield optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldOptimizationSuggestion {
    pub suggestion_id: uuid::Uuid,
    pub provider: LiquidityProvider,
    pub current_pool: PoolId,
    pub suggested_pool: PoolId,
    pub expected_yield_improvement: Decimal,
    pub migration_cost: Decimal,
    pub confidence: Decimal,
    pub valid_until: chrono::DateTime<chrono::Utc>,
}

/// Fee structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    pub pool_id: PoolId,
    pub base_fee: Decimal,
    pub dynamic_fee_enabled: bool,
    pub fee_tiers: Vec<FeeTier>,
    pub protocol_fee_share: Decimal,
}

/// Fee tier based on volume or other criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTier {
    pub threshold: Decimal,
    pub fee_rate: Decimal,
}

/// Configuration structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityManagementConfig {
    pub amm_config: AMMConfig,
    pub pool_config: PoolConfig,
    pub arbitrage_config: ArbitrageConfig,
    pub risk_config: RiskConfig,
    pub bridge_config: BridgeConfig,
    pub oracle_config: OracleConfig,
    pub yield_config: YieldConfig,
    pub default_max_slippage: Decimal,
    pub proof_coordination_enabled: bool,
    pub cross_chain_enabled: bool,
}

impl Default for LiquidityManagementConfig {
    fn default() -> Self {
        Self {
            amm_config: AMMConfig::default(),
            pool_config: PoolConfig::default(),
            arbitrage_config: ArbitrageConfig::default(),
            risk_config: RiskConfig::default(),
            bridge_config: BridgeConfig::default(),
            oracle_config: OracleConfig::default(),
            yield_config: YieldConfig::default(),
            default_max_slippage: Decimal::from(5), // 5%
            proof_coordination_enabled: true,
            cross_chain_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AMMConfig {
    pub supported_curves: Vec<BondingCurveType>,
    pub default_fee_rate: Decimal,
    pub min_liquidity: Decimal,
    pub max_price_impact: Decimal,
    pub enable_dynamic_fees: bool,
    pub proof_generation_enabled: bool,
}

impl Default for AMMConfig {
    fn default() -> Self {
        Self {
            supported_curves: vec![
                BondingCurveType::ConstantProduct,
                BondingCurveType::ConstantSum,
                BondingCurveType::ConstantMean,
                BondingCurveType::Stable,
            ],
            default_fee_rate: Decimal::from(3) / Decimal::from(1000), // 0.3%
            min_liquidity: Decimal::from(1000),
            max_price_impact: Decimal::from(10), // 10%
            enable_dynamic_fees: true,
            proof_generation_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub max_pools: usize,
    pub min_assets_per_pool: usize,
    pub max_assets_per_pool: usize,
    pub default_risk_parameters: RiskParameters,
    pub enable_cross_chain_pools: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pools: 1000,
            min_assets_per_pool: 2,
            max_assets_per_pool: 8,
            default_risk_parameters: RiskParameters::default(),
            enable_cross_chain_pools: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    pub detection_enabled: bool,
    pub min_profit_threshold: Decimal,
    pub max_execution_time: chrono::Duration,
    pub mev_protection_enabled: bool,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            detection_enabled: true,
            min_profit_threshold: Decimal::from(1), // 1%
            max_execution_time: chrono::Duration::seconds(30),
            mev_protection_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub risk_assessment_enabled: bool,
    pub max_pool_risk_score: Decimal,
    pub impermanent_loss_protection: bool,
    pub volatility_monitoring: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            risk_assessment_enabled: true,
            max_pool_risk_score: Decimal::from(80), // 80%
            impermanent_loss_protection: true,
            volatility_monitoring: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub supported_chains: Vec<ChainId>,
    pub bridge_fee: Decimal,
    pub confirmation_blocks: HashMap<ChainId, u64>,
    pub max_transfer_amount: Decimal,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            supported_chains: vec![ChainId(1), ChainId(137), ChainId(56)], // Ethereum, Polygon, BSC
            bridge_fee: Decimal::from(1) / Decimal::from(1000), // 0.1%
            confirmation_blocks: HashMap::new(),
            max_transfer_amount: Decimal::from(1000000), // $1M
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    pub price_feeds: Vec<String>,
    pub update_frequency: chrono::Duration,
    pub price_deviation_threshold: Decimal,
    pub fallback_enabled: bool,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            price_feeds: vec!["chainlink".to_string(), "uniswap".to_string()],
            update_frequency: chrono::Duration::seconds(60),
            price_deviation_threshold: Decimal::from(5), // 5%
            fallback_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldConfig {
    pub optimization_enabled: bool,
    pub rebalancing_threshold: Decimal,
    pub compound_frequency: chrono::Duration,
    pub gas_optimization: bool,
}

impl Default for YieldConfig {
    fn default() -> Self {
        Self {
            optimization_enabled: true,
            rebalancing_threshold: Decimal::from(5), // 5%
            compound_frequency: chrono::Duration::hours(24),
            gas_optimization: true,
        }
    }
}

/// Transaction for MEV analysis
#[derive(Debug, Clone)]
pub struct Transaction {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub value: Decimal,
    pub gas_price: Decimal,
    pub gas_limit: u64,
    pub input_data: Vec<u8>,
    pub block_number: u64,
    pub transaction_index: u32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Transaction pattern analysis result
#[derive(Debug, Clone)]
pub struct TransactionPatternAnalysis {
    pub pattern_id: String,
    pub transaction_hashes: Vec<String>,
    pub validator_id: Option<qross_consensus::ValidatorId>,
    pub extracted_value: Option<Decimal>,
    pub price_impact: Option<Decimal>,
    pub timing_analysis: Option<TimingAnalysis>,
    pub confidence: Decimal,
}

/// Timing analysis for MEV detection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimingAnalysis {
    pub sequence_timing: Vec<std::time::Duration>,
    pub average_interval: std::time::Duration,
    pub timing_variance: f64,
    pub suspicious_patterns: Vec<String>,
}

/// Timing constraints for patterns
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    pub min_interval: std::time::Duration,
    pub max_interval: std::time::Duration,
    pub sequence_length: usize,
}

/// Parameter pattern for transaction matching
#[derive(Debug, Clone)]
pub struct ParameterPattern {
    pub parameter_type: String,
    pub pattern_regex: String,
    pub value_range: Option<(Decimal, Decimal)>,
}

/// Value constraints for transactions
#[derive(Debug, Clone)]
pub struct ValueConstraints {
    pub min_value: Option<Decimal>,
    pub max_value: Option<Decimal>,
    pub value_pattern: Option<String>,
}

/// Gas constraints for transactions
#[derive(Debug, Clone)]
pub struct GasConstraints {
    pub min_gas_price: Option<Decimal>,
    pub max_gas_price: Option<Decimal>,
    pub gas_price_pattern: Option<String>,
}

/// Gas pattern for analysis
#[derive(Debug, Clone)]
pub struct GasPattern {
    pub base_gas_price: Decimal,
    pub gas_price_variance: Decimal,
    pub gas_escalation_pattern: Vec<Decimal>,
}

/// Profit extraction information
#[derive(Debug, Clone)]
pub struct ProfitExtraction {
    pub extraction_method: String,
    pub estimated_profit: Decimal,
    pub profit_source: String,
}

/// Price matrix for cross-chain analysis
#[derive(Debug, Clone)]
pub struct PriceMatrix {
    prices: std::collections::HashMap<(ChainId, AssetId), PricePoint>,
}

impl PriceMatrix {
    pub fn new() -> Self {
        Self {
            prices: std::collections::HashMap::new(),
        }
    }

    pub fn set_price(&mut self, chain_id: ChainId, asset_id: AssetId, price_point: PricePoint) {
        self.prices.insert((chain_id, asset_id), price_point);
    }

    pub fn get_all_assets(&self) -> std::collections::HashSet<AssetId> {
        self.prices.keys().map(|(_, asset_id)| *asset_id).collect()
    }

    pub fn get_asset_prices(&self, asset_id: AssetId) -> Vec<(ChainId, PricePoint)> {
        self.prices.iter()
            .filter(|((_, aid), _)| *aid == asset_id)
            .map(|((chain_id, _), price_point)| (*chain_id, price_point.clone()))
            .collect()
    }
}

/// Latency profile for oracles
#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub average_latency: std::time::Duration,
    pub p95_latency: std::time::Duration,
    pub p99_latency: std::time::Duration,
}

impl LatencyProfile {
    pub fn new() -> Self {
        Self {
            average_latency: std::time::Duration::from_millis(100),
            p95_latency: std::time::Duration::from_millis(200),
            p99_latency: std::time::Duration::from_millis(500),
        }
    }
}

/// Fair value result
#[derive(Debug, Clone)]
pub struct FairValueResult {
    pub asset_id: AssetId,
    pub fair_value: Decimal,
    pub confidence: Decimal,
    pub price_sources: std::collections::HashMap<ChainId, PricePoint>,
    pub calculated_at: chrono::DateTime<chrono::Utc>,
    pub volatility_adjustment: Decimal,
    pub liquidity_adjustment: Decimal,
}

/// Fair value calculation result
#[derive(Debug, Clone)]
pub struct FairValue {
    pub value: Decimal,
    pub volatility_adjustment: Decimal,
    pub liquidity_adjustment: Decimal,
    pub confidence: Decimal,
}

/// Fair value validation result
#[derive(Debug, Clone)]
pub struct FairValueValidation {
    pub is_valid: bool,
    pub confidence: Decimal,
    pub validation_errors: Vec<String>,
}

/// Opportunity validation result
#[derive(Debug, Clone)]
pub struct OpportunityValidation {
    pub opportunity_id: uuid::Uuid,
    pub is_valid: bool,
    pub profit_potential: Decimal,
    pub risk_score: Decimal,
    pub execution_probability: Decimal,
    pub validated_at: chrono::DateTime<chrono::Utc>,
}

/// Current profit calculation
#[derive(Debug, Clone)]
pub struct CurrentProfit {
    pub profit_potential: Decimal,
    pub execution_cost: Decimal,
    pub net_profit: Decimal,
    pub calculated_at: chrono::DateTime<chrono::Utc>,
}

/// Risk assessment result
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk_score: Decimal,
    pub execution_probability: Decimal,
    pub risk_factors: std::collections::HashMap<String, Decimal>,
    pub mitigation_suggestions: Vec<String>,
}

/// Arbitrage execution result
#[derive(Debug, Clone)]
pub struct ArbitrageExecutionResult {
    pub execution_path: Vec<crate::arbitrage_detection::ExecutionStep>,
    pub actual_profit: Decimal,
    pub execution_time: std::time::Duration,
    pub gas_used: u64,
    pub slippage_experienced: Decimal,
}

/// Monitoring thread for real-time MEV detection
#[derive(Debug, Clone)]
pub struct MonitoringThread {
    pub thread_id: String,
    pub monitored_chains: Vec<ChainId>,
    pub detection_algorithms: Vec<String>,
}

/// Alert system for MEV detection
#[derive(Debug, Clone)]
pub struct AlertSystem {
    pub alert_channels: Vec<String>,
    pub severity_thresholds: std::collections::HashMap<String, Decimal>,
}

/// Response coordinator for MEV mitigation
#[derive(Debug, Clone)]
pub struct ResponseCoordinator {
    pub response_strategies: Vec<String>,
    pub escalation_policies: std::collections::HashMap<String, String>,
}
