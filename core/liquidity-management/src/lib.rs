//! Cross-chain liquidity management system with AMM principles and arbitrage detection

pub mod amm_core;
pub mod bonding_curves;
pub mod liquidity_pools;
pub mod arbitrage_detection;
pub mod risk_management;
pub mod cross_chain_bridge;
pub mod price_oracle;
pub mod yield_optimization;
pub mod types;
pub mod error;
pub mod metrics;

pub use amm_core::*;
pub use types::*;
pub use error::*;

use crate::{types::*, error::*};
use qross_consensus::ValidatorId;
use qross_zk_verification::ProofId;
use qross_p2p_network::PeerId;
use std::collections::HashMap;

/// Main liquidity management engine
pub struct LiquidityManagementEngine {
    amm_core: amm_core::AMMCoreEngine,
    liquidity_pools: liquidity_pools::CrossChainLiquidityPools,
    arbitrage_detector: arbitrage_detection::ArbitrageDetectionEngine,
    risk_manager: risk_management::RiskManagementSystem,
    cross_chain_bridge: cross_chain_bridge::CrossChainBridge,
    price_oracle: price_oracle::PriceOracle,
    yield_optimizer: yield_optimization::YieldOptimizer,
    metrics: metrics::LiquidityMetrics,
    config: LiquidityManagementConfig,
    active_pools: dashmap::DashMap<PoolId, PoolInfo>,
    pending_operations: dashmap::DashMap<OperationId, PendingOperation>,
}

impl LiquidityManagementEngine {
    /// Create a new liquidity management engine
    pub fn new(config: LiquidityManagementConfig) -> Self {
        Self {
            amm_core: amm_core::AMMCoreEngine::new(config.amm_config.clone()),
            liquidity_pools: liquidity_pools::CrossChainLiquidityPools::new(config.pool_config.clone()),
            arbitrage_detector: arbitrage_detection::ArbitrageDetectionEngine::new(config.arbitrage_config.clone()),
            risk_manager: risk_management::RiskManagementSystem::new(config.risk_config.clone()),
            cross_chain_bridge: cross_chain_bridge::CrossChainBridge::new(config.bridge_config.clone()),
            price_oracle: price_oracle::PriceOracle::new(config.oracle_config.clone()),
            yield_optimizer: yield_optimization::YieldOptimizer::new(config.yield_config.clone()),
            metrics: metrics::LiquidityMetrics::new(),
            active_pools: dashmap::DashMap::new(),
            pending_operations: dashmap::DashMap::new(),
            config,
        }
    }
    
    /// Start the liquidity management engine
    pub async fn start(&mut self) -> Result<()> {
        // Initialize all subsystems
        self.amm_core.start().await?;
        self.liquidity_pools.start().await?;
        self.arbitrage_detector.start().await?;
        self.risk_manager.start().await?;
        self.cross_chain_bridge.start().await?;
        self.price_oracle.start().await?;
        self.yield_optimizer.start().await?;
        
        tracing::info!("Liquidity management engine started");
        
        Ok(())
    }
    
    /// Create a new liquidity pool
    pub async fn create_pool(
        &mut self,
        pool_config: PoolConfiguration,
        initial_liquidity: Vec<AssetAmount>,
    ) -> Result<PoolId> {
        // Validate pool configuration
        self.validate_pool_configuration(&pool_config).await?;
        
        // Create pool through AMM core
        let pool_id = self.amm_core.create_pool(pool_config.clone(), initial_liquidity.clone()).await?;
        
        // Initialize cross-chain pool if needed
        if pool_config.is_cross_chain {
            self.liquidity_pools.initialize_cross_chain_pool(pool_id, &pool_config).await?;
        }
        
        // Set up risk management
        self.risk_manager.initialize_pool_risk_management(pool_id, &pool_config).await?;
        
        // Register with arbitrage detector
        self.arbitrage_detector.register_pool(pool_id, &pool_config).await?;
        
        // Store pool information
        let pool_info = PoolInfo {
            pool_id,
            configuration: pool_config,
            created_at: chrono::Utc::now(),
            total_liquidity: initial_liquidity.iter().map(|a| a.amount).sum(),
            active_providers: 0,
            volume_24h: rust_decimal::Decimal::ZERO,
            fees_collected: rust_decimal::Decimal::ZERO,
        };
        
        self.active_pools.insert(pool_id, pool_info);
        
        tracing::info!("Created liquidity pool: {}", pool_id);
        
        Ok(pool_id)
    }
    
    /// Add liquidity to a pool
    pub async fn add_liquidity(
        &mut self,
        pool_id: PoolId,
        provider: LiquidityProvider,
        assets: Vec<AssetAmount>,
    ) -> Result<LiquidityPosition> {
        // Validate liquidity addition
        self.validate_liquidity_addition(pool_id, &assets).await?;
        
        // Check risk limits
        self.risk_manager.check_liquidity_addition_risk(pool_id, &assets).await?;
        
        // Execute liquidity addition through AMM core
        let position = self.amm_core.add_liquidity(pool_id, provider, assets).await?;
        
        // Update pool information
        if let Some(mut pool_info) = self.active_pools.get_mut(&pool_id) {
            pool_info.active_providers += 1;
            pool_info.total_liquidity += position.liquidity_amount;
        }
        
        // Update metrics
        self.metrics.record_liquidity_addition(pool_id, &position);
        
        tracing::info!("Added liquidity to pool {}: {:?}", pool_id, position);
        
        Ok(position)
    }
    
    /// Remove liquidity from a pool
    pub async fn remove_liquidity(
        &mut self,
        pool_id: PoolId,
        provider: LiquidityProvider,
        liquidity_amount: rust_decimal::Decimal,
    ) -> Result<Vec<AssetAmount>> {
        // Validate liquidity removal
        self.validate_liquidity_removal(pool_id, provider, liquidity_amount).await?;
        
        // Check risk limits
        self.risk_manager.check_liquidity_removal_risk(pool_id, liquidity_amount).await?;
        
        // Execute liquidity removal through AMM core
        let assets = self.amm_core.remove_liquidity(pool_id, provider, liquidity_amount).await?;
        
        // Update pool information
        if let Some(mut pool_info) = self.active_pools.get_mut(&pool_id) {
            pool_info.total_liquidity -= liquidity_amount;
        }
        
        // Update metrics
        self.metrics.record_liquidity_removal(pool_id, liquidity_amount);
        
        tracing::info!("Removed liquidity from pool {}: {:?}", pool_id, assets);
        
        Ok(assets)
    }
    
    /// Execute a swap
    pub async fn swap(
        &mut self,
        pool_id: PoolId,
        input_asset: AssetAmount,
        output_asset_id: AssetId,
        min_output_amount: rust_decimal::Decimal,
        trader: TraderId,
    ) -> Result<SwapResult> {
        // Validate swap parameters
        self.validate_swap_parameters(pool_id, &input_asset, output_asset_id, min_output_amount).await?;
        
        // Check for arbitrage opportunities
        let arbitrage_info = self.arbitrage_detector.check_arbitrage_opportunity(
            pool_id, 
            input_asset.asset_id, 
            output_asset_id
        ).await?;
        
        // Apply MEV protection if needed
        let protected_swap = if arbitrage_info.is_some() {
            self.apply_mev_protection(pool_id, input_asset, output_asset_id, min_output_amount).await?
        } else {
            SwapParameters {
                pool_id,
                input_asset,
                output_asset_id,
                min_output_amount,
                max_slippage: self.config.default_max_slippage,
            }
        };
        
        // Execute swap through AMM core
        let swap_result = self.amm_core.execute_swap(protected_swap, trader).await?;
        
        // Update pool metrics
        if let Some(mut pool_info) = self.active_pools.get_mut(&pool_id) {
            pool_info.volume_24h += swap_result.input_amount;
            pool_info.fees_collected += swap_result.fee_amount;
        }
        
        // Update metrics
        self.metrics.record_swap(pool_id, &swap_result);
        
        tracing::info!("Executed swap in pool {}: {:?}", pool_id, swap_result);
        
        Ok(swap_result)
    }
    
    /// Get pool information
    pub async fn get_pool_info(&self, pool_id: PoolId) -> Result<PoolInfo> {
        self.active_pools.get(&pool_id)
            .map(|info| info.clone())
            .ok_or(LiquidityError::PoolNotFound(pool_id))
    }
    
    /// Get current pool prices
    pub async fn get_pool_prices(&self, pool_id: PoolId) -> Result<HashMap<AssetId, rust_decimal::Decimal>> {
        self.amm_core.get_current_prices(pool_id).await
    }
    
    /// Get liquidity provider position
    pub async fn get_liquidity_position(
        &self,
        pool_id: PoolId,
        provider: LiquidityProvider,
    ) -> Result<LiquidityPosition> {
        self.amm_core.get_liquidity_position(pool_id, provider).await
    }
    
    /// Calculate swap output
    pub async fn calculate_swap_output(
        &self,
        pool_id: PoolId,
        input_asset: AssetAmount,
        output_asset_id: AssetId,
    ) -> Result<SwapQuote> {
        self.amm_core.calculate_swap_output(pool_id, input_asset, output_asset_id).await
    }
    
    /// Get arbitrage opportunities
    pub async fn get_arbitrage_opportunities(&self) -> Result<Vec<ArbitrageOpportunity>> {
        self.arbitrage_detector.get_current_opportunities().await
    }
    
    /// Get risk metrics for a pool
    pub async fn get_pool_risk_metrics(&self, pool_id: PoolId) -> Result<PoolRiskMetrics> {
        self.risk_manager.get_pool_risk_metrics(pool_id).await
    }
    
    /// Get yield optimization suggestions
    pub async fn get_yield_optimization_suggestions(
        &self,
        provider: LiquidityProvider,
    ) -> Result<Vec<YieldOptimizationSuggestion>> {
        self.yield_optimizer.get_optimization_suggestions(provider).await
    }
    
    /// Get liquidity management statistics
    pub fn get_statistics(&self) -> &metrics::LiquidityMetrics {
        &self.metrics
    }
    
    // Private helper methods
    
    async fn validate_pool_configuration(&self, config: &PoolConfiguration) -> Result<()> {
        // Validate asset pairs
        if config.assets.len() < 2 {
            return Err(LiquidityError::InvalidPoolConfiguration("Pool must have at least 2 assets".to_string()));
        }
        
        // Validate bonding curve
        if !self.is_valid_bonding_curve(&config.bonding_curve) {
            return Err(LiquidityError::InvalidPoolConfiguration("Invalid bonding curve".to_string()));
        }
        
        // Validate fee structure
        if config.fee_rate > rust_decimal::Decimal::from(100) {
            return Err(LiquidityError::InvalidPoolConfiguration("Fee rate cannot exceed 100%".to_string()));
        }
        
        Ok(())
    }
    
    async fn validate_liquidity_addition(&self, pool_id: PoolId, assets: &[AssetAmount]) -> Result<()> {
        // Check if pool exists
        if !self.active_pools.contains_key(&pool_id) {
            return Err(LiquidityError::PoolNotFound(pool_id));
        }
        
        // Validate asset amounts
        for asset in assets {
            if asset.amount <= rust_decimal::Decimal::ZERO {
                return Err(LiquidityError::InvalidAmount("Asset amount must be positive".to_string()));
            }
        }
        
        Ok(())
    }
    
    async fn validate_liquidity_removal(
        &self,
        pool_id: PoolId,
        provider: LiquidityProvider,
        amount: rust_decimal::Decimal,
    ) -> Result<()> {
        // Check if pool exists
        if !self.active_pools.contains_key(&pool_id) {
            return Err(LiquidityError::PoolNotFound(pool_id));
        }
        
        // Validate amount
        if amount <= rust_decimal::Decimal::ZERO {
            return Err(LiquidityError::InvalidAmount("Liquidity amount must be positive".to_string()));
        }
        
        // Check if provider has sufficient liquidity
        let position = self.amm_core.get_liquidity_position(pool_id, provider).await?;
        if position.liquidity_amount < amount {
            return Err(LiquidityError::InsufficientLiquidity);
        }
        
        Ok(())
    }
    
    async fn validate_swap_parameters(
        &self,
        pool_id: PoolId,
        input_asset: &AssetAmount,
        output_asset_id: AssetId,
        min_output_amount: rust_decimal::Decimal,
    ) -> Result<()> {
        // Check if pool exists
        if !self.active_pools.contains_key(&pool_id) {
            return Err(LiquidityError::PoolNotFound(pool_id));
        }
        
        // Validate input amount
        if input_asset.amount <= rust_decimal::Decimal::ZERO {
            return Err(LiquidityError::InvalidAmount("Input amount must be positive".to_string()));
        }
        
        // Validate minimum output amount
        if min_output_amount < rust_decimal::Decimal::ZERO {
            return Err(LiquidityError::InvalidAmount("Minimum output amount cannot be negative".to_string()));
        }
        
        // Check if assets are different
        if input_asset.asset_id == output_asset_id {
            return Err(LiquidityError::InvalidSwap("Cannot swap asset for itself".to_string()));
        }
        
        Ok(())
    }
    
    fn is_valid_bonding_curve(&self, curve: &BondingCurve) -> bool {
        match curve {
            BondingCurve::ConstantProduct { .. } => true,
            BondingCurve::ConstantSum { .. } => true,
            BondingCurve::ConstantMean { .. } => true,
            BondingCurve::Stable { .. } => true,
            BondingCurve::Custom { .. } => true, // TODO: Add custom curve validation
        }
    }
    
    async fn apply_mev_protection(
        &self,
        pool_id: PoolId,
        input_asset: AssetAmount,
        output_asset_id: AssetId,
        min_output_amount: rust_decimal::Decimal,
    ) -> Result<SwapParameters> {
        // Apply MEV protection strategies
        let protected_slippage = self.calculate_mev_protected_slippage(pool_id, &input_asset).await?;
        
        Ok(SwapParameters {
            pool_id,
            input_asset,
            output_asset_id,
            min_output_amount,
            max_slippage: protected_slippage,
        })
    }
    
    async fn calculate_mev_protected_slippage(
        &self,
        _pool_id: PoolId,
        _input_asset: &AssetAmount,
    ) -> Result<rust_decimal::Decimal> {
        // TODO: Implement MEV protection slippage calculation
        Ok(self.config.default_max_slippage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_liquidity_management_engine_creation() {
        let config = LiquidityManagementConfig::default();
        let engine = LiquidityManagementEngine::new(config);
        
        assert_eq!(engine.active_pools.len(), 0);
        assert_eq!(engine.pending_operations.len(), 0);
    }
    
    #[tokio::test]
    async fn test_pool_configuration_validation() {
        let config = LiquidityManagementConfig::default();
        let engine = LiquidityManagementEngine::new(config);
        
        // Test invalid pool configuration (less than 2 assets)
        let invalid_config = PoolConfiguration {
            pool_id: PoolId::new(),
            assets: vec![AssetId::new()],
            bonding_curve: BondingCurve::ConstantProduct { k: rust_decimal::Decimal::ONE },
            fee_rate: rust_decimal::Decimal::from(3) / rust_decimal::Decimal::from(1000), // 0.3%
            is_cross_chain: false,
            supported_chains: vec![],
            risk_parameters: RiskParameters::default(),
        };
        
        let result = engine.validate_pool_configuration(&invalid_config).await;
        assert!(result.is_err());
    }
}
