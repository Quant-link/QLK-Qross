//! Error types for liquidity management system

use thiserror::Error;
use crate::types::*;

/// Result type for liquidity management operations
pub type Result<T> = std::result::Result<T, LiquidityError>;

/// Liquidity management error types
#[derive(Error, Debug)]
pub enum LiquidityError {
    /// Pool not found
    #[error("Pool not found: {0}")]
    PoolNotFound(PoolId),
    
    /// Asset not found
    #[error("Asset not found: {0}")]
    AssetNotFound(AssetId),
    
    /// Liquidity provider not found
    #[error("Liquidity provider not found: {0}")]
    ProviderNotFound(LiquidityProvider),
    
    /// Insufficient liquidity
    #[error("Insufficient liquidity")]
    InsufficientLiquidity,
    
    /// Invalid amount
    #[error("Invalid amount: {0}")]
    InvalidAmount(String),
    
    /// Invalid swap parameters
    #[error("Invalid swap: {0}")]
    InvalidSwap(String),
    
    /// Slippage exceeded
    #[error("Slippage exceeded maximum tolerance")]
    SlippageExceeded,
    
    /// Price impact too high
    #[error("Price impact exceeds maximum threshold")]
    PriceImpactTooHigh,
    
    /// Invalid pool configuration
    #[error("Invalid pool configuration: {0}")]
    InvalidPoolConfiguration(String),
    
    /// Unsupported bonding curve
    #[error("Unsupported bonding curve: {0:?}")]
    UnsupportedCurve(BondingCurveType),
    
    /// Calculation error
    #[error("Calculation error: {0}")]
    CalculationError(String),
    
    /// Cross-chain operation error
    #[error("Cross-chain operation failed: {0}")]
    CrossChainError(String),
    
    /// Bridge operation error
    #[error("Bridge operation failed: {0}")]
    BridgeError(String),
    
    /// Oracle error
    #[error("Oracle error: {0}")]
    OracleError(String),
    
    /// Proof generation error
    #[error("Proof generation failed: {0}")]
    ProofError(String),
    
    /// State synchronization error
    #[error("State synchronization failed: {0}")]
    StateSyncError(String),
    
    /// Risk management error
    #[error("Risk management error: {0}")]
    RiskError(String),
    
    /// Arbitrage detection error
    #[error("Arbitrage detection error: {0}")]
    ArbitrageError(String),
    
    /// Yield optimization error
    #[error("Yield optimization error: {0}")]
    YieldError(String),
    
    /// MEV protection error
    #[error("MEV protection error: {0}")]
    MEVError(String),
    
    /// Fee calculation error
    #[error("Fee calculation error: {0}")]
    FeeError(String),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Timeout error
    #[error("Operation timed out")]
    Timeout,
    
    /// Operation cancelled
    #[error("Operation was cancelled")]
    Cancelled,
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// External dependency error
    #[error("External dependency error: {0}")]
    ExternalError(String),
}

impl From<qross_zk_verification::error::ZKError> for LiquidityError {
    fn from(err: qross_zk_verification::error::ZKError) -> Self {
        LiquidityError::ProofError(err.to_string())
    }
}

impl From<qross_consensus::error::ConsensusError> for LiquidityError {
    fn from(err: qross_consensus::error::ConsensusError) -> Self {
        LiquidityError::StateSyncError(err.to_string())
    }
}

impl From<qross_p2p_network::error::NetworkError> for LiquidityError {
    fn from(err: qross_p2p_network::error::NetworkError) -> Self {
        LiquidityError::NetworkError(err.to_string())
    }
}

impl From<serde_json::Error> for LiquidityError {
    fn from(err: serde_json::Error) -> Self {
        LiquidityError::SerializationError(err.to_string())
    }
}

impl From<bincode::Error> for LiquidityError {
    fn from(err: bincode::Error) -> Self {
        LiquidityError::SerializationError(err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for LiquidityError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        LiquidityError::Timeout
    }
}

impl From<std::num::ParseFloatError> for LiquidityError {
    fn from(err: std::num::ParseFloatError) -> Self {
        LiquidityError::CalculationError(err.to_string())
    }
}

impl From<rust_decimal::Error> for LiquidityError {
    fn from(err: rust_decimal::Error) -> Self {
        LiquidityError::CalculationError(err.to_string())
    }
}

/// Pool operation errors
#[derive(Error, Debug)]
pub enum PoolError {
    #[error("Pool already exists: {0}")]
    PoolAlreadyExists(PoolId),
    
    #[error("Pool is paused: {0}")]
    PoolPaused(PoolId),
    
    #[error("Pool has insufficient reserves")]
    InsufficientReserves,
    
    #[error("Pool configuration is invalid: {0}")]
    InvalidConfiguration(String),
    
    #[error("Pool operation not supported: {0}")]
    OperationNotSupported(String),
}

/// Swap operation errors
#[derive(Error, Debug)]
pub enum SwapError {
    #[error("Invalid swap path")]
    InvalidPath,
    
    #[error("Swap amount too small")]
    AmountTooSmall,
    
    #[error("Swap amount too large")]
    AmountTooLarge,
    
    #[error("Deadline exceeded")]
    DeadlineExceeded,
    
    #[error("Insufficient output amount")]
    InsufficientOutput,
    
    #[error("Swap temporarily disabled")]
    SwapDisabled,
}

/// Liquidity operation errors
#[derive(Error, Debug)]
pub enum LiquidityOperationError {
    #[error("Minimum liquidity not met")]
    MinimumLiquidityNotMet,
    
    #[error("Maximum liquidity exceeded")]
    MaximumLiquidityExceeded,
    
    #[error("Liquidity position not found")]
    PositionNotFound,
    
    #[error("Liquidity locked")]
    LiquidityLocked,
    
    #[error("Withdrawal amount exceeds position")]
    WithdrawalExceedsPosition,
}

/// Cross-chain operation errors
#[derive(Error, Debug)]
pub enum CrossChainError {
    #[error("Chain not supported: {0}")]
    ChainNotSupported(ChainId),
    
    #[error("Bridge not available")]
    BridgeNotAvailable,
    
    #[error("Insufficient bridge liquidity")]
    InsufficientBridgeLiquidity,
    
    #[error("Cross-chain transfer failed: {0}")]
    TransferFailed(String),
    
    #[error("State synchronization failed")]
    StateSyncFailed,
    
    #[error("Finality not reached")]
    FinalityNotReached,
}

/// Risk management errors
#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Risk threshold exceeded")]
    RiskThresholdExceeded,
    
    #[error("Volatility too high")]
    VolatilityTooHigh,
    
    #[error("Correlation risk detected")]
    CorrelationRisk,
    
    #[error("Impermanent loss risk too high")]
    ImpermanentLossRisk,
    
    #[error("Liquidity risk detected")]
    LiquidityRisk,
}

/// Oracle errors
#[derive(Error, Debug)]
pub enum OracleError {
    #[error("Price feed not available")]
    PriceFeedNotAvailable,
    
    #[error("Price data stale")]
    PriceDataStale,
    
    #[error("Price deviation too high")]
    PriceDeviationTooHigh,
    
    #[error("Oracle consensus failed")]
    ConsensusFailure,
    
    #[error("Oracle data invalid")]
    InvalidData,
}

/// Arbitrage errors
#[derive(Error, Debug)]
pub enum ArbitrageError {
    #[error("No arbitrage opportunity found")]
    NoOpportunityFound,
    
    #[error("Arbitrage opportunity expired")]
    OpportunityExpired,
    
    #[error("Insufficient capital for arbitrage")]
    InsufficientCapital,
    
    #[error("Arbitrage execution failed")]
    ExecutionFailed,
    
    #[error("MEV attack detected")]
    MEVAttackDetected,
}

/// Yield optimization errors
#[derive(Error, Debug)]
pub enum YieldError {
    #[error("No yield optimization available")]
    NoOptimizationAvailable,
    
    #[error("Yield calculation failed")]
    CalculationFailed,
    
    #[error("Rebalancing failed")]
    RebalancingFailed,
    
    #[error("Compound operation failed")]
    CompoundFailed,
    
    #[error("Gas optimization failed")]
    GasOptimizationFailed,
}

/// Fee calculation errors
#[derive(Error, Debug)]
pub enum FeeError {
    #[error("Fee calculation failed")]
    CalculationFailed,
    
    #[error("Fee structure invalid")]
    InvalidStructure,
    
    #[error("Fee tier not found")]
    TierNotFound,
    
    #[error("Dynamic fee calculation failed")]
    DynamicCalculationFailed,
    
    #[error("Fee distribution failed")]
    DistributionFailed,
}

/// Proof generation errors
#[derive(Error, Debug)]
pub enum ProofError {
    #[error("Proof generation failed: {0}")]
    GenerationFailed(String),
    
    #[error("Proof verification failed")]
    VerificationFailed,
    
    #[error("Proof batch coordination failed")]
    BatchCoordinationFailed,
    
    #[error("State transition proof failed")]
    StateTransitionFailed,
    
    #[error("Proof aggregation failed")]
    AggregationFailed,
}

/// State synchronization errors
#[derive(Error, Debug)]
pub enum StateSyncError {
    #[error("State commitment mismatch")]
    CommitmentMismatch,
    
    #[error("Cross-chain coordination failed")]
    CoordinationFailed,
    
    #[error("Finality tracking failed")]
    FinalityTrackingFailed,
    
    #[error("State proof invalid")]
    InvalidStateProof,
    
    #[error("Synchronization timeout")]
    SyncTimeout,
}

impl From<PoolError> for LiquidityError {
    fn from(err: PoolError) -> Self {
        LiquidityError::Internal(err.to_string())
    }
}

impl From<SwapError> for LiquidityError {
    fn from(err: SwapError) -> Self {
        LiquidityError::InvalidSwap(err.to_string())
    }
}

impl From<LiquidityOperationError> for LiquidityError {
    fn from(err: LiquidityOperationError) -> Self {
        LiquidityError::Internal(err.to_string())
    }
}

impl From<CrossChainError> for LiquidityError {
    fn from(err: CrossChainError) -> Self {
        LiquidityError::CrossChainError(err.to_string())
    }
}

impl From<RiskError> for LiquidityError {
    fn from(err: RiskError) -> Self {
        LiquidityError::RiskError(err.to_string())
    }
}

impl From<OracleError> for LiquidityError {
    fn from(err: OracleError) -> Self {
        LiquidityError::OracleError(err.to_string())
    }
}

impl From<ArbitrageError> for LiquidityError {
    fn from(err: ArbitrageError) -> Self {
        LiquidityError::ArbitrageError(err.to_string())
    }
}

impl From<YieldError> for LiquidityError {
    fn from(err: YieldError) -> Self {
        LiquidityError::YieldError(err.to_string())
    }
}

impl From<FeeError> for LiquidityError {
    fn from(err: FeeError) -> Self {
        LiquidityError::FeeError(err.to_string())
    }
}

impl From<ProofError> for LiquidityError {
    fn from(err: ProofError) -> Self {
        LiquidityError::ProofError(err.to_string())
    }
}

impl From<StateSyncError> for LiquidityError {
    fn from(err: StateSyncError) -> Self {
        LiquidityError::StateSyncError(err.to_string())
    }
}
