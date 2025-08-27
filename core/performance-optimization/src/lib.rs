//! Performance Optimization Engine for Quantlink Qross
//! 
//! Layer 6: Performance Optimization Engine provides enterprise-grade performance optimization
//! with adaptive fee modeling, batch processing optimization, and comprehensive monitoring.

pub mod fee_optimization;
pub mod batch_processing;
pub mod distributed_cache;
pub mod observability;
pub mod performance_monitoring;
pub mod adaptive_algorithms;
pub mod cost_modeling;
pub mod priority_management;
pub mod gas_prediction;
pub mod optimization_metrics;
pub mod types;
pub mod error;

// Re-export main components
pub use fee_optimization::*;
pub use batch_processing::*;
pub use distributed_cache::*;
pub use observability::*;
pub use performance_monitoring::*;
pub use adaptive_algorithms::*;
pub use cost_modeling::*;
pub use priority_management::*;
pub use gas_prediction::*;
pub use optimization_metrics::*;
pub use types::*;
pub use error::*;

/// Performance optimization engine for enterprise-grade efficiency
pub struct PerformanceOptimizationEngine {
    fee_optimizer: FeeOptimizationEngine,
    batch_processor: BatchProcessingEngine,
    performance_monitor: PerformanceMonitor,
    adaptive_algorithms: AdaptiveAlgorithmEngine,
    cost_modeler: CostModelingEngine,
    priority_manager: PriorityManagementEngine,
    gas_predictor: GasPredictionEngine,
    metrics_collector: OptimizationMetricsCollector,
}

impl PerformanceOptimizationEngine {
    /// Create new performance optimization engine
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            fee_optimizer: FeeOptimizationEngine::new(config.fee_config),
            batch_processor: BatchProcessingEngine::new(config.batch_config),
            performance_monitor: PerformanceMonitor::new(config.monitoring_config),
            adaptive_algorithms: AdaptiveAlgorithmEngine::new(config.algorithm_config),
            cost_modeler: CostModelingEngine::new(config.cost_config),
            priority_manager: PriorityManagementEngine::new(config.priority_config),
            gas_predictor: GasPredictionEngine::new(config.gas_config),
            metrics_collector: OptimizationMetricsCollector::new(config.metrics_config),
        }
    }
    
    /// Start the performance optimization engine
    pub async fn start(&mut self) -> Result<()> {
        self.fee_optimizer.start().await?;
        self.batch_processor.start().await?;
        self.performance_monitor.start().await?;
        self.adaptive_algorithms.start().await?;
        self.cost_modeler.start().await?;
        self.priority_manager.start().await?;
        self.gas_predictor.start().await?;
        self.metrics_collector.start().await?;
        
        tracing::info!("Performance optimization engine started");
        Ok(())
    }
    
    /// Stop the performance optimization engine
    pub async fn stop(&mut self) -> Result<()> {
        self.metrics_collector.stop().await?;
        self.gas_predictor.stop().await?;
        self.priority_manager.stop().await?;
        self.cost_modeler.stop().await?;
        self.adaptive_algorithms.stop().await?;
        self.performance_monitor.stop().await?;
        self.batch_processor.stop().await?;
        self.fee_optimizer.stop().await?;
        
        tracing::info!("Performance optimization engine stopped");
        Ok(())
    }
    
    /// Optimize transaction fees
    pub async fn optimize_fees(&mut self, transactions: Vec<Transaction>) -> Result<Vec<OptimizedTransaction>> {
        self.fee_optimizer.optimize_transaction_fees(transactions).await
    }
    
    /// Process batch optimization
    pub async fn optimize_batch(&mut self, batch: TransactionBatch) -> Result<OptimizedBatch> {
        self.batch_processor.optimize_batch(batch).await
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_monitor.get_current_metrics()
    }
    
    /// Get optimization recommendations
    pub async fn get_optimization_recommendations(&self) -> Result<Vec<OptimizationRecommendation>> {
        self.adaptive_algorithms.generate_recommendations().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_optimization_engine() {
        let config = OptimizationConfig::default();
        let mut engine = PerformanceOptimizationEngine::new(config);
        
        assert!(engine.start().await.is_ok());
        assert!(engine.stop().await.is_ok());
    }
}
