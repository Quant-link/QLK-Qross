//! Performance analysis engine for validators

use crate::{types::*, error::*};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Performance analysis engine
pub struct PerformanceAnalyzer {
    config: PerformanceConfig,
    performance_cache: HashMap<ValidatorId, PerformanceMetrics>,
    performance_data: HashMap<ValidatorId, Vec<PerformanceData>>,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config,
            performance_cache: HashMap::new(),
            performance_data: HashMap::new(),
        }
    }
    
    /// Analyze validator performance from historical records
    pub async fn analyze_performance(
        &mut self,
        performance_history: &[PerformanceRecord],
    ) -> Result<PerformanceMetrics> {
        if performance_history.is_empty() {
            return Ok(PerformanceMetrics {
                overall_score: 0.5,
                uptime_percentage: 0.0,
                block_production_rate: 0.0,
                attestation_rate: 0.0,
                response_time_ms: 0.0,
                missed_blocks: 0,
                missed_attestations: 0,
                last_calculated: Utc::now(),
            });
        }
        
        // Calculate uptime percentage
        let total_uptime: u64 = performance_history.iter().map(|r| r.uptime_seconds).sum();
        let total_time: u64 = performance_history.iter().map(|r| r.total_seconds).sum();
        let uptime_percentage = if total_time > 0 {
            (total_uptime as f64 / total_time as f64) * 100.0
        } else {
            0.0
        };
        
        // Calculate block production rate
        let total_produced: u64 = performance_history.iter().map(|r| r.blocks_produced).sum();
        let total_expected: u64 = performance_history.iter().map(|r| r.blocks_expected).sum();
        let block_production_rate = if total_expected > 0 {
            (total_produced as f64 / total_expected as f64) * 100.0
        } else {
            0.0
        };
        
        // Calculate attestation rate
        let attestations_made: u64 = performance_history.iter().map(|r| r.attestations_made).sum();
        let attestations_expected: u64 = performance_history.iter().map(|r| r.attestations_expected).sum();
        let attestation_rate = if attestations_expected > 0 {
            (attestations_made as f64 / attestations_expected as f64) * 100.0
        } else {
            0.0
        };
        
        // Calculate average response time
        let response_times: Vec<f64> = performance_history.iter()
            .map(|r| r.average_response_time)
            .filter(|&rt| rt > 0.0)
            .collect();
        let response_time_ms = if !response_times.is_empty() {
            response_times.iter().sum::<f64>() / response_times.len() as f64
        } else {
            0.0
        };
        
        // Calculate missed counts
        let missed_blocks = total_expected - total_produced;
        let missed_attestations = attestations_expected - attestations_made;
        
        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            uptime_percentage,
            block_production_rate,
            attestation_rate,
            response_time_ms,
        );
        
        Ok(PerformanceMetrics {
            overall_score,
            uptime_percentage,
            block_production_rate,
            attestation_rate,
            response_time_ms,
            missed_blocks,
            missed_attestations,
            last_calculated: Utc::now(),
        })
    }
    
    /// Calculate overall performance score
    fn calculate_overall_score(
        &self,
        uptime_percentage: f64,
        block_production_rate: f64,
        attestation_rate: f64,
        response_time_ms: f64,
    ) -> f64 {
        // Normalize uptime (0-1 scale)
        let uptime_score = (uptime_percentage / 100.0).min(1.0);
        
        // Normalize block production (0-1 scale)
        let block_score = (block_production_rate / 100.0).min(1.0);
        
        // Normalize attestation rate (0-1 scale)
        let attestation_score = (attestation_rate / 100.0).min(1.0);
        
        // Normalize response time (inverse relationship)
        let response_score = if response_time_ms > 0.0 {
            (self.config.max_response_time / (response_time_ms + 100.0)).min(1.0)
        } else {
            1.0
        };
        
        // Calculate weighted score
        (uptime_score * self.config.uptime_weight) +
        (block_score * self.config.block_production_weight) +
        (attestation_score * self.config.attestation_weight) +
        (response_score * self.config.response_time_weight)
    }
    
    /// Update validator performance data
    pub async fn update_performance(
        &mut self,
        validator_id: ValidatorId,
        data: PerformanceData,
    ) -> Result<()> {
        self.performance_data.entry(validator_id).or_insert_with(Vec::new).push(data);
        Ok(())
    }
    
    /// Get performance statistics
    pub fn get_performance_statistics(&self) -> PerformanceStatistics {
        let total_validators = self.performance_cache.len();
        
        if total_validators == 0 {
            return PerformanceStatistics {
                total_validators: 0,
                average_uptime: 0.0,
                average_block_production: 0.0,
                average_attestation_rate: 0.0,
                average_response_time: 0.0,
            };
        }
        
        let metrics: Vec<&PerformanceMetrics> = self.performance_cache.values().collect();
        
        let average_uptime = metrics.iter().map(|m| m.uptime_percentage).sum::<f64>() / total_validators as f64;
        let average_block_production = metrics.iter().map(|m| m.block_production_rate).sum::<f64>() / total_validators as f64;
        let average_attestation_rate = metrics.iter().map(|m| m.attestation_rate).sum::<f64>() / total_validators as f64;
        let average_response_time = metrics.iter().map(|m| m.response_time_ms).sum::<f64>() / total_validators as f64;
        
        PerformanceStatistics {
            total_validators,
            average_uptime,
            average_block_production,
            average_attestation_rate,
            average_response_time,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub total_validators: usize,
    pub average_uptime: f64,
    pub average_block_production: f64,
    pub average_attestation_rate: f64,
    pub average_response_time: f64,
}
