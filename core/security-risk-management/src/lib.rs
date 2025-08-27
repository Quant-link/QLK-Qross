//! Security and Risk Management Layer for Quantlink Qross
//! 
//! This module provides comprehensive security mechanisms including:
//! - Threshold signature schemes with BLS aggregation
//! - Multi-signature governance protocols
//! - Emergency pause mechanisms
//! - Formal verification frameworks
//! - Security monitoring and threat detection

pub mod threshold_signatures;
pub mod multi_sig_governance;
pub mod emergency_mechanisms;
pub mod formal_verification;
pub mod security_monitoring;
pub mod types;
pub mod error;

// Re-export main types
pub use threshold_signatures::*;
pub use multi_sig_governance::*;
pub use emergency_mechanisms::*;
pub use formal_verification::*;
pub use security_monitoring::*;
pub use types::*;
pub use error::*;

/// Security and Risk Management System
pub struct SecurityRiskManagementSystem {
    threshold_signature_manager: ThresholdSignatureManager,
    governance_system: MultiSigGovernanceSystem,
    emergency_coordinator: EmergencyCoordinator,
    verification_engine: FormalVerificationEngine,
    security_monitor: SecurityMonitor,
}

impl SecurityRiskManagementSystem {
    /// Create a new security and risk management system
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            threshold_signature_manager: ThresholdSignatureManager::new(config.threshold_config),
            governance_system: MultiSigGovernanceSystem::new(config.governance_config),
            emergency_coordinator: EmergencyCoordinator::new(config.emergency_config),
            verification_engine: FormalVerificationEngine::new(config.verification_config),
            security_monitor: SecurityMonitor::new(config.monitoring_config),
        }
    }
    
    /// Start the security and risk management system
    pub async fn start(&mut self) -> Result<()> {
        // Start all subsystems
        self.threshold_signature_manager.start().await?;
        self.governance_system.start().await?;
        self.emergency_coordinator.start().await?;
        self.verification_engine.start().await?;
        self.security_monitor.start().await?;
        
        tracing::info!("Security and Risk Management System started");
        
        Ok(())
    }
    
    /// Stop the security and risk management system
    pub async fn stop(&mut self) -> Result<()> {
        // Stop all subsystems in reverse order
        self.security_monitor.stop().await?;
        self.verification_engine.stop().await?;
        self.emergency_coordinator.stop().await?;
        self.governance_system.stop().await?;
        self.threshold_signature_manager.stop().await?;
        
        tracing::info!("Security and Risk Management System stopped");
        
        Ok(())
    }
    
    /// Get system status
    pub async fn get_system_status(&self) -> SecuritySystemStatus {
        SecuritySystemStatus {
            threshold_signatures_active: self.threshold_signature_manager.is_active(),
            governance_active: self.governance_system.is_active(),
            emergency_systems_active: self.emergency_coordinator.is_active(),
            verification_active: self.verification_engine.is_active(),
            monitoring_active: self.security_monitor.is_active(),
            last_updated: chrono::Utc::now(),
        }
    }
}

/// Security system status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecuritySystemStatus {
    pub threshold_signatures_active: bool,
    pub governance_active: bool,
    pub emergency_systems_active: bool,
    pub verification_active: bool,
    pub monitoring_active: bool,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_security_system_initialization() {
        let config = SecurityConfig::default();
        let mut system = SecurityRiskManagementSystem::new(config);
        
        assert!(system.start().await.is_ok());
        
        let status = system.get_system_status().await;
        assert!(status.threshold_signatures_active);
        assert!(status.governance_active);
        assert!(status.emergency_systems_active);
        assert!(status.verification_active);
        assert!(status.monitoring_active);
        
        assert!(system.stop().await.is_ok());
    }
}
