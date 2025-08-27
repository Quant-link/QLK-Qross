use serde::{Deserialize, Serialize};
use qross_consensus::{ValidatorId, ValidatorSet, BlockProductionSchedule};
use uuid::Uuid;

/// Messages exchanged between validator nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorMessage {
    /// Announce validator presence and stake
    ValidatorAnnouncement {
        validator_id: ValidatorId,
        stake: rust_decimal::Decimal,
        block_height: u64,
        signature: Vec<u8>,
    },
    /// Request current validator set
    ValidatorSetRequest {
        request_id: Uuid,
        from_validator: ValidatorId,
    },
    /// Response with current validator set
    ValidatorSetResponse {
        request_id: Uuid,
        validators: Vec<ValidatorInfo>,
        total_stake: rust_decimal::Decimal,
    },
    /// Announce block production assignment
    BlockProductionAnnouncement {
        schedule: BlockProductionSchedule,
        from_validator: ValidatorId,
        signature: Vec<u8>,
    },
    /// Report missed block production
    MissedBlockReport {
        validator_id: ValidatorId,
        block_height: u64,
        reporter: ValidatorId,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Heartbeat to maintain connection
    Heartbeat {
        validator_id: ValidatorId,
        timestamp: chrono::DateTime<chrono::Utc>,
        block_height: u64,
    },
}

/// Validator information for network sharing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorInfo {
    pub validator_id: ValidatorId,
    pub stake: rust_decimal::Decimal,
    pub delegated_stake: rust_decimal::Decimal,
    pub status: ValidatorStatus,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// Validator status for network communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidatorStatus {
    Active,
    Inactive,
    Slashed { until_block: u64 },
}

impl ValidatorMessage {
    /// Get the validator ID associated with this message
    pub fn validator_id(&self) -> ValidatorId {
        match self {
            ValidatorMessage::ValidatorAnnouncement { validator_id, .. } => *validator_id,
            ValidatorMessage::ValidatorSetRequest { from_validator, .. } => *from_validator,
            ValidatorMessage::BlockProductionAnnouncement { from_validator, .. } => *from_validator,
            ValidatorMessage::MissedBlockReport { reporter, .. } => *reporter,
            ValidatorMessage::Heartbeat { validator_id, .. } => *validator_id,
            ValidatorMessage::ValidatorSetResponse { .. } => {
                // Response messages don't have a single validator ID
                ValidatorId(Uuid::nil())
            }
        }
    }

    /// Check if message requires signature verification
    pub fn requires_signature(&self) -> bool {
        matches!(
            self,
            ValidatorMessage::ValidatorAnnouncement { .. }
                | ValidatorMessage::BlockProductionAnnouncement { .. }
        )
    }

    /// Get message signature if present
    pub fn signature(&self) -> Option<&[u8]> {
        match self {
            ValidatorMessage::ValidatorAnnouncement { signature, .. } => Some(signature),
            ValidatorMessage::BlockProductionAnnouncement { signature, .. } => Some(signature),
            _ => None,
        }
    }

    /// Create a heartbeat message
    pub fn heartbeat(validator_id: ValidatorId, block_height: u64) -> Self {
        ValidatorMessage::Heartbeat {
            validator_id,
            timestamp: chrono::Utc::now(),
            block_height,
        }
    }

    /// Create a validator set request
    pub fn validator_set_request(from_validator: ValidatorId) -> Self {
        ValidatorMessage::ValidatorSetRequest {
            request_id: Uuid::new_v4(),
            from_validator,
        }
    }
}

impl From<qross_consensus::ValidatorStatus> for ValidatorStatus {
    fn from(status: qross_consensus::ValidatorStatus) -> Self {
        match status {
            qross_consensus::ValidatorStatus::Active => ValidatorStatus::Active,
            qross_consensus::ValidatorStatus::Inactive => ValidatorStatus::Inactive,
            qross_consensus::ValidatorStatus::Slashed { until_block } => {
                ValidatorStatus::Slashed { until_block }
            }
        }
    }
}

impl From<ValidatorStatus> for qross_consensus::ValidatorStatus {
    fn from(status: ValidatorStatus) -> Self {
        match status {
            ValidatorStatus::Active => qross_consensus::ValidatorStatus::Active,
            ValidatorStatus::Inactive => qross_consensus::ValidatorStatus::Inactive,
            ValidatorStatus::Slashed { until_block } => {
                qross_consensus::ValidatorStatus::Slashed { until_block }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_message_creation() {
        let validator_id = ValidatorId(Uuid::new_v4());
        
        let heartbeat = ValidatorMessage::heartbeat(validator_id, 100);
        assert_eq!(heartbeat.validator_id(), validator_id);
        assert!(!heartbeat.requires_signature());
        
        let request = ValidatorMessage::validator_set_request(validator_id);
        assert_eq!(request.validator_id(), validator_id);
        assert!(!request.requires_signature());
    }

    #[test]
    fn test_validator_status_conversion() {
        let active = qross_consensus::ValidatorStatus::Active;
        let network_active: ValidatorStatus = active.into();
        let back_to_consensus: qross_consensus::ValidatorStatus = network_active.into();
        
        assert!(matches!(back_to_consensus, qross_consensus::ValidatorStatus::Active));
    }

    #[test]
    fn test_message_signature_requirements() {
        let validator_id = ValidatorId(Uuid::new_v4());
        
        let announcement = ValidatorMessage::ValidatorAnnouncement {
            validator_id,
            stake: rust_decimal::Decimal::from(1000),
            block_height: 100,
            signature: vec![1, 2, 3],
        };
        
        assert!(announcement.requires_signature());
        assert_eq!(announcement.signature(), Some([1, 2, 3].as_slice()));
    }
}
