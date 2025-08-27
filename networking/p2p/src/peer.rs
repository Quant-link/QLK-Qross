use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use qross_consensus::ValidatorId;
use uuid::Uuid;

/// Peer information for network nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub validator_id: Option<ValidatorId>,
    pub address: SocketAddr,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub connection_status: ConnectionStatus,
    pub protocol_version: String,
}

/// Unique identifier for network peers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerId(pub Uuid);

impl PeerId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Connection status with peers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Failed { reason: String },
}

/// Peer registry for managing network connections
pub struct PeerRegistry {
    peers: HashMap<PeerId, PeerInfo>,
    validator_to_peer: HashMap<ValidatorId, PeerId>,
    address_to_peer: HashMap<SocketAddr, PeerId>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            validator_to_peer: HashMap::new(),
            address_to_peer: HashMap::new(),
        }
    }

    /// Add a new peer to the registry
    pub fn add_peer(&mut self, peer_info: PeerInfo) -> Result<(), String> {
        let peer_id = peer_info.peer_id;
        
        // Check for duplicate address
        if self.address_to_peer.contains_key(&peer_info.address) {
            return Err("Address already registered".to_string());
        }

        // Update validator mapping if present
        if let Some(validator_id) = peer_info.validator_id {
            self.validator_to_peer.insert(validator_id, peer_id);
        }

        // Update address mapping
        self.address_to_peer.insert(peer_info.address, peer_id);
        
        // Store peer info
        self.peers.insert(peer_id, peer_info);
        
        Ok(())
    }

    /// Remove a peer from the registry
    pub fn remove_peer(&mut self, peer_id: &PeerId) -> Option<PeerInfo> {
        if let Some(peer_info) = self.peers.remove(peer_id) {
            // Clean up mappings
            if let Some(validator_id) = peer_info.validator_id {
                self.validator_to_peer.remove(&validator_id);
            }
            self.address_to_peer.remove(&peer_info.address);
            
            Some(peer_info)
        } else {
            None
        }
    }

    /// Get peer information by peer ID
    pub fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        self.peers.get(peer_id)
    }

    /// Get peer by validator ID
    pub fn get_peer_by_validator(&self, validator_id: &ValidatorId) -> Option<&PeerInfo> {
        self.validator_to_peer
            .get(validator_id)
            .and_then(|peer_id| self.peers.get(peer_id))
    }

    /// Get peer by address
    pub fn get_peer_by_address(&self, address: &SocketAddr) -> Option<&PeerInfo> {
        self.address_to_peer
            .get(address)
            .and_then(|peer_id| self.peers.get(peer_id))
    }

    /// Update peer connection status
    pub fn update_connection_status(&mut self, peer_id: &PeerId, status: ConnectionStatus) -> Result<(), String> {
        if let Some(peer_info) = self.peers.get_mut(peer_id) {
            peer_info.connection_status = status;
            peer_info.last_seen = chrono::Utc::now();
            Ok(())
        } else {
            Err("Peer not found".to_string())
        }
    }

    /// Update peer last seen time and validator ID
    pub fn update_peer_info(&mut self, peer_id: &PeerId, validator_id: Option<ValidatorId>) -> Result<(), String> {
        if let Some(peer_info) = self.peers.get_mut(peer_id) {
            peer_info.last_seen = chrono::Utc::now();
            if let Some(vid) = validator_id {
                peer_info.validator_id = Some(vid);
            }
            Ok(())
        } else {
            Err("Peer not found".to_string())
        }
    }

    /// Get all connected peers
    pub fn get_connected_peers(&self) -> Vec<&PeerInfo> {
        self.peers
            .values()
            .filter(|peer| matches!(peer.connection_status, ConnectionStatus::Connected))
            .collect()
    }

    /// Get all validator peers
    pub fn get_validator_peers(&self) -> Vec<&PeerInfo> {
        self.peers
            .values()
            .filter(|peer| peer.validator_id.is_some())
            .collect()
    }

    /// Get peer count
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get connected peer count
    pub fn connected_peer_count(&self) -> usize {
        self.peers
            .values()
            .filter(|peer| matches!(peer.connection_status, ConnectionStatus::Connected))
            .count()
    }

    /// Clean up stale peers (not seen for more than threshold)
    pub fn cleanup_stale_peers(&mut self, threshold: chrono::Duration) {
        let now = chrono::Utc::now();
        let stale_peers: Vec<PeerId> = self
            .peers
            .iter()
            .filter(|(_, peer)| {
                now.signed_duration_since(peer.last_seen) > threshold
                    && !matches!(peer.connection_status, ConnectionStatus::Connected)
            })
            .map(|(peer_id, _)| *peer_id)
            .collect();

        for peer_id in stale_peers {
            self.remove_peer(&peer_id);
        }
    }
}

impl Default for PeerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn create_test_peer(validator_id: Option<ValidatorId>) -> PeerInfo {
        PeerInfo {
            peer_id: PeerId::new(),
            validator_id,
            address: SocketAddr::from_str("127.0.0.1:8080").unwrap(),
            last_seen: chrono::Utc::now(),
            connection_status: ConnectionStatus::Connected,
            protocol_version: "1.0.0".to_string(),
        }
    }

    #[test]
    fn test_peer_registry_add_remove() {
        let mut registry = PeerRegistry::new();
        let peer = create_test_peer(None);
        let peer_id = peer.peer_id;

        // Add peer
        registry.add_peer(peer).unwrap();
        assert_eq!(registry.peer_count(), 1);

        // Get peer
        assert!(registry.get_peer(&peer_id).is_some());

        // Remove peer
        let removed = registry.remove_peer(&peer_id);
        assert!(removed.is_some());
        assert_eq!(registry.peer_count(), 0);
    }

    #[test]
    fn test_validator_peer_mapping() {
        let mut registry = PeerRegistry::new();
        let validator_id = ValidatorId(Uuid::new_v4());
        let peer = create_test_peer(Some(validator_id));

        registry.add_peer(peer).unwrap();

        // Should be able to find peer by validator ID
        let found_peer = registry.get_peer_by_validator(&validator_id);
        assert!(found_peer.is_some());
        assert_eq!(found_peer.unwrap().validator_id, Some(validator_id));
    }

    #[test]
    fn test_connection_status_update() {
        let mut registry = PeerRegistry::new();
        let peer = create_test_peer(None);
        let peer_id = peer.peer_id;

        registry.add_peer(peer).unwrap();

        // Update status
        registry
            .update_connection_status(&peer_id, ConnectionStatus::Disconnected)
            .unwrap();

        let updated_peer = registry.get_peer(&peer_id).unwrap();
        assert!(matches!(updated_peer.connection_status, ConnectionStatus::Disconnected));
    }

    #[test]
    fn test_connected_peers_filter() {
        let mut registry = PeerRegistry::new();
        
        let mut peer1 = create_test_peer(None);
        peer1.connection_status = ConnectionStatus::Connected;
        
        let mut peer2 = create_test_peer(None);
        peer2.address = SocketAddr::from_str("127.0.0.1:8081").unwrap();
        peer2.connection_status = ConnectionStatus::Disconnected;

        registry.add_peer(peer1).unwrap();
        registry.add_peer(peer2).unwrap();

        let connected = registry.get_connected_peers();
        assert_eq!(connected.len(), 1);
        assert!(matches!(connected[0].connection_status, ConnectionStatus::Connected));
    }
}
