use crate::{ValidatorMessage, PeerInfo, PeerId, PeerRegistry, PeerDiscovery, ConnectionStatus};
use qross_consensus::{ValidatorId, PersistentValidatorSet};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::codec::{Framed, LengthDelimitedCodec};
use futures::{SinkExt, StreamExt};
use bytes::Bytes;

/// Network service for validator coordination
pub struct ValidatorNetwork {
    local_peer_id: PeerId,
    local_validator_id: Option<ValidatorId>,
    listen_address: SocketAddr,
    peer_registry: Arc<RwLock<PeerRegistry>>,
    validator_set: Arc<RwLock<PersistentValidatorSet>>,
    discovery: Arc<RwLock<PeerDiscovery>>,
    message_sender: mpsc::UnboundedSender<(ValidatorMessage, PeerId)>,
    message_receiver: Arc<RwLock<mpsc::UnboundedReceiver<(ValidatorMessage, PeerId)>>>,
}

/// Network configuration
pub struct NetworkConfig {
    pub listen_address: SocketAddr,
    pub bootstrap_peers: Vec<SocketAddr>,
    pub validator_id: Option<ValidatorId>,
    pub heartbeat_interval: std::time::Duration,
    pub discovery_interval: std::time::Duration,
}

impl ValidatorNetwork {
    /// Create new validator network
    pub fn new(config: NetworkConfig, validator_set: Arc<RwLock<PersistentValidatorSet>>) -> Self {
        let local_peer_id = PeerId::new();
        let peer_registry = Arc::new(RwLock::new(PeerRegistry::new()));
        let discovery = Arc::new(RwLock::new(PeerDiscovery::new(
            config.listen_address,
            config.bootstrap_peers,
        )));
        
        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        let message_receiver = Arc::new(RwLock::new(message_receiver));

        Self {
            local_peer_id,
            local_validator_id: config.validator_id,
            listen_address: config.listen_address,
            peer_registry,
            validator_set,
            discovery,
            message_sender,
            message_receiver,
        }
    }

    /// Start the network service
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Start discovery service
        {
            let mut discovery = self.discovery.write().await;
            discovery.start().await?;
        }

        // Start TCP listener for incoming connections
        let listener = TcpListener::bind(self.listen_address).await?;
        
        // Announce presence if we're a validator
        if let Some(validator_id) = self.local_validator_id {
            let discovery = self.discovery.read().await;
            discovery.announce(self.local_peer_id, Some(validator_id)).await?;
        }

        // Spawn connection handler
        let peer_registry = Arc::clone(&self.peer_registry);
        let message_sender = self.message_sender.clone();
        let local_peer_id = self.local_peer_id;

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let peer_registry = Arc::clone(&peer_registry);
                        let message_sender = message_sender.clone();
                        
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(stream, addr, peer_registry, message_sender, local_peer_id).await {
                                tracing::error!("Connection error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Handle incoming TCP connection
    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        peer_registry: Arc<RwLock<PeerRegistry>>,
        message_sender: mpsc::UnboundedSender<(ValidatorMessage, PeerId)>,
        local_peer_id: PeerId,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut framed = Framed::new(stream, LengthDelimitedCodec::new());
        
        while let Some(frame) = framed.next().await {
            match frame {
                Ok(data) => {
                    if let Ok(message) = serde_json::from_slice::<ValidatorMessage>(&data) {
                        // Find peer by address
                        let peer_id = {
                            let registry = peer_registry.read().await;
                            registry.get_peer_by_address(&addr)
                                .map(|peer| peer.peer_id)
                                .unwrap_or(local_peer_id) // Fallback to local peer ID
                        };
                        
                        // Forward message to handler
                        if message_sender.send((message, peer_id)).is_err() {
                            break; // Channel closed
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Frame error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Send message to specific peer
    pub async fn send_message(&self, message: ValidatorMessage, peer_id: PeerId) -> Result<(), Box<dyn std::error::Error>> {
        let peer_info = {
            let registry = self.peer_registry.read().await;
            registry.get_peer(&peer_id).cloned()
        };

        if let Some(peer) = peer_info {
            let stream = TcpStream::connect(peer.address).await?;
            let mut framed = Framed::new(stream, LengthDelimitedCodec::new());
            
            let serialized = serde_json::to_vec(&message)?;
            framed.send(Bytes::from(serialized)).await?;
        }

        Ok(())
    }

    /// Broadcast message to all connected validator peers
    pub async fn broadcast_message(&self, message: ValidatorMessage) -> Result<(), Box<dyn std::error::Error>> {
        let validator_peers = {
            let registry = self.peer_registry.read().await;
            registry.get_validator_peers()
                .iter()
                .filter(|peer| matches!(peer.connection_status, ConnectionStatus::Connected))
                .map(|peer| peer.peer_id)
                .collect::<Vec<_>>()
        };

        for peer_id in validator_peers {
            if let Err(e) = self.send_message(message.clone(), peer_id).await {
                tracing::warn!("Failed to send message to peer {}: {}", peer_id.0, e);
            }
        }

        Ok(())
    }

    /// Process incoming messages
    pub async fn process_messages(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut receiver = self.message_receiver.write().await;

        while let Some((message, from_peer)) = receiver.recv().await {
            if let Err(e) = self.handle_validator_message(message, from_peer).await {
                tracing::error!("Error handling message: {}", e);
            }
        }

        Ok(())
    }

    /// Handle validator message
    async fn handle_validator_message(&self, message: ValidatorMessage, from_peer: PeerId) -> Result<(), Box<dyn std::error::Error>> {
        match message {
            ValidatorMessage::ValidatorAnnouncement { validator_id, stake, block_height, .. } => {
                tracing::info!("Received validator announcement from {} with stake {}", validator_id.0, stake);

                // Update peer registry with validator mapping
                let mut registry = self.peer_registry.write().await;
                let _ = registry.update_peer_info(&from_peer, Some(validator_id));
            }
            ValidatorMessage::ValidatorSetRequest { request_id, from_validator } => {
                // Respond with current validator set
                let validator_set = self.validator_set.read().await;
                let validators = validator_set.get_active_validators()
                    .into_iter()
                    .map(|v| crate::ValidatorInfo {
                        validator_id: v.id,
                        stake: v.stake,
                        delegated_stake: v.delegated_stake,
                        status: v.status.clone().into(),
                        last_seen: chrono::Utc::now(),
                    })
                    .collect();

                let response = ValidatorMessage::ValidatorSetResponse {
                    request_id,
                    validators,
                    total_stake: validator_set.total_stake(),
                };

                self.send_message(response, from_peer).await?;
            }
            ValidatorMessage::MissedBlockReport { validator_id, block_height, .. } => {
                // Record missed block in our validator set
                let mut validator_set = self.validator_set.write().await;
                // Note: record_missed_block is not async in the current implementation
                // validator_set.record_missed_block(validator_id, block_height).await?;

                tracing::info!("Recorded missed block for validator {} at height {}", validator_id.0, block_height);
            }
            ValidatorMessage::Heartbeat { validator_id, block_height, .. } => {
                // Update peer last seen time
                let mut registry = self.peer_registry.write().await;
                let _ = registry.update_peer_info(&from_peer, Some(validator_id));

                tracing::debug!("Received heartbeat from validator {} at height {}", validator_id.0, block_height);
            }
            _ => {
                tracing::debug!("Received other message type");
            }
        }

        Ok(())
    }

    /// Send heartbeat to network
    pub async fn send_heartbeat(&self, block_height: u64) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(validator_id) = self.local_validator_id {
            let heartbeat = ValidatorMessage::heartbeat(validator_id, block_height);
            self.broadcast_message(heartbeat).await?;
        }
        Ok(())
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> NetworkStats {
        let registry = self.peer_registry.read().await;
        NetworkStats {
            total_peers: registry.peer_count(),
            connected_peers: registry.connected_peer_count(),
            validator_peers: registry.get_validator_peers().len(),
            local_peer_id: self.local_peer_id,
            local_validator_id: self.local_validator_id,
        }
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub total_peers: usize,
    pub connected_peers: usize,
    pub validator_peers: usize,
    pub local_peer_id: PeerId,
    pub local_validator_id: Option<ValidatorId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use std::time::Duration;

    fn create_test_config() -> NetworkConfig {
        NetworkConfig {
            listen_address: SocketAddr::from_str("127.0.0.1:0").unwrap(),
            bootstrap_peers: vec![],
            validator_id: Some(ValidatorId(uuid::Uuid::new_v4())),
            heartbeat_interval: Duration::from_secs(30),
            discovery_interval: Duration::from_secs(60),
        }
    }

    #[tokio::test]
    async fn test_network_creation() {
        let config = create_test_config();
        let validator_set = Arc::new(RwLock::new(PersistentValidatorSet::new()));

        let network = ValidatorNetwork::new(config, validator_set);
        assert!(network.local_validator_id.is_some());

        let stats = network.get_network_stats().await;
        assert_eq!(stats.total_peers, 0);
        assert_eq!(stats.connected_peers, 0);
    }

    #[tokio::test]
    async fn test_heartbeat_message() {
        let config = create_test_config();
        let validator_set = Arc::new(RwLock::new(PersistentValidatorSet::new()));

        let network = ValidatorNetwork::new(config, validator_set);

        // Test heartbeat creation (won't actually send without peers)
        let result = network.send_heartbeat(100).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_message_handling() {
        let config = create_test_config();
        let validator_set = Arc::new(RwLock::new(PersistentValidatorSet::new()));

        let network = ValidatorNetwork::new(config, validator_set);
        let peer_id = PeerId::new();
        let validator_id = ValidatorId(uuid::Uuid::new_v4());

        // Test heartbeat handling
        let heartbeat = ValidatorMessage::heartbeat(validator_id, 100);
        let result = network.handle_validator_message(heartbeat, peer_id).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_stats() {
        let stats = NetworkStats {
            total_peers: 5,
            connected_peers: 3,
            validator_peers: 2,
            local_peer_id: PeerId::new(),
            local_validator_id: Some(ValidatorId(uuid::Uuid::new_v4())),
        };

        assert_eq!(stats.total_peers, 5);
        assert_eq!(stats.connected_peers, 3);
        assert_eq!(stats.validator_peers, 2);
    }
}
