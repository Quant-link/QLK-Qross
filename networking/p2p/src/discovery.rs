use crate::{PeerInfo, PeerId, ConnectionStatus};
use std::collections::HashSet;
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use serde::{Deserialize, Serialize};

/// Discovery service for finding validator peers
pub struct PeerDiscovery {
    local_address: SocketAddr,
    bootstrap_peers: Vec<SocketAddr>,
    discovered_peers: HashSet<SocketAddr>,
    socket: Option<UdpSocket>,
}

/// Discovery message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryMessage {
    /// Announce presence to network
    Announce {
        peer_id: PeerId,
        listen_address: SocketAddr,
        protocol_version: String,
        validator_id: Option<qross_consensus::ValidatorId>,
    },
    /// Request peer list from another node
    PeerRequest {
        peer_id: PeerId,
    },
    /// Response with known peers
    PeerResponse {
        peer_id: PeerId,
        peers: Vec<PeerAdvertisement>,
    },
    /// Ping to check if peer is alive
    Ping {
        peer_id: PeerId,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Pong response to ping
    Pong {
        peer_id: PeerId,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

/// Peer advertisement for discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerAdvertisement {
    pub peer_id: PeerId,
    pub address: SocketAddr,
    pub protocol_version: String,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

impl PeerDiscovery {
    /// Create new peer discovery service
    pub fn new(local_address: SocketAddr, bootstrap_peers: Vec<SocketAddr>) -> Self {
        Self {
            local_address,
            bootstrap_peers,
            discovered_peers: HashSet::new(),
            socket: None,
        }
    }

    /// Start the discovery service
    pub async fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let socket = UdpSocket::bind(self.local_address).await?;
        self.socket = Some(socket);
        
        // Add bootstrap peers to discovered set
        for peer in &self.bootstrap_peers {
            self.discovered_peers.insert(*peer);
        }
        
        Ok(())
    }

    /// Announce presence to bootstrap peers
    pub async fn announce(&self, peer_id: PeerId, validator_id: Option<qross_consensus::ValidatorId>) -> Result<(), Box<dyn std::error::Error>> {
        let message = DiscoveryMessage::Announce {
            peer_id,
            listen_address: self.local_address,
            protocol_version: "1.0.0".to_string(),
            validator_id,
        };

        let serialized = serde_json::to_vec(&message)?;

        if let Some(socket) = &self.socket {
            for peer_addr in &self.bootstrap_peers {
                socket.send_to(&serialized, peer_addr).await?;
            }
        }

        Ok(())
    }

    /// Request peer list from known peers
    pub async fn request_peers(&self, peer_id: PeerId) -> Result<(), Box<dyn std::error::Error>> {
        let message = DiscoveryMessage::PeerRequest { peer_id };
        let serialized = serde_json::to_vec(&message)?;

        if let Some(socket) = &self.socket {
            for peer_addr in &self.discovered_peers {
                socket.send_to(&serialized, peer_addr).await?;
            }
        }

        Ok(())
    }

    /// Send ping to check peer liveness
    pub async fn ping_peer(&self, peer_id: PeerId, target: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
        let message = DiscoveryMessage::Ping {
            peer_id,
            timestamp: chrono::Utc::now(),
        };
        let serialized = serde_json::to_vec(&message)?;

        if let Some(socket) = &self.socket {
            socket.send_to(&serialized, &target).await?;
        }

        Ok(())
    }

    /// Listen for discovery messages
    pub async fn listen(&mut self) -> Result<Option<(DiscoveryMessage, SocketAddr)>, Box<dyn std::error::Error>> {
        if let Some(socket) = &self.socket {
            let mut buf = vec![0u8; 1024];
            let (len, addr) = socket.recv_from(&mut buf).await?;
            buf.truncate(len);

            match serde_json::from_slice::<DiscoveryMessage>(&buf) {
                Ok(message) => {
                    // Add sender to discovered peers
                    self.discovered_peers.insert(addr);
                    Ok(Some((message, addr)))
                }
                Err(_) => Ok(None), // Invalid message, ignore
            }
        } else {
            Err("Discovery service not started".into())
        }
    }

    /// Handle incoming discovery message
    pub async fn handle_message(&mut self, message: DiscoveryMessage, from: SocketAddr, local_peer_id: PeerId) -> Result<Option<PeerInfo>, Box<dyn std::error::Error>> {
        match message {
            DiscoveryMessage::Announce { peer_id, listen_address, protocol_version, validator_id } => {
                // Create peer info from announcement
                let peer_info = PeerInfo {
                    peer_id,
                    validator_id,
                    address: listen_address,
                    last_seen: chrono::Utc::now(),
                    connection_status: ConnectionStatus::Disconnected,
                    protocol_version,
                };
                
                self.discovered_peers.insert(listen_address);
                Ok(Some(peer_info))
            }
            DiscoveryMessage::PeerRequest { .. } => {
                // Respond with known peers
                let peers: Vec<PeerAdvertisement> = self.discovered_peers
                    .iter()
                    .map(|addr| PeerAdvertisement {
                        peer_id: PeerId::new(), // We don't track peer IDs in discovery
                        address: *addr,
                        protocol_version: "1.0.0".to_string(),
                        last_seen: chrono::Utc::now(),
                    })
                    .collect();

                let response = DiscoveryMessage::PeerResponse {
                    peer_id: local_peer_id,
                    peers,
                };

                let serialized = serde_json::to_vec(&response)?;
                if let Some(socket) = &self.socket {
                    socket.send_to(&serialized, &from).await?;
                }
                
                Ok(None)
            }
            DiscoveryMessage::PeerResponse { peers, .. } => {
                // Add discovered peers to our set
                for peer in peers {
                    self.discovered_peers.insert(peer.address);
                }
                Ok(None)
            }
            DiscoveryMessage::Ping { peer_id, timestamp } => {
                // Respond with pong
                let pong = DiscoveryMessage::Pong {
                    peer_id: local_peer_id,
                    timestamp,
                };

                let serialized = serde_json::to_vec(&pong)?;
                if let Some(socket) = &self.socket {
                    socket.send_to(&serialized, &from).await?;
                }
                Ok(None)
            }
            DiscoveryMessage::Pong { .. } => {
                // Peer is alive, update discovered peers
                self.discovered_peers.insert(from);
                Ok(None)
            }
        }
    }

    /// Get discovered peer addresses
    pub fn get_discovered_peers(&self) -> Vec<SocketAddr> {
        self.discovered_peers.iter().copied().collect()
    }

    /// Get peer count
    pub fn peer_count(&self) -> usize {
        self.discovered_peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_discovery_creation() {
        let local_addr = SocketAddr::from_str("127.0.0.1:8080").unwrap();
        let bootstrap = vec![SocketAddr::from_str("127.0.0.1:8081").unwrap()];
        
        let discovery = PeerDiscovery::new(local_addr, bootstrap.clone());
        assert_eq!(discovery.local_address, local_addr);
        assert_eq!(discovery.bootstrap_peers, bootstrap);
    }

    #[test]
    fn test_discovery_message_serialization() {
        let peer_id = PeerId::new();
        let message = DiscoveryMessage::Announce {
            peer_id,
            listen_address: SocketAddr::from_str("127.0.0.1:8080").unwrap(),
            protocol_version: "1.0.0".to_string(),
            validator_id: None,
        };

        let serialized = serde_json::to_vec(&message).unwrap();
        let deserialized: DiscoveryMessage = serde_json::from_slice(&serialized).unwrap();

        match deserialized {
            DiscoveryMessage::Announce { peer_id: id, .. } => {
                assert_eq!(id, peer_id);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[tokio::test]
    async fn test_discovery_start() {
        let local_addr = SocketAddr::from_str("127.0.0.1:0").unwrap(); // Use port 0 for auto-assignment
        let mut discovery = PeerDiscovery::new(local_addr, vec![]);
        
        let result = discovery.start().await;
        assert!(result.is_ok());
        assert!(discovery.socket.is_some());
    }
}
