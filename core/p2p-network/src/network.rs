//! Network manager for P2P operations

use crate::{types::*, error::*, ValidatorNetworkIntegration};
use libp2p::{PeerId, Multiaddr, Swarm, SwarmBuilder, SwarmEvent};
use std::collections::HashMap;

/// Network manager coordinating all P2P operations
pub struct NetworkManager {
    config: NetworkManagerConfig,
    swarm: Option<Swarm<NetworkBehaviour>>,
    validator_integration: Box<dyn ValidatorNetworkIntegration>,
    peer_registry: PeerRegistry,
    connection_manager: ConnectionManager,
    event_handler: EventHandler,
}

/// Custom network behaviour combining multiple protocols
#[derive(libp2p::swarm::NetworkBehaviour)]
pub struct NetworkBehaviour {
    pub gossipsub: libp2p::gossipsub::Gossipsub,
    pub kad: libp2p::kad::Kademlia<libp2p::kad::store::MemoryStore>,
    pub identify: libp2p::identify::Identify,
    pub ping: libp2p::ping::Ping,
    pub autonat: libp2p::autonat::Behaviour,
    pub relay: libp2p::relay::Behaviour,
    pub dcutr: libp2p::dcutr::Behaviour,
}

/// Peer registry for managing known peers
pub struct PeerRegistry {
    peers: HashMap<PeerId, PeerInfo>,
    validator_peers: HashMap<qross_consensus::ValidatorId, PeerId>,
    geographic_index: HashMap<GeographicRegion, Vec<PeerId>>,
}

/// Connection manager for handling peer connections
pub struct ConnectionManager {
    active_connections: HashMap<PeerId, ConnectionInfo>,
    connection_pool: HashMap<PeerId, Vec<ConnectionInfo>>,
    connection_limits: ConnectionLimits,
}

/// Connection limits configuration
#[derive(Debug, Clone)]
pub struct ConnectionLimits {
    pub max_connections: usize,
    pub max_connections_per_peer: usize,
    pub connection_timeout: std::time::Duration,
    pub idle_timeout: std::time::Duration,
}

/// Event handler for network events
pub struct EventHandler {
    event_queue: tokio::sync::mpsc::UnboundedReceiver<NetworkEvent>,
    event_sender: tokio::sync::mpsc::UnboundedSender<NetworkEvent>,
}

/// Network events
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    MessageReceived { from: PeerId, message: NetworkMessage },
    ConnectionFailed { peer: PeerId, error: String },
    DiscoveryUpdate { peers: Vec<PeerInfo> },
}

impl NetworkManager {
    /// Create a new network manager
    pub fn new(
        config: NetworkManagerConfig,
        validator_integration: Box<dyn ValidatorNetworkIntegration>,
    ) -> Self {
        let (event_sender, event_queue) = tokio::sync::mpsc::unbounded_channel();
        
        Self {
            config,
            swarm: None,
            validator_integration,
            peer_registry: PeerRegistry::new(),
            connection_manager: ConnectionManager::new(ConnectionLimits::default()),
            event_handler: EventHandler {
                event_queue,
                event_sender,
            },
        }
    }
    
    /// Start network manager
    pub async fn start(&mut self) -> Result<()> {
        // Create libp2p identity
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        
        tracing::info!("Starting network manager with peer ID: {}", local_peer_id);
        
        // Create transport
        let transport = libp2p::tokio_development_transport(local_key.clone())
            .map_err(|e| NetworkError::TransportError(format!("Failed to create transport: {}", e)))?;
        
        // Create network behaviour
        let behaviour = self.create_network_behaviour(local_key).await?;
        
        // Create swarm
        let mut swarm = SwarmBuilder::with_tokio_executor(transport, behaviour, local_peer_id)
            .build();
        
        // Configure listening addresses
        for addr_str in &self.config.listen_addresses {
            let addr: Multiaddr = addr_str.parse()
                .map_err(|e| NetworkError::ConfigurationError(format!("Invalid listen address: {}", e)))?;
            swarm.listen_on(addr)
                .map_err(|e| NetworkError::TransportError(format!("Failed to listen on address: {}", e)))?;
        }
        
        self.swarm = Some(swarm);
        
        // Start event processing
        self.start_event_processing().await?;
        
        tracing::info!("Network manager started successfully");
        
        Ok(())
    }
    
    /// Stop network manager
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(swarm) = &mut self.swarm {
            // Close all connections gracefully
            for peer_id in swarm.connected_peers().cloned().collect::<Vec<_>>() {
                let _ = swarm.disconnect_peer_id(peer_id);
            }
        }
        
        self.swarm = None;
        
        tracing::info!("Network manager stopped");
        
        Ok(())
    }
    
    /// Send message via route
    pub async fn send_message_via_route(&self, route: &RoutingPath, message: NetworkMessage) -> Result<()> {
        if route.hops.is_empty() {
            return Err(NetworkError::RoutingError("Empty route".to_string()));
        }
        
        // For now, send directly to the target (last hop)
        let target = route.hops.last().unwrap();
        self.send_direct_message(target, message).await
    }
    
    /// Send direct message to peer
    pub async fn send_direct_message(&self, peer_id: &PeerId, message: NetworkMessage) -> Result<()> {
        if let Some(swarm) = &self.swarm {
            // Check if peer is connected
            if !swarm.is_connected(peer_id) {
                return Err(NetworkError::PeerNotConnected(*peer_id));
            }
            
            // Serialize message
            let serialized = bincode::serialize(&message)
                .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
            
            // TODO: Send message using appropriate protocol
            // This would depend on the message type and protocol
            
            tracing::debug!("Sent direct message to peer {}", peer_id);
            
            Ok(())
        } else {
            Err(NetworkError::Internal("Network manager not started".to_string()))
        }
    }
    
    /// Get local peer ID
    pub fn get_local_peer_id(&self) -> PeerId {
        if let Some(swarm) = &self.swarm {
            *swarm.local_peer_id()
        } else {
            PeerId::random() // Fallback
        }
    }
    
    /// Create network behaviour
    async fn create_network_behaviour(&self, local_key: libp2p::identity::Keypair) -> Result<NetworkBehaviour> {
        // Create gossipsub
        let gossipsub_config = libp2p::gossipsub::GossipsubConfigBuilder::default()
            .heartbeat_interval(std::time::Duration::from_secs(1))
            .validation_mode(libp2p::gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|e| NetworkError::ConfigurationError(format!("Gossipsub config error: {}", e)))?;
        
        let gossipsub = libp2p::gossipsub::Gossipsub::new(
            libp2p::gossipsub::MessageAuthenticity::Signed(local_key.clone()),
            gossipsub_config,
        ).map_err(|e| NetworkError::GossipError(format!("Failed to create gossipsub: {}", e)))?;
        
        // Create Kademlia DHT
        let store = libp2p::kad::store::MemoryStore::new(self.get_local_peer_id());
        let kad = libp2p::kad::Kademlia::new(self.get_local_peer_id(), store);
        
        // Create identify protocol
        let identify = libp2p::identify::Identify::new(
            libp2p::identify::Config::new("qross/1.0.0".to_string(), local_key.public())
                .with_agent_version("qross-node/0.1.0".to_string()),
        );
        
        // Create ping protocol
        let ping = libp2p::ping::Ping::new(
            libp2p::ping::Config::new()
                .with_interval(std::time::Duration::from_secs(30))
                .with_timeout(std::time::Duration::from_secs(10)),
        );
        
        // Create AutoNAT
        let autonat = libp2p::autonat::Behaviour::new(
            self.get_local_peer_id(),
            libp2p::autonat::Config::default(),
        );
        
        // Create relay behaviour
        let relay = libp2p::relay::Behaviour::new(
            self.get_local_peer_id(),
            libp2p::relay::Config::default(),
        );
        
        // Create DCUtR behaviour
        let dcutr = libp2p::dcutr::Behaviour::new(self.get_local_peer_id());
        
        Ok(NetworkBehaviour {
            gossipsub,
            kad,
            identify,
            ping,
            autonat,
            relay,
            dcutr,
        })
    }
    
    /// Start event processing loop
    async fn start_event_processing(&mut self) -> Result<()> {
        let swarm = self.swarm.take()
            .ok_or_else(|| NetworkError::Internal("Swarm not initialized".to_string()))?;
        
        let event_sender = self.event_handler.event_sender.clone();
        
        // Spawn event processing task
        tokio::spawn(async move {
            let mut swarm = swarm;
            loop {
                match swarm.select_next_some().await {
                    SwarmEvent::NewListenAddr { address, .. } => {
                        tracing::info!("Listening on {}", address);
                    }
                    SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                        tracing::info!("Connected to peer: {}", peer_id);
                        let _ = event_sender.send(NetworkEvent::PeerConnected(peer_id));
                    }
                    SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                        tracing::info!("Disconnected from peer: {} (cause: {:?})", peer_id, cause);
                        let _ = event_sender.send(NetworkEvent::PeerDisconnected(peer_id));
                    }
                    SwarmEvent::Behaviour(event) => {
                        if let Err(e) = Self::handle_behaviour_event(event, &event_sender).await {
                            tracing::error!("Error handling behaviour event: {}", e);
                        }
                    }
                    _ => {}
                }
            }
        });
        
        Ok(())
    }
    
    /// Handle behaviour events
    async fn handle_behaviour_event(
        event: <NetworkBehaviour as libp2p::swarm::NetworkBehaviour>::OutEvent,
        event_sender: &tokio::sync::mpsc::UnboundedSender<NetworkEvent>,
    ) -> Result<()> {
        match event {
            NetworkBehaviourEvent::Gossipsub(gossip_event) => {
                // Handle gossipsub events
                tracing::debug!("Gossipsub event: {:?}", gossip_event);
            }
            NetworkBehaviourEvent::Kad(kad_event) => {
                // Handle Kademlia events
                tracing::debug!("Kademlia event: {:?}", kad_event);
            }
            NetworkBehaviourEvent::Identify(identify_event) => {
                // Handle identify events
                tracing::debug!("Identify event: {:?}", identify_event);
            }
            NetworkBehaviourEvent::Ping(ping_event) => {
                // Handle ping events
                tracing::debug!("Ping event: {:?}", ping_event);
            }
            NetworkBehaviourEvent::Autonat(autonat_event) => {
                // Handle AutoNAT events
                tracing::debug!("AutoNAT event: {:?}", autonat_event);
            }
            NetworkBehaviourEvent::Relay(relay_event) => {
                // Handle relay events
                tracing::debug!("Relay event: {:?}", relay_event);
            }
            NetworkBehaviourEvent::Dcutr(dcutr_event) => {
                // Handle DCUtR events
                tracing::debug!("DCUtR event: {:?}", dcutr_event);
            }
        }
        
        Ok(())
    }
    
    /// Add peer to registry
    pub async fn add_peer(&mut self, peer_info: PeerInfo) -> Result<()> {
        self.peer_registry.add_peer(peer_info).await
    }
    
    /// Get peer information
    pub async fn get_peer_info(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        self.peer_registry.get_peer(peer_id)
    }
    
    /// Get connected peers
    pub fn get_connected_peers(&self) -> Vec<PeerId> {
        if let Some(swarm) = &self.swarm {
            swarm.connected_peers().cloned().collect()
        } else {
            Vec::new()
        }
    }
}

impl PeerRegistry {
    fn new() -> Self {
        Self {
            peers: HashMap::new(),
            validator_peers: HashMap::new(),
            geographic_index: HashMap::new(),
        }
    }
    
    async fn add_peer(&mut self, peer_info: PeerInfo) -> Result<()> {
        // Add to main registry
        self.peers.insert(peer_info.peer_id, peer_info.clone());
        
        // Add to validator mapping if applicable
        if let Some(validator_id) = peer_info.validator_id {
            self.validator_peers.insert(validator_id, peer_info.peer_id);
        }
        
        // Add to geographic index
        if let Some(location) = &peer_info.geographic_location {
            self.geographic_index.entry(location.region.clone())
                .or_insert_with(Vec::new)
                .push(peer_info.peer_id);
        }
        
        Ok(())
    }
    
    fn get_peer(&self, peer_id: &PeerId) -> Option<&PeerInfo> {
        self.peers.get(peer_id)
    }
}

impl ConnectionManager {
    fn new(limits: ConnectionLimits) -> Self {
        Self {
            active_connections: HashMap::new(),
            connection_pool: HashMap::new(),
            connection_limits: limits,
        }
    }
}

impl Default for ConnectionLimits {
    fn default() -> Self {
        Self {
            max_connections: 1000,
            max_connections_per_peer: 5,
            connection_timeout: std::time::Duration::from_secs(30),
            idle_timeout: std::time::Duration::from_secs(300),
        }
    }
}

// Define the behaviour event enum
#[derive(Debug)]
pub enum NetworkBehaviourEvent {
    Gossipsub(libp2p::gossipsub::GossipsubEvent),
    Kad(libp2p::kad::KademliaEvent),
    Identify(libp2p::identify::Event),
    Ping(libp2p::ping::Event),
    Autonat(libp2p::autonat::Event),
    Relay(libp2p::relay::Event),
    Dcutr(libp2p::dcutr::Event),
}

// Implement From traits for behaviour events
impl From<libp2p::gossipsub::GossipsubEvent> for NetworkBehaviourEvent {
    fn from(event: libp2p::gossipsub::GossipsubEvent) -> Self {
        NetworkBehaviourEvent::Gossipsub(event)
    }
}

impl From<libp2p::kad::KademliaEvent> for NetworkBehaviourEvent {
    fn from(event: libp2p::kad::KademliaEvent) -> Self {
        NetworkBehaviourEvent::Kad(event)
    }
}

impl From<libp2p::identify::Event> for NetworkBehaviourEvent {
    fn from(event: libp2p::identify::Event) -> Self {
        NetworkBehaviourEvent::Identify(event)
    }
}

impl From<libp2p::ping::Event> for NetworkBehaviourEvent {
    fn from(event: libp2p::ping::Event) -> Self {
        NetworkBehaviourEvent::Ping(event)
    }
}

impl From<libp2p::autonat::Event> for NetworkBehaviourEvent {
    fn from(event: libp2p::autonat::Event) -> Self {
        NetworkBehaviourEvent::Autonat(event)
    }
}

impl From<libp2p::relay::Event> for NetworkBehaviourEvent {
    fn from(event: libp2p::relay::Event) -> Self {
        NetworkBehaviourEvent::Relay(event)
    }
}

impl From<libp2p::dcutr::Event> for NetworkBehaviourEvent {
    fn from(event: libp2p::dcutr::Event) -> Self {
        NetworkBehaviourEvent::Dcutr(event)
    }
}
