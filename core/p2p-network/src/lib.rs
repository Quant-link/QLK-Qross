//! Quantlink Qross P2P Network Stack
//! 
//! This module implements libp2p-based custom networking with QUIC protocol for
//! 0-RTT connection establishment, message routing protocols coordinating with
//! proof aggregation, and Byzantine fault tolerant communication.

pub mod network;
pub mod transport;
pub mod routing;
pub mod gossip;
pub mod discovery;
pub mod relay;
pub mod security;
pub mod bandwidth;
pub mod topology;
pub mod types;
pub mod error;
pub mod metrics;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use libp2p::{PeerId, Multiaddr};
use qross_consensus::ValidatorId;
use qross_proof_aggregation::{ProofId, AggregatedProof};

pub use error::{NetworkError, Result};
pub use types::*;

/// Main P2P network engine
pub struct P2PNetworkEngine {
    network_manager: network::NetworkManager,
    transport_layer: transport::TransportLayer,
    routing_engine: routing::RoutingEngine,
    gossip_protocol: gossip::GossipProtocol,
    discovery_service: discovery::DiscoveryService,
    relay_service: relay::RelayService,
    security_manager: security::SecurityManager,
    bandwidth_manager: bandwidth::BandwidthManager,
    topology_manager: topology::DynamicTopologyManager,
    metrics: metrics::NetworkMetrics,
    config: NetworkConfig,
    active_connections: dashmap::DashMap<PeerId, ConnectionInfo>,
    message_cache: dashmap::DashMap<MessageId, CachedMessage>,
}

/// Trait for proof distribution integration
#[async_trait]
pub trait ProofDistributionIntegration: Send + Sync {
    /// Distribute zk-STARK proof to network
    async fn distribute_proof(&self, proof: &AggregatedProof, targets: &[PeerId]) -> Result<()>;
    
    /// Request proof from network
    async fn request_proof(&self, proof_id: ProofId, from: &PeerId) -> Result<Option<AggregatedProof>>;
    
    /// Coordinate batch distribution with aggregation protocol
    async fn coordinate_batch_distribution(&self, batch_info: &BatchDistributionInfo) -> Result<()>;
}

/// Trait for validator network integration
#[async_trait]
pub trait ValidatorNetworkIntegration: Send + Sync {
    /// Get validator network topology
    async fn get_validator_topology(&self) -> Result<HashMap<ValidatorId, ValidatorNetworkInfo>>;
    
    /// Update validator connectivity status
    async fn update_validator_connectivity(&self, validator_id: &ValidatorId, status: ConnectivityStatus) -> Result<()>;
    
    /// Get geographic distribution for latency optimization
    async fn get_geographic_distribution(&self) -> Result<HashMap<ValidatorId, GeographicLocation>>;
}

impl P2PNetworkEngine {
    /// Create a new P2P network engine
    pub fn new(
        config: NetworkConfig,
        validator_integration: Box<dyn ValidatorNetworkIntegration>,
        proof_integration: Box<dyn ProofDistributionIntegration>,
    ) -> Self {
        let network_manager = network::NetworkManager::new(
            config.network_config.clone(),
            validator_integration,
        );
        let transport_layer = transport::TransportLayer::new(config.transport_config.clone());
        let routing_engine = routing::RoutingEngine::new(config.routing_config.clone());
        let gossip_protocol = gossip::GossipProtocol::new(config.gossip_config.clone());
        let discovery_service = discovery::DiscoveryService::new(config.discovery_config.clone());
        let relay_service = relay::RelayService::new(config.relay_config.clone());
        let security_manager = security::SecurityManager::new(config.security_config.clone());
        let bandwidth_manager = bandwidth::BandwidthManager::new(config.bandwidth_config.clone());
        let topology_manager = topology::DynamicTopologyManager::new(config.topology_config.clone());
        let metrics = metrics::NetworkMetrics::new();
        
        Self {
            network_manager,
            transport_layer,
            routing_engine,
            gossip_protocol,
            discovery_service,
            relay_service,
            security_manager,
            bandwidth_manager,
            topology_manager,
            metrics,
            config,
            active_connections: dashmap::DashMap::new(),
            message_cache: dashmap::DashMap::new(),
        }
    }
    
    /// Start the P2P network
    pub async fn start(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Initialize transport layer with QUIC support
        self.transport_layer.initialize().await?;
        
        // Start discovery service
        self.discovery_service.start().await?;
        
        // Initialize routing engine
        self.routing_engine.initialize().await?;
        
        // Start gossip protocol
        self.gossip_protocol.start().await?;
        
        // Initialize relay service
        self.relay_service.start().await?;
        
        // Start bandwidth management
        self.bandwidth_manager.start().await?;

        // Start topology management
        self.topology_manager.start().await?;

        // Begin network operations
        self.network_manager.start().await?;
        
        self.metrics.record_startup_time(start_time.elapsed().as_secs_f64());
        
        tracing::info!("P2P network started successfully in {:.2}s", start_time.elapsed().as_secs_f64());
        
        Ok(())
    }
    
    /// Send message to specific peer
    pub async fn send_message(&self, peer_id: &PeerId, message: NetworkMessage) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Check connection status
        if !self.is_peer_connected(peer_id).await? {
            self.establish_connection(peer_id).await?;
        }
        
        // Apply security policies
        let secured_message = self.security_manager.secure_message(&message).await?;
        
        // Route message through optimal path
        let route = self.routing_engine.find_optimal_route(peer_id).await?;
        
        // Send message
        self.network_manager.send_message_via_route(&route, secured_message).await?;
        
        // Update metrics
        self.metrics.record_message_sent();
        self.metrics.record_message_latency(start_time.elapsed().as_secs_f64());
        
        tracing::debug!("Sent message to peer {} via route with {} hops", peer_id, route.hops.len());
        
        Ok(())
    }
    
    /// Broadcast message to network
    pub async fn broadcast_message(&self, message: NetworkMessage, targets: BroadcastTargets) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Determine target peers
        let target_peers = self.resolve_broadcast_targets(targets).await?;
        
        // Use gossip protocol for efficient broadcasting
        self.gossip_protocol.broadcast_message(message, &target_peers).await?;
        
        self.metrics.record_broadcast_sent();
        self.metrics.record_broadcast_latency(start_time.elapsed().as_secs_f64());
        
        tracing::info!("Broadcast message to {} peers", target_peers.len());
        
        Ok(())
    }
    
    /// Distribute zk-STARK proof to network
    pub async fn distribute_proof(&self, proof: &AggregatedProof) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Determine optimal distribution strategy
        let distribution_strategy = self.calculate_proof_distribution_strategy(proof).await?;
        
        // Create proof distribution message
        let message = NetworkMessage::ProofDistribution {
            proof_id: proof.id,
            proof_data: self.serialize_proof(proof)?,
            distribution_strategy: distribution_strategy.clone(),
            timestamp: Utc::now(),
        };
        
        // Distribute based on strategy
        match distribution_strategy {
            ProofDistributionStrategy::Gossip => {
                self.gossip_protocol.distribute_proof(message).await?;
            }
            ProofDistributionStrategy::DirectRouting { targets } => {
                for target in targets {
                    self.send_message(&target, message.clone()).await?;
                }
            }
            ProofDistributionStrategy::Hybrid { gossip_targets, direct_targets } => {
                // Use gossip for general distribution
                self.gossip_protocol.broadcast_message(message.clone(), &gossip_targets).await?;
                
                // Use direct routing for critical validators
                for target in direct_targets {
                    self.send_message(&target, message.clone()).await?;
                }
            }
        }
        
        self.metrics.record_proof_distribution_time(start_time.elapsed().as_secs_f64());
        
        tracing::info!("Distributed proof {} using strategy {:?}", proof.id, distribution_strategy);
        
        Ok(())
    }
    
    /// Request proof from network
    pub async fn request_proof(&self, proof_id: ProofId, preferred_sources: &[PeerId]) -> Result<Option<AggregatedProof>> {
        let start_time = std::time::Instant::now();
        
        // Check local cache first
        if let Some(cached) = self.get_cached_proof(proof_id).await? {
            return Ok(Some(cached));
        }
        
        // Create proof request message
        let request = NetworkMessage::ProofRequest {
            proof_id,
            requester: self.network_manager.get_local_peer_id(),
            timestamp: Utc::now(),
        };
        
        // Try preferred sources first
        for source in preferred_sources {
            if let Ok(Some(proof)) = self.request_proof_from_peer(source, &request).await {
                self.cache_proof(proof.clone()).await?;
                self.metrics.record_proof_request_success();
                return Ok(Some(proof));
            }
        }
        
        // Fallback to network-wide request
        let response = self.gossip_protocol.request_proof(request).await?;
        
        if let Some(proof) = response {
            self.cache_proof(proof.clone()).await?;
            self.metrics.record_proof_request_success();
        } else {
            self.metrics.record_proof_request_failure();
        }
        
        self.metrics.record_proof_request_time(start_time.elapsed().as_secs_f64());
        
        Ok(response)
    }
    
    /// Establish connection to peer
    async fn establish_connection(&self, peer_id: &PeerId) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Get peer addresses from discovery
        let addresses = self.discovery_service.get_peer_addresses(peer_id).await?;
        
        if addresses.is_empty() {
            return Err(NetworkError::PeerNotFound(*peer_id));
        }
        
        // Try QUIC connection first for 0-RTT
        for address in &addresses {
            if let Ok(connection) = self.transport_layer.establish_quic_connection(peer_id, address).await {
                let connection_info = ConnectionInfo {
                    peer_id: *peer_id,
                    address: address.clone(),
                    connection_type: ConnectionType::Quic,
                    established_at: Utc::now(),
                    last_activity: Utc::now(),
                    bandwidth_usage: BandwidthUsage::default(),
                    latency_stats: LatencyStats::default(),
                };
                
                self.active_connections.insert(*peer_id, connection_info);
                self.metrics.record_connection_established();
                self.metrics.record_connection_time(start_time.elapsed().as_secs_f64());
                
                tracing::info!("Established QUIC connection to peer {} at {}", peer_id, address);
                
                return Ok(());
            }
        }
        
        // Fallback to TCP if QUIC fails
        for address in &addresses {
            if let Ok(connection) = self.transport_layer.establish_tcp_connection(peer_id, address).await {
                let connection_info = ConnectionInfo {
                    peer_id: *peer_id,
                    address: address.clone(),
                    connection_type: ConnectionType::Tcp,
                    established_at: Utc::now(),
                    last_activity: Utc::now(),
                    bandwidth_usage: BandwidthUsage::default(),
                    latency_stats: LatencyStats::default(),
                };
                
                self.active_connections.insert(*peer_id, connection_info);
                self.metrics.record_connection_established();
                self.metrics.record_connection_time(start_time.elapsed().as_secs_f64());
                
                tracing::info!("Established TCP connection to peer {} at {}", peer_id, address);
                
                return Ok(());
            }
        }
        
        Err(NetworkError::ConnectionFailed(*peer_id))
    }
    
    /// Check if peer is connected
    async fn is_peer_connected(&self, peer_id: &PeerId) -> Result<bool> {
        Ok(self.active_connections.contains_key(peer_id))
    }
    
    /// Calculate optimal proof distribution strategy
    async fn calculate_proof_distribution_strategy(&self, proof: &AggregatedProof) -> Result<ProofDistributionStrategy> {
        let proof_size = self.estimate_proof_size(proof)?;
        let network_topology = self.routing_engine.get_network_topology().await?;
        let bandwidth_availability = self.bandwidth_manager.get_available_bandwidth().await?;
        
        // Use gossip for small proofs or high bandwidth
        if proof_size < 1024 * 1024 || bandwidth_availability > 0.8 { // 1MB threshold
            Ok(ProofDistributionStrategy::Gossip)
        } else {
            // Use hybrid approach for large proofs
            let critical_validators = self.get_critical_validators().await?;
            let gossip_targets = self.get_gossip_targets(&network_topology).await?;
            
            Ok(ProofDistributionStrategy::Hybrid {
                gossip_targets,
                direct_targets: critical_validators,
            })
        }
    }
    
    /// Resolve broadcast targets
    async fn resolve_broadcast_targets(&self, targets: BroadcastTargets) -> Result<Vec<PeerId>> {
        match targets {
            BroadcastTargets::All => {
                Ok(self.active_connections.iter().map(|entry| *entry.key()).collect())
            }
            BroadcastTargets::Validators => {
                self.get_validator_peers().await
            }
            BroadcastTargets::Specific(peers) => {
                Ok(peers)
            }
            BroadcastTargets::Geographic(region) => {
                self.get_peers_in_region(region).await
            }
        }
    }
    
    /// Get validator peers
    async fn get_validator_peers(&self) -> Result<Vec<PeerId>> {
        // TODO: Integrate with validator network to get peer IDs
        Ok(self.active_connections.iter()
            .filter(|entry| entry.value().connection_type == ConnectionType::Quic)
            .map(|entry| *entry.key())
            .collect())
    }
    
    /// Get peers in geographic region
    async fn get_peers_in_region(&self, _region: GeographicRegion) -> Result<Vec<PeerId>> {
        // TODO: Implement geographic filtering based on peer locations
        Ok(Vec::new())
    }
    
    /// Get critical validators for direct routing
    async fn get_critical_validators(&self) -> Result<Vec<PeerId>> {
        // TODO: Integrate with validator selection to get critical validators
        Ok(Vec::new())
    }
    
    /// Get gossip targets from network topology
    async fn get_gossip_targets(&self, _topology: &NetworkTopology) -> Result<Vec<PeerId>> {
        // TODO: Implement topology-aware gossip target selection
        Ok(self.active_connections.iter().map(|entry| *entry.key()).take(10).collect())
    }
    
    /// Serialize proof for network transmission
    fn serialize_proof(&self, proof: &AggregatedProof) -> Result<Vec<u8>> {
        bincode::serialize(proof)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))
    }
    
    /// Estimate proof size for distribution strategy
    fn estimate_proof_size(&self, proof: &AggregatedProof) -> Result<usize> {
        self.serialize_proof(proof).map(|data| data.len())
    }
    
    /// Request proof from specific peer
    async fn request_proof_from_peer(&self, peer_id: &PeerId, request: &NetworkMessage) -> Result<Option<AggregatedProof>> {
        // TODO: Implement direct peer proof request
        Ok(None)
    }
    
    /// Get cached proof
    async fn get_cached_proof(&self, proof_id: ProofId) -> Result<Option<AggregatedProof>> {
        if let Some(cached) = self.message_cache.get(&MessageId::from(proof_id)) {
            if let NetworkMessage::ProofDistribution { proof_data, .. } = &cached.message {
                let proof: AggregatedProof = bincode::deserialize(proof_data)
                    .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
                return Ok(Some(proof));
            }
        }
        Ok(None)
    }
    
    /// Cache proof for future requests
    async fn cache_proof(&self, proof: AggregatedProof) -> Result<()> {
        let message = NetworkMessage::ProofDistribution {
            proof_id: proof.id,
            proof_data: self.serialize_proof(&proof)?,
            distribution_strategy: ProofDistributionStrategy::Gossip,
            timestamp: Utc::now(),
        };
        
        let cached_message = CachedMessage {
            message,
            cached_at: Utc::now(),
            access_count: 0,
            size: self.estimate_proof_size(&proof)?,
        };
        
        self.message_cache.insert(MessageId::from(proof.id), cached_message);
        
        Ok(())
    }
    
    /// Perform network health check
    pub async fn perform_health_check(&mut self) -> Result<topology::NetworkHealthReport> {
        self.topology_manager.perform_health_check().await
    }

    /// Get topology statistics
    pub fn get_topology_statistics(&self) -> topology::TopologyStatistics {
        self.topology_manager.get_topology_statistics()
    }

    /// Get network statistics
    pub fn get_network_statistics(&self) -> NetworkStatistics {
        let active_connections = self.active_connections.len();
        let cached_messages = self.message_cache.len();

        NetworkStatistics {
            active_connections,
            cached_messages,
            total_messages_sent: self.metrics.get_total_messages_sent(),
            total_messages_received: self.metrics.get_total_messages_received(),
            average_latency: self.metrics.get_average_latency(),
            bandwidth_utilization: self.bandwidth_manager.get_current_utilization(),
        }
    }
    
    /// Shutdown the network
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down P2P network");
        
        // Stop all services
        self.bandwidth_manager.stop().await?;
        self.relay_service.stop().await?;
        self.gossip_protocol.stop().await?;
        self.discovery_service.stop().await?;
        self.network_manager.stop().await?;
        
        // Close all connections
        self.active_connections.clear();
        self.message_cache.clear();
        
        tracing::info!("P2P network shutdown complete");
        
        Ok(())
    }
}
