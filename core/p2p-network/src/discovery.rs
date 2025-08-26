//! Peer discovery service with mDNS and Kademlia DHT

use crate::{types::*, error::*};
use libp2p::{PeerId, Multiaddr, kad::{Kademlia, KademliaEvent}, mdns::{Mdns, MdnsEvent}};
use std::collections::{HashMap, HashSet};

/// Discovery service for finding and maintaining peer connections
pub struct DiscoveryService {
    config: DiscoveryConfig,
    kademlia: Option<Kademlia<libp2p::kad::store::MemoryStore>>,
    mdns: Option<Mdns>,
    discovered_peers: HashMap<PeerId, PeerInfo>,
    bootstrap_peers: Vec<PeerInfo>,
    discovery_cache: DiscoveryCache,
}

/// Discovery cache for efficient peer lookup
pub struct DiscoveryCache {
    peer_addresses: HashMap<PeerId, Vec<Multiaddr>>,
    geographic_peers: HashMap<GeographicRegion, HashSet<PeerId>>,
    validator_peers: HashMap<qross_consensus::ValidatorId, PeerId>,
    cache_expiry: HashMap<PeerId, chrono::DateTime<chrono::Utc>>,
}

impl DiscoveryService {
    /// Create a new discovery service
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            kademlia: None,
            mdns: None,
            discovered_peers: HashMap::new(),
            bootstrap_peers: Vec::new(),
            discovery_cache: DiscoveryCache::new(),
        }
    }
    
    /// Start discovery service
    pub async fn start(&mut self) -> Result<()> {
        // Initialize Kademlia DHT if enabled
        if self.config.enable_kad {
            self.initialize_kademlia().await?;
        }
        
        // Initialize mDNS if enabled
        if self.config.enable_mdns {
            self.initialize_mdns().await?;
        }
        
        // Connect to bootstrap peers
        self.connect_to_bootstrap_peers().await?;
        
        tracing::info!("Discovery service started with Kademlia: {}, mDNS: {}", 
                      self.config.enable_kad, self.config.enable_mdns);
        
        Ok(())
    }
    
    /// Initialize Kademlia DHT
    async fn initialize_kademlia(&mut self) -> Result<()> {
        let local_peer_id = self.get_local_peer_id();
        let store = libp2p::kad::store::MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::new(local_peer_id, store);
        
        // Configure Kademlia
        kademlia.set_replication_factor(
            std::num::NonZeroUsize::new(self.config.kad_replication_factor)
                .ok_or_else(|| NetworkError::ConfigurationError("Invalid replication factor".to_string()))?
        );
        
        // Add bootstrap peers to Kademlia
        for bootstrap_addr in &self.config.bootstrap_peers {
            if let Ok(addr) = bootstrap_addr.parse::<Multiaddr>() {
                if let Some(peer_id) = self.extract_peer_id_from_multiaddr(&addr) {
                    kademlia.add_address(&peer_id, addr);
                }
            }
        }
        
        self.kademlia = Some(kademlia);
        
        tracing::info!("Kademlia DHT initialized with {} bootstrap peers", self.config.bootstrap_peers.len());
        
        Ok(())
    }
    
    /// Initialize mDNS discovery
    async fn initialize_mdns(&mut self) -> Result<()> {
        let mdns = Mdns::new(libp2p::mdns::Config::default())
            .await
            .map_err(|e| NetworkError::DiscoveryError(format!("Failed to initialize mDNS: {}", e)))?;
        
        self.mdns = Some(mdns);
        
        tracing::info!("mDNS discovery initialized");
        
        Ok(())
    }
    
    /// Connect to bootstrap peers
    async fn connect_to_bootstrap_peers(&mut self) -> Result<()> {
        for bootstrap_addr in &self.config.bootstrap_peers {
            if let Ok(addr) = bootstrap_addr.parse::<Multiaddr>() {
                if let Some(peer_id) = self.extract_peer_id_from_multiaddr(&addr) {
                    let peer_info = PeerInfo {
                        peer_id,
                        addresses: vec![addr],
                        validator_id: None,
                        capabilities: PeerCapabilities::default(),
                        geographic_location: None,
                        last_seen: chrono::Utc::now(),
                    };
                    
                    self.bootstrap_peers.push(peer_info);
                }
            }
        }
        
        tracing::info!("Connected to {} bootstrap peers", self.bootstrap_peers.len());
        
        Ok(())
    }
    
    /// Handle Kademlia event
    pub async fn handle_kademlia_event(&mut self, event: KademliaEvent) -> Result<()> {
        match event {
            KademliaEvent::OutboundQueryProgressed { result, .. } => {
                self.handle_query_result(result).await?;
            }
            KademliaEvent::RoutingUpdated { peer, .. } => {
                self.handle_routing_update(peer).await?;
            }
            KademliaEvent::UnroutablePeer { peer } => {
                self.handle_unroutable_peer(peer).await?;
            }
            KademliaEvent::RoutablePeer { peer, address } => {
                self.handle_routable_peer(peer, address).await?;
            }
            KademliaEvent::PendingRoutablePeer { peer, address } => {
                self.handle_pending_routable_peer(peer, address).await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle mDNS event
    pub async fn handle_mdns_event(&mut self, event: MdnsEvent) -> Result<()> {
        match event {
            MdnsEvent::Discovered(list) => {
                for (peer_id, multiaddr) in list {
                    self.handle_peer_discovered(peer_id, multiaddr).await?;
                }
            }
            MdnsEvent::Expired(list) => {
                for (peer_id, _) in list {
                    self.handle_peer_expired(peer_id).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle query result
    async fn handle_query_result(&mut self, result: libp2p::kad::QueryResult) -> Result<()> {
        match result {
            libp2p::kad::QueryResult::GetClosestPeers(Ok(peers)) => {
                for peer in peers.peers {
                    self.add_discovered_peer(peer, Vec::new()).await?;
                }
            }
            libp2p::kad::QueryResult::GetProviders(Ok(providers)) => {
                for peer in providers.providers {
                    self.add_discovered_peer(peer, Vec::new()).await?;
                }
            }
            libp2p::kad::QueryResult::Bootstrap(Ok(bootstrap)) => {
                tracing::info!("Bootstrap completed with {} peers", bootstrap.num_remaining);
            }
            _ => {
                tracing::debug!("Kademlia query result: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    /// Handle routing update
    async fn handle_routing_update(&mut self, peer: PeerId) -> Result<()> {
        tracing::debug!("Routing updated for peer: {}", peer);
        Ok(())
    }
    
    /// Handle unroutable peer
    async fn handle_unroutable_peer(&mut self, peer: PeerId) -> Result<()> {
        // Remove from cache
        self.discovery_cache.remove_peer(&peer);
        tracing::debug!("Peer {} is unroutable", peer);
        Ok(())
    }
    
    /// Handle routable peer
    async fn handle_routable_peer(&mut self, peer: PeerId, address: Multiaddr) -> Result<()> {
        self.add_discovered_peer(peer, vec![address]).await
    }
    
    /// Handle pending routable peer
    async fn handle_pending_routable_peer(&mut self, peer: PeerId, address: Multiaddr) -> Result<()> {
        tracing::debug!("Pending routable peer: {} at {}", peer, address);
        Ok(())
    }
    
    /// Handle peer discovered via mDNS
    async fn handle_peer_discovered(&mut self, peer_id: PeerId, multiaddr: Multiaddr) -> Result<()> {
        tracing::info!("Discovered peer via mDNS: {} at {}", peer_id, multiaddr);
        self.add_discovered_peer(peer_id, vec![multiaddr]).await
    }
    
    /// Handle peer expired from mDNS
    async fn handle_peer_expired(&mut self, peer_id: PeerId) -> Result<()> {
        self.discovered_peers.remove(&peer_id);
        self.discovery_cache.remove_peer(&peer_id);
        tracing::info!("Peer expired from mDNS: {}", peer_id);
        Ok(())
    }
    
    /// Add discovered peer
    async fn add_discovered_peer(&mut self, peer_id: PeerId, addresses: Vec<Multiaddr>) -> Result<()> {
        let peer_info = PeerInfo {
            peer_id,
            addresses: addresses.clone(),
            validator_id: None, // Will be determined later
            capabilities: PeerCapabilities::default(),
            geographic_location: None, // Will be determined later
            last_seen: chrono::Utc::now(),
        };
        
        self.discovered_peers.insert(peer_id, peer_info);
        self.discovery_cache.add_peer(peer_id, addresses);
        
        tracing::debug!("Added discovered peer: {}", peer_id);
        
        Ok(())
    }
    
    /// Get peer addresses
    pub async fn get_peer_addresses(&self, peer_id: &PeerId) -> Result<Vec<Multiaddr>> {
        if let Some(addresses) = self.discovery_cache.peer_addresses.get(peer_id) {
            Ok(addresses.clone())
        } else if let Some(peer_info) = self.discovered_peers.get(peer_id) {
            Ok(peer_info.addresses.clone())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Find peers in geographic region
    pub async fn find_peers_in_region(&self, region: &GeographicRegion) -> Result<Vec<PeerId>> {
        if let Some(peers) = self.discovery_cache.geographic_peers.get(region) {
            Ok(peers.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Find validator peer
    pub async fn find_validator_peer(&self, validator_id: &qross_consensus::ValidatorId) -> Result<Option<PeerId>> {
        Ok(self.discovery_cache.validator_peers.get(validator_id).cloned())
    }
    
    /// Perform peer discovery
    pub async fn discover_peers(&mut self) -> Result<()> {
        // Bootstrap Kademlia if available
        if let Some(kademlia) = &mut self.kademlia {
            if let Err(e) = kademlia.bootstrap() {
                tracing::warn!("Kademlia bootstrap failed: {}", e);
            }
        }
        
        // Perform periodic discovery
        self.periodic_discovery().await?;
        
        Ok(())
    }
    
    /// Periodic discovery routine
    async fn periodic_discovery(&mut self) -> Result<()> {
        // Clean expired cache entries
        self.discovery_cache.cleanup_expired();
        
        // Query for random peers to maintain connectivity
        if let Some(kademlia) = &mut self.kademlia {
            let random_peer_id = PeerId::random();
            let _ = kademlia.get_closest_peers(random_peer_id);
        }
        
        Ok(())
    }
    
    /// Extract peer ID from multiaddr
    fn extract_peer_id_from_multiaddr(&self, addr: &Multiaddr) -> Option<PeerId> {
        // TODO: Implement proper peer ID extraction from multiaddr
        None
    }
    
    /// Get local peer ID
    fn get_local_peer_id(&self) -> PeerId {
        // TODO: Get actual local peer ID
        PeerId::random()
    }
    
    /// Get discovery statistics
    pub fn get_discovery_statistics(&self) -> DiscoveryStatistics {
        DiscoveryStatistics {
            discovered_peers: self.discovered_peers.len(),
            bootstrap_peers: self.bootstrap_peers.len(),
            cached_addresses: self.discovery_cache.peer_addresses.len(),
            geographic_regions: self.discovery_cache.geographic_peers.len(),
            validator_peers: self.discovery_cache.validator_peers.len(),
        }
    }
}

impl DiscoveryCache {
    fn new() -> Self {
        Self {
            peer_addresses: HashMap::new(),
            geographic_peers: HashMap::new(),
            validator_peers: HashMap::new(),
            cache_expiry: HashMap::new(),
        }
    }
    
    fn add_peer(&mut self, peer_id: PeerId, addresses: Vec<Multiaddr>) {
        self.peer_addresses.insert(peer_id, addresses);
        self.cache_expiry.insert(peer_id, chrono::Utc::now() + chrono::Duration::hours(1));
    }
    
    fn remove_peer(&mut self, peer_id: &PeerId) {
        self.peer_addresses.remove(peer_id);
        self.cache_expiry.remove(peer_id);
        
        // Remove from geographic index
        for peers in self.geographic_peers.values_mut() {
            peers.remove(peer_id);
        }
        
        // Remove from validator index
        self.validator_peers.retain(|_, &mut v| v != *peer_id);
    }
    
    fn cleanup_expired(&mut self) {
        let now = chrono::Utc::now();
        let expired_peers: Vec<PeerId> = self.cache_expiry.iter()
            .filter(|(_, &expiry)| expiry < now)
            .map(|(&peer_id, _)| peer_id)
            .collect();
        
        for peer_id in expired_peers {
            self.remove_peer(&peer_id);
        }
    }
}

impl Default for PeerCapabilities {
    fn default() -> Self {
        Self {
            supports_quic: true,
            supports_relay: true,
            max_bandwidth: 100 * 1024 * 1024, // 100MB/s
            proof_verification: true,
            consensus_participation: false,
        }
    }
}

/// Discovery statistics
#[derive(Debug, Clone)]
pub struct DiscoveryStatistics {
    pub discovered_peers: usize,
    pub bootstrap_peers: usize,
    pub cached_addresses: usize,
    pub geographic_regions: usize,
    pub validator_peers: usize,
}
