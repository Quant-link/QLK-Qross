//! Core types for P2P network stack

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use libp2p::{PeerId, Multiaddr};
use qross_consensus::ValidatorId;
use qross_proof_aggregation::ProofId;

/// Message identifier
pub type MessageId = Uuid;

/// Network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    /// Proof distribution message
    ProofDistribution {
        proof_id: ProofId,
        proof_data: Vec<u8>,
        distribution_strategy: ProofDistributionStrategy,
        timestamp: DateTime<Utc>,
    },
    
    /// Proof request message
    ProofRequest {
        proof_id: ProofId,
        requester: PeerId,
        timestamp: DateTime<Utc>,
    },
    
    /// Proof response message
    ProofResponse {
        proof_id: ProofId,
        proof_data: Option<Vec<u8>>,
        responder: PeerId,
        timestamp: DateTime<Utc>,
    },
    
    /// Consensus message
    ConsensusMessage {
        message_type: ConsensusMessageType,
        payload: Vec<u8>,
        sender: ValidatorId,
        timestamp: DateTime<Utc>,
    },
    
    /// Discovery message
    Discovery {
        peer_info: PeerInfo,
        timestamp: DateTime<Utc>,
    },
    
    /// Heartbeat message
    Heartbeat {
        peer_id: PeerId,
        timestamp: DateTime<Utc>,
    },
    
    /// Relay message
    Relay {
        target: PeerId,
        payload: Box<NetworkMessage>,
        hop_count: u32,
        timestamp: DateTime<Utc>,
    },
}

/// Proof distribution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofDistributionStrategy {
    /// Use gossip protocol for distribution
    Gossip,
    
    /// Direct routing to specific targets
    DirectRouting { targets: Vec<PeerId> },
    
    /// Hybrid approach combining gossip and direct routing
    Hybrid {
        gossip_targets: Vec<PeerId>,
        direct_targets: Vec<PeerId>,
    },
}

/// Consensus message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessageType {
    Proposal,
    Vote,
    Commit,
    ViewChange,
}

/// Peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub validator_id: Option<ValidatorId>,
    pub capabilities: PeerCapabilities,
    pub geographic_location: Option<GeographicLocation>,
    pub last_seen: DateTime<Utc>,
}

/// Peer capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerCapabilities {
    pub supports_quic: bool,
    pub supports_relay: bool,
    pub max_bandwidth: u64,
    pub proof_verification: bool,
    pub consensus_participation: bool,
}

/// Geographic location for latency optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicLocation {
    pub region: GeographicRegion,
    pub latitude: f64,
    pub longitude: f64,
    pub city: Option<String>,
    pub country: String,
}

/// Geographic regions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GeographicRegion {
    NorthAmerica,
    SouthAmerica,
    Europe,
    Asia,
    Africa,
    Oceania,
}

/// Connection information
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    pub peer_id: PeerId,
    pub address: Multiaddr,
    pub connection_type: ConnectionType,
    pub established_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub bandwidth_usage: BandwidthUsage,
    pub latency_stats: LatencyStats,
}

/// Connection types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionType {
    Quic,
    Tcp,
    Relay,
    WebRTC,
}

/// Bandwidth usage tracking
#[derive(Debug, Clone, Default)]
pub struct BandwidthUsage {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub last_reset: DateTime<Utc>,
}

/// Latency statistics
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    pub min_latency: f64,
    pub max_latency: f64,
    pub average_latency: f64,
    pub sample_count: u64,
}

/// Cached message
#[derive(Debug, Clone)]
pub struct CachedMessage {
    pub message: NetworkMessage,
    pub cached_at: DateTime<Utc>,
    pub access_count: u64,
    pub size: usize,
}

/// Broadcast targets
#[derive(Debug, Clone)]
pub enum BroadcastTargets {
    All,
    Validators,
    Specific(Vec<PeerId>),
    Geographic(GeographicRegion),
}

/// Network topology information
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    pub nodes: HashMap<PeerId, NodeInfo>,
    pub edges: Vec<NetworkEdge>,
    pub clusters: Vec<NetworkCluster>,
    pub diameter: u32,
    pub average_path_length: f64,
}

/// Node information in topology
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub peer_id: PeerId,
    pub degree: u32,
    pub betweenness_centrality: f64,
    pub clustering_coefficient: f64,
    pub geographic_location: Option<GeographicLocation>,
}

/// Network edge between nodes
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    pub source: PeerId,
    pub target: PeerId,
    pub weight: f64,
    pub latency: f64,
    pub bandwidth: u64,
}

/// Network cluster
#[derive(Debug, Clone)]
pub struct NetworkCluster {
    pub cluster_id: Uuid,
    pub nodes: Vec<PeerId>,
    pub center: PeerId,
    pub radius: f64,
}

/// Routing path through network
#[derive(Debug, Clone)]
pub struct RoutingPath {
    pub source: PeerId,
    pub target: PeerId,
    pub hops: Vec<PeerId>,
    pub total_latency: f64,
    pub total_cost: f64,
    pub reliability: f64,
}

/// Validator network information
#[derive(Debug, Clone)]
pub struct ValidatorNetworkInfo {
    pub validator_id: ValidatorId,
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub reputation_score: f64,
    pub connectivity_status: ConnectivityStatus,
    pub geographic_location: GeographicLocation,
}

/// Connectivity status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityStatus {
    Connected,
    Disconnected,
    Connecting,
    Error(String),
}

/// Batch distribution information
#[derive(Debug, Clone)]
pub struct BatchDistributionInfo {
    pub batch_id: Uuid,
    pub proof_ids: Vec<ProofId>,
    pub target_validators: Vec<ValidatorId>,
    pub priority: DistributionPriority,
    pub deadline: Option<DateTime<Utc>>,
}

/// Distribution priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DistributionPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub active_connections: usize,
    pub cached_messages: usize,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub average_latency: f64,
    pub bandwidth_utilization: f64,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub network_config: NetworkManagerConfig,
    pub transport_config: TransportConfig,
    pub routing_config: RoutingConfig,
    pub gossip_config: GossipConfig,
    pub gossip_optimization_config: GossipOptimizationConfig,
    pub discovery_config: DiscoveryConfig,
    pub relay_config: RelayConfig,
    pub security_config: SecurityConfig,
    pub bandwidth_config: BandwidthConfig,
    pub topology_config: TopologyConfig,
}

/// Network manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkManagerConfig {
    pub listen_addresses: Vec<String>,
    pub external_addresses: Vec<String>,
    pub max_connections: usize,
    pub connection_timeout: u64,
    pub heartbeat_interval: u64,
    pub enable_mdns: bool,
    pub enable_upnp: bool,
}

/// Transport layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub enable_quic: bool,
    pub enable_tcp: bool,
    pub quic_config: QuicConfig,
    pub tcp_config: TcpConfig,
    pub noise_config: NoiseConfig,
}

/// QUIC protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuicConfig {
    pub max_concurrent_streams: u32,
    pub max_idle_timeout: u64,
    pub keep_alive_interval: u64,
    pub max_packet_size: u32,
    pub enable_0rtt: bool,
}

/// TCP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpConfig {
    pub nodelay: bool,
    pub keepalive: bool,
    pub keepalive_interval: u64,
    pub buffer_size: usize,
}

/// Noise protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub keypair_path: Option<String>,
    pub generate_keypair: bool,
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub algorithm: RoutingAlgorithm,
    pub max_hops: u32,
    pub route_cache_size: usize,
    pub route_cache_ttl: u64,
    pub enable_shortest_path: bool,
    pub enable_load_balancing: bool,
}

/// Routing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    Dijkstra,
    AStar,
    FloydWarshall,
    Gossip,
    Hybrid,
}

/// Gossip protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipConfig {
    pub fanout: usize,
    pub heartbeat_interval: u64,
    pub history_length: usize,
    pub history_gossip: usize,
    pub mesh_n: usize,
    pub mesh_n_low: usize,
    pub mesh_n_high: usize,
    pub gossip_lazy: usize,
}

/// Discovery service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub enable_mdns: bool,
    pub enable_kad: bool,
    pub bootstrap_peers: Vec<String>,
    pub kad_replication_factor: usize,
    pub discovery_interval: u64,
}

/// Relay service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayConfig {
    pub enable_relay: bool,
    pub max_relay_connections: usize,
    pub relay_timeout: u64,
    pub enable_dcutr: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_encryption: bool,
    pub enable_authentication: bool,
    pub rate_limit_config: RateLimitConfig,
    pub blacklist_config: BlacklistConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub messages_per_second: u32,
    pub bytes_per_second: u64,
    pub burst_size: u32,
    pub window_size: u64,
}

/// Blacklist configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlacklistConfig {
    pub enable_blacklist: bool,
    pub blacklist_duration: u64,
    pub max_violations: u32,
    pub violation_window: u64,
}

/// Bandwidth management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConfig {
    pub max_bandwidth: u64,
    pub priority_allocation: HashMap<String, f64>,
    pub congestion_control: CongestionControlConfig,
    pub compression_config: CompressionConfig,
}

/// Congestion control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionControlConfig {
    pub algorithm: CongestionControlAlgorithm,
    pub initial_window: u32,
    pub max_window: u32,
    pub slow_start_threshold: u32,
}

/// Congestion control algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionControlAlgorithm {
    Reno,
    Cubic,
    BBR,
    Vegas,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enable_compression: bool,
    pub algorithm: CompressionAlgorithm,
    pub compression_level: u32,
    pub min_size_threshold: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    LZ4,
    Zstd,
    Gzip,
    Brotli,
}

/// Topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    pub monitoring_interval: u64,
    pub health_check_timeout: u64,
    pub rebalancing_cooldown: u64,
    pub enable_auto_rebalancing: bool,
    pub max_concurrent_rebalancing: usize,
    pub degradation_thresholds: TopologyDegradationThresholds,
}

/// Topology degradation thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyDegradationThresholds {
    pub latency_increase_threshold: f64,
    pub bandwidth_decrease_threshold: f64,
    pub reliability_decrease_threshold: f64,
    pub delivery_rate_threshold: f64,
}

/// Gossip optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipOptimizationConfig {
    pub enable_deduplication: bool,
    pub enable_compression: bool,
    pub enable_batching: bool,
    pub enable_topic_optimization: bool,
    pub enable_proof_distribution_optimization: bool,
    pub bloom_filter_size: usize,
    pub false_positive_rate: f64,
    pub message_cache_size: usize,
    pub default_ttl_seconds: u64,
    pub compression_algorithms: Vec<String>,
    pub batch_size_threshold: usize,
    pub batch_timeout_ms: u64,
    pub optimization_interval_seconds: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            network_config: NetworkManagerConfig::default(),
            transport_config: TransportConfig::default(),
            routing_config: RoutingConfig::default(),
            gossip_config: GossipConfig::default(),
            gossip_optimization_config: GossipOptimizationConfig::default(),
            discovery_config: DiscoveryConfig::default(),
            relay_config: RelayConfig::default(),
            security_config: SecurityConfig::default(),
            bandwidth_config: BandwidthConfig::default(),
            topology_config: TopologyConfig::default(),
        }
    }
}

impl Default for NetworkManagerConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec!["/ip4/0.0.0.0/tcp/0".to_string()],
            external_addresses: Vec::new(),
            max_connections: 1000,
            connection_timeout: 30,
            heartbeat_interval: 30,
            enable_mdns: true,
            enable_upnp: false,
        }
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            enable_quic: true,
            enable_tcp: true,
            quic_config: QuicConfig::default(),
            tcp_config: TcpConfig::default(),
            noise_config: NoiseConfig::default(),
        }
    }
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 100,
            max_idle_timeout: 30000,
            keep_alive_interval: 5000,
            max_packet_size: 1452,
            enable_0rtt: true,
        }
    }
}

impl Default for TcpConfig {
    fn default() -> Self {
        Self {
            nodelay: true,
            keepalive: true,
            keepalive_interval: 30,
            buffer_size: 65536,
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            keypair_path: None,
            generate_keypair: true,
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            algorithm: RoutingAlgorithm::Hybrid,
            max_hops: 10,
            route_cache_size: 1000,
            route_cache_ttl: 300,
            enable_shortest_path: true,
            enable_load_balancing: true,
        }
    }
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 6,
            heartbeat_interval: 1000,
            history_length: 5,
            history_gossip: 3,
            mesh_n: 6,
            mesh_n_low: 4,
            mesh_n_high: 12,
            gossip_lazy: 6,
        }
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_mdns: true,
            enable_kad: true,
            bootstrap_peers: Vec::new(),
            kad_replication_factor: 20,
            discovery_interval: 60,
        }
    }
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            enable_relay: true,
            max_relay_connections: 100,
            relay_timeout: 30,
            enable_dcutr: true,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            enable_authentication: true,
            rate_limit_config: RateLimitConfig::default(),
            blacklist_config: BlacklistConfig::default(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            messages_per_second: 100,
            bytes_per_second: 1024 * 1024, // 1MB/s
            burst_size: 10,
            window_size: 60,
        }
    }
}

impl Default for BlacklistConfig {
    fn default() -> Self {
        Self {
            enable_blacklist: true,
            blacklist_duration: 3600, // 1 hour
            max_violations: 5,
            violation_window: 300, // 5 minutes
        }
    }
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            max_bandwidth: 100 * 1024 * 1024, // 100MB/s
            priority_allocation: HashMap::new(),
            congestion_control: CongestionControlConfig::default(),
            compression_config: CompressionConfig::default(),
        }
    }
}

impl Default for CongestionControlConfig {
    fn default() -> Self {
        Self {
            algorithm: CongestionControlAlgorithm::Cubic,
            initial_window: 10,
            max_window: 1000,
            slow_start_threshold: 100,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            algorithm: CompressionAlgorithm::LZ4,
            compression_level: 3,
            min_size_threshold: 1024,
        }
    }
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: 30,
            health_check_timeout: 10,
            rebalancing_cooldown: 300,
            enable_auto_rebalancing: true,
            max_concurrent_rebalancing: 2,
            degradation_thresholds: TopologyDegradationThresholds::default(),
        }
    }
}

impl Default for TopologyDegradationThresholds {
    fn default() -> Self {
        Self {
            latency_increase_threshold: 2.0,
            bandwidth_decrease_threshold: 0.5,
            reliability_decrease_threshold: 0.1,
            delivery_rate_threshold: 0.95,
        }
    }
}

impl Default for GossipOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            enable_compression: true,
            enable_batching: true,
            enable_topic_optimization: true,
            enable_proof_distribution_optimization: true,
            bloom_filter_size: 1000000, // 1M elements
            false_positive_rate: 0.01, // 1%
            message_cache_size: 10000,
            default_ttl_seconds: 300, // 5 minutes
            compression_algorithms: vec!["lz4".to_string(), "zstd".to_string()],
            batch_size_threshold: 10,
            batch_timeout_ms: 100, // 100ms
            optimization_interval_seconds: 60, // 1 minute
        }
    }
}

impl From<ProofId> for MessageId {
    fn from(proof_id: ProofId) -> Self {
        proof_id
    }
}
