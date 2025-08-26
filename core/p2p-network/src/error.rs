//! Error types for P2P network stack

use thiserror::Error;
use libp2p::PeerId;

/// Result type alias
pub type Result<T> = std::result::Result<T, NetworkError>;

/// P2P network error types
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Peer not found
    #[error("Peer not found: {0}")]
    PeerNotFound(PeerId),
    
    /// Peer not connected
    #[error("Peer not connected: {0}")]
    PeerNotConnected(PeerId),
    
    /// Connection failed
    #[error("Connection failed to peer: {0}")]
    ConnectionFailed(PeerId),
    
    /// Route not found
    #[error("Route not found from {source} to {target}")]
    RouteNotFound { source: PeerId, target: PeerId },
    
    /// Transport error
    #[error("Transport error: {0}")]
    TransportError(String),
    
    /// Routing error
    #[error("Routing error: {0}")]
    RoutingError(String),
    
    /// Gossip protocol error
    #[error("Gossip protocol error: {0}")]
    GossipError(String),
    
    /// Discovery error
    #[error("Discovery error: {0}")]
    DiscoveryError(String),
    
    /// Relay error
    #[error("Relay error: {0}")]
    RelayError(String),
    
    /// Security error
    #[error("Security error: {0}")]
    SecurityError(String),
    
    /// Bandwidth limit exceeded
    #[error("Bandwidth limit exceeded: used {used}, limit {limit}")]
    BandwidthLimitExceeded { used: u64, limit: u64 },
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded for peer: {0}")]
    RateLimitExceeded(PeerId),
    
    /// Message too large
    #[error("Message too large: size {size}, limit {limit}")]
    MessageTooLarge { size: usize, limit: usize },
    
    /// Invalid message format
    #[error("Invalid message format: {0}")]
    InvalidMessageFormat(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),
    
    /// Encryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),
    
    /// Authentication failed
    #[error("Authentication failed for peer: {0}")]
    AuthenticationFailed(PeerId),
    
    /// Network timeout
    #[error("Network timeout")]
    Timeout,
    
    /// Network congestion
    #[error("Network congestion detected")]
    NetworkCongestion,
    
    /// Insufficient peers
    #[error("Insufficient peers: required {required}, available {available}")]
    InsufficientPeers { required: usize, available: usize },
    
    /// Protocol version mismatch
    #[error("Protocol version mismatch: local {local}, remote {remote}")]
    ProtocolVersionMismatch { local: String, remote: String },
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl NetworkError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            NetworkError::ConnectionFailed(_) => true,
            NetworkError::Timeout => true,
            NetworkError::NetworkCongestion => true,
            NetworkError::TransportError(_) => true,
            NetworkError::Io(_) => true,
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NetworkError::PeerNotFound(_) => ErrorSeverity::Medium,
            NetworkError::PeerNotConnected(_) => ErrorSeverity::Medium,
            NetworkError::ConnectionFailed(_) => ErrorSeverity::Medium,
            NetworkError::RouteNotFound { .. } => ErrorSeverity::Medium,
            NetworkError::TransportError(_) => ErrorSeverity::High,
            NetworkError::RoutingError(_) => ErrorSeverity::Medium,
            NetworkError::GossipError(_) => ErrorSeverity::Medium,
            NetworkError::DiscoveryError(_) => ErrorSeverity::Medium,
            NetworkError::RelayError(_) => ErrorSeverity::Medium,
            NetworkError::SecurityError(_) => ErrorSeverity::High,
            NetworkError::BandwidthLimitExceeded { .. } => ErrorSeverity::Medium,
            NetworkError::RateLimitExceeded(_) => ErrorSeverity::Low,
            NetworkError::MessageTooLarge { .. } => ErrorSeverity::Medium,
            NetworkError::InvalidMessageFormat(_) => ErrorSeverity::Medium,
            NetworkError::SerializationError(_) => ErrorSeverity::Medium,
            NetworkError::CompressionError(_) => ErrorSeverity::Medium,
            NetworkError::EncryptionError(_) => ErrorSeverity::High,
            NetworkError::AuthenticationFailed(_) => ErrorSeverity::High,
            NetworkError::Timeout => ErrorSeverity::Medium,
            NetworkError::NetworkCongestion => ErrorSeverity::Medium,
            NetworkError::InsufficientPeers { .. } => ErrorSeverity::High,
            NetworkError::ProtocolVersionMismatch { .. } => ErrorSeverity::High,
            NetworkError::ConfigurationError(_) => ErrorSeverity::High,
            NetworkError::Io(_) => ErrorSeverity::Medium,
            NetworkError::Internal(_) => ErrorSeverity::High,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            NetworkError::PeerNotFound(_) => "peer_not_found",
            NetworkError::PeerNotConnected(_) => "peer_not_connected",
            NetworkError::ConnectionFailed(_) => "connection_failed",
            NetworkError::RouteNotFound { .. } => "route_not_found",
            NetworkError::TransportError(_) => "transport_error",
            NetworkError::RoutingError(_) => "routing_error",
            NetworkError::GossipError(_) => "gossip_error",
            NetworkError::DiscoveryError(_) => "discovery_error",
            NetworkError::RelayError(_) => "relay_error",
            NetworkError::SecurityError(_) => "security_error",
            NetworkError::BandwidthLimitExceeded { .. } => "bandwidth_limit_exceeded",
            NetworkError::RateLimitExceeded(_) => "rate_limit_exceeded",
            NetworkError::MessageTooLarge { .. } => "message_too_large",
            NetworkError::InvalidMessageFormat(_) => "invalid_message_format",
            NetworkError::SerializationError(_) => "serialization_error",
            NetworkError::CompressionError(_) => "compression_error",
            NetworkError::EncryptionError(_) => "encryption_error",
            NetworkError::AuthenticationFailed(_) => "authentication_failed",
            NetworkError::Timeout => "timeout",
            NetworkError::NetworkCongestion => "network_congestion",
            NetworkError::InsufficientPeers { .. } => "insufficient_peers",
            NetworkError::ProtocolVersionMismatch { .. } => "protocol_version_mismatch",
            NetworkError::ConfigurationError(_) => "configuration_error",
            NetworkError::Io(_) => "io_error",
            NetworkError::Internal(_) => "internal_error",
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "low"),
            ErrorSeverity::Medium => write!(f, "medium"),
            ErrorSeverity::High => write!(f, "high"),
            ErrorSeverity::Critical => write!(f, "critical"),
        }
    }
}
