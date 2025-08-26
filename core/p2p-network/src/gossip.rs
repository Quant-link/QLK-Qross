//! Gossip protocol for efficient message broadcasting

use crate::{types::*, error::*};
use libp2p::{PeerId, gossipsub::{Gossipsub, GossipsubEvent, MessageAuthenticity, ValidationMode}};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Gossip protocol implementation for message broadcasting
pub struct GossipProtocol {
    config: GossipConfig,
    gossipsub: Option<Gossipsub>,
    message_cache: HashMap<MessageId, GossipMessage>,
    peer_scores: HashMap<PeerId, PeerScore>,
    topic_subscriptions: HashMap<String, HashSet<PeerId>>,
    message_history: VecDeque<MessageHistoryEntry>,
    bandwidth_tracker: BandwidthTracker,
}

/// Gossip message wrapper
#[derive(Debug, Clone)]
pub struct GossipMessage {
    pub id: MessageId,
    pub content: NetworkMessage,
    pub topic: String,
    pub sender: PeerId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: u32,
    pub priority: MessagePriority,
}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Peer scoring for gossip optimization
#[derive(Debug, Clone)]
pub struct PeerScore {
    pub reliability: f64,
    pub latency: f64,
    pub bandwidth: f64,
    pub message_delivery_rate: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Message history entry
#[derive(Debug, Clone)]
pub struct MessageHistoryEntry {
    pub message_id: MessageId,
    pub sender: PeerId,
    pub recipients: HashSet<PeerId>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub delivery_confirmations: HashSet<PeerId>,
}

/// Bandwidth tracking for gossip optimization
pub struct BandwidthTracker {
    peer_bandwidth_usage: HashMap<PeerId, BandwidthUsage>,
    total_bandwidth_limit: u64,
    current_bandwidth_usage: u64,
    measurement_window: Duration,
}

impl GossipProtocol {
    /// Create a new gossip protocol
    pub fn new(config: GossipConfig) -> Self {
        Self {
            config,
            gossipsub: None,
            message_cache: HashMap::new(),
            peer_scores: HashMap::new(),
            topic_subscriptions: HashMap::new(),
            message_history: VecDeque::new(),
            bandwidth_tracker: BandwidthTracker::new(100 * 1024 * 1024), // 100MB/s default
        }
    }
    
    /// Start gossip protocol
    pub async fn start(&mut self) -> Result<()> {
        // Initialize gossipsub
        let gossipsub_config = libp2p::gossipsub::GossipsubConfigBuilder::default()
            .heartbeat_interval(Duration::from_millis(self.config.heartbeat_interval))
            .validation_mode(ValidationMode::Strict)
            .message_id_fn(|message| {
                // Custom message ID function
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                message.data.hash(&mut hasher);
                hasher.finish().to_string()
            })
            .max_transmit_size(1024 * 1024) // 1MB max message size
            .duplicate_cache_time(Duration::from_secs(60))
            .build()
            .map_err(|e| NetworkError::GossipError(format!("Failed to build gossipsub config: {}", e)))?;
        
        let gossipsub = Gossipsub::new(
            MessageAuthenticity::Signed(libp2p::identity::Keypair::generate_ed25519()),
            gossipsub_config,
        ).map_err(|e| NetworkError::GossipError(format!("Failed to create gossipsub: {}", e)))?;
        
        self.gossipsub = Some(gossipsub);
        
        tracing::info!("Gossip protocol started");
        
        Ok(())
    }
    
    /// Stop gossip protocol
    pub async fn stop(&mut self) -> Result<()> {
        self.gossipsub = None;
        self.message_cache.clear();
        self.message_history.clear();
        
        tracing::info!("Gossip protocol stopped");
        
        Ok(())
    }
    
    /// Broadcast message to network
    pub async fn broadcast_message(&mut self, message: NetworkMessage, targets: &[PeerId]) -> Result<()> {
        let gossip_message = GossipMessage {
            id: uuid::Uuid::new_v4(),
            content: message.clone(),
            topic: self.determine_message_topic(&message),
            sender: self.get_local_peer_id(),
            timestamp: chrono::Utc::now(),
            ttl: 10, // Default TTL
            priority: self.determine_message_priority(&message),
        };
        
        // Check bandwidth limits
        self.bandwidth_tracker.check_bandwidth_limit(&gossip_message)?;
        
        // Select optimal peers for broadcasting
        let selected_peers = self.select_broadcast_peers(targets, &gossip_message).await?;
        
        // Serialize message
        let serialized = self.serialize_gossip_message(&gossip_message)?;
        
        // Broadcast using gossipsub
        if let Some(gossipsub) = &mut self.gossipsub {
            let topic = libp2p::gossipsub::IdentTopic::new(&gossip_message.topic);
            gossipsub.publish(topic, serialized)
                .map_err(|e| NetworkError::GossipError(format!("Failed to publish message: {}", e)))?;
        }
        
        // Cache message
        self.message_cache.insert(gossip_message.id, gossip_message.clone());
        
        // Record in history
        self.record_message_history(gossip_message, selected_peers);
        
        tracing::debug!("Broadcast message {} to {} peers", gossip_message.id, targets.len());
        
        Ok(())
    }
    
    /// Distribute proof using gossip
    pub async fn distribute_proof(&mut self, message: NetworkMessage) -> Result<()> {
        let all_peers = self.get_all_connected_peers().await?;
        self.broadcast_message(message, &all_peers).await
    }
    
    /// Request proof from network
    pub async fn request_proof(&mut self, request: NetworkMessage) -> Result<Option<qross_proof_aggregation::AggregatedProof>> {
        // Broadcast proof request
        let all_peers = self.get_all_connected_peers().await?;
        self.broadcast_message(request, &all_peers).await?;
        
        // Wait for responses (simplified implementation)
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // TODO: Implement actual proof response handling
        Ok(None)
    }
    
    /// Subscribe to topic
    pub async fn subscribe_to_topic(&mut self, topic: &str) -> Result<()> {
        if let Some(gossipsub) = &mut self.gossipsub {
            let topic = libp2p::gossipsub::IdentTopic::new(topic);
            gossipsub.subscribe(&topic)
                .map_err(|e| NetworkError::GossipError(format!("Failed to subscribe to topic: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Unsubscribe from topic
    pub async fn unsubscribe_from_topic(&mut self, topic: &str) -> Result<()> {
        if let Some(gossipsub) = &mut self.gossipsub {
            let topic = libp2p::gossipsub::IdentTopic::new(topic);
            gossipsub.unsubscribe(&topic)
                .map_err(|e| NetworkError::GossipError(format!("Failed to unsubscribe from topic: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Handle incoming gossip event
    pub async fn handle_gossip_event(&mut self, event: GossipsubEvent) -> Result<()> {
        match event {
            GossipsubEvent::Message {
                propagation_source,
                message_id,
                message,
            } => {
                self.handle_incoming_message(propagation_source, message_id, message).await?;
            }
            GossipsubEvent::Subscribed { peer_id, topic } => {
                self.handle_peer_subscribed(peer_id, topic).await?;
            }
            GossipsubEvent::Unsubscribed { peer_id, topic } => {
                self.handle_peer_unsubscribed(peer_id, topic).await?;
            }
            GossipsubEvent::GossipsubNotSupported { peer_id } => {
                tracing::warn!("Peer {} does not support gossipsub", peer_id);
            }
        }
        
        Ok(())
    }
    
    /// Handle incoming message
    async fn handle_incoming_message(
        &mut self,
        source: PeerId,
        message_id: libp2p::gossipsub::MessageId,
        message: libp2p::gossipsub::Message,
    ) -> Result<()> {
        // Deserialize gossip message
        let gossip_message = self.deserialize_gossip_message(&message.data)?;
        
        // Check if message is duplicate
        if self.message_cache.contains_key(&gossip_message.id) {
            return Ok(());
        }
        
        // Validate message
        if !self.validate_message(&gossip_message, &source).await? {
            tracing::warn!("Invalid message {} from peer {}", gossip_message.id, source);
            return Ok(());
        }
        
        // Update peer score
        self.update_peer_score(&source, true).await?;
        
        // Cache message
        self.message_cache.insert(gossip_message.id, gossip_message.clone());
        
        // Process message based on type
        self.process_incoming_message(gossip_message).await?;
        
        Ok(())
    }
    
    /// Handle peer subscription
    async fn handle_peer_subscribed(&mut self, peer_id: PeerId, topic: libp2p::gossipsub::TopicHash) -> Result<()> {
        let topic_str = topic.to_string();
        self.topic_subscriptions.entry(topic_str.clone())
            .or_insert_with(HashSet::new)
            .insert(peer_id);
        
        tracing::debug!("Peer {} subscribed to topic {}", peer_id, topic_str);
        
        Ok(())
    }
    
    /// Handle peer unsubscription
    async fn handle_peer_unsubscribed(&mut self, peer_id: PeerId, topic: libp2p::gossipsub::TopicHash) -> Result<()> {
        let topic_str = topic.to_string();
        if let Some(subscribers) = self.topic_subscriptions.get_mut(&topic_str) {
            subscribers.remove(&peer_id);
        }
        
        tracing::debug!("Peer {} unsubscribed from topic {}", peer_id, topic_str);
        
        Ok(())
    }
    
    /// Select optimal peers for broadcasting
    async fn select_broadcast_peers(&self, targets: &[PeerId], message: &GossipMessage) -> Result<Vec<PeerId>> {
        let mut selected_peers = Vec::new();
        
        // Sort targets by peer score
        let mut scored_targets: Vec<_> = targets.iter()
            .filter_map(|peer_id| {
                self.peer_scores.get(peer_id).map(|score| (*peer_id, score))
            })
            .collect();
        
        scored_targets.sort_by(|a, b| {
            let score_a = self.calculate_broadcast_score(a.1, message);
            let score_b = self.calculate_broadcast_score(b.1, message);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Select top peers based on fanout
        let fanout = std::cmp::min(self.config.fanout, scored_targets.len());
        for (peer_id, _) in scored_targets.into_iter().take(fanout) {
            selected_peers.push(peer_id);
        }
        
        Ok(selected_peers)
    }
    
    /// Calculate broadcast score for peer
    fn calculate_broadcast_score(&self, peer_score: &PeerScore, message: &GossipMessage) -> f64 {
        let mut score = peer_score.reliability * 0.4 + peer_score.message_delivery_rate * 0.3;
        
        // Adjust for latency (lower is better)
        score += (1.0 - peer_score.latency) * 0.2;
        
        // Adjust for bandwidth
        score += peer_score.bandwidth * 0.1;
        
        // Priority adjustment
        match message.priority {
            MessagePriority::Critical => score * 1.5,
            MessagePriority::High => score * 1.2,
            MessagePriority::Normal => score,
            MessagePriority::Low => score * 0.8,
        }
    }
    
    /// Validate incoming message
    async fn validate_message(&self, message: &GossipMessage, source: &PeerId) -> Result<bool> {
        // Check TTL
        if message.ttl == 0 {
            return Ok(false);
        }
        
        // Check timestamp (not too old or too far in future)
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(message.timestamp);
        if age.num_seconds() > 300 || age.num_seconds() < -60 {
            return Ok(false);
        }
        
        // Check sender reputation
        if let Some(peer_score) = self.peer_scores.get(source) {
            if peer_score.reliability < 0.3 {
                return Ok(false);
            }
        }
        
        // TODO: Add cryptographic signature validation
        
        Ok(true)
    }
    
    /// Update peer score based on behavior
    async fn update_peer_score(&mut self, peer_id: &PeerId, positive: bool) -> Result<()> {
        let score = self.peer_scores.entry(*peer_id).or_insert_with(|| PeerScore {
            reliability: 0.5,
            latency: 0.5,
            bandwidth: 0.5,
            message_delivery_rate: 0.5,
            last_updated: chrono::Utc::now(),
        });
        
        if positive {
            score.reliability = (score.reliability + 0.1).min(1.0);
            score.message_delivery_rate = (score.message_delivery_rate + 0.05).min(1.0);
        } else {
            score.reliability = (score.reliability - 0.1).max(0.0);
            score.message_delivery_rate = (score.message_delivery_rate - 0.05).max(0.0);
        }
        
        score.last_updated = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Process incoming message
    async fn process_incoming_message(&mut self, message: GossipMessage) -> Result<()> {
        match &message.content {
            NetworkMessage::ProofDistribution { .. } => {
                // Handle proof distribution
                tracing::debug!("Received proof distribution message");
            }
            NetworkMessage::ProofRequest { .. } => {
                // Handle proof request
                tracing::debug!("Received proof request message");
            }
            NetworkMessage::ConsensusMessage { .. } => {
                // Handle consensus message
                tracing::debug!("Received consensus message");
            }
            _ => {
                tracing::debug!("Received other message type");
            }
        }
        
        Ok(())
    }
    
    /// Record message in history
    fn record_message_history(&mut self, message: GossipMessage, recipients: Vec<PeerId>) {
        let history_entry = MessageHistoryEntry {
            message_id: message.id,
            sender: message.sender,
            recipients: recipients.into_iter().collect(),
            timestamp: message.timestamp,
            delivery_confirmations: HashSet::new(),
        };
        
        self.message_history.push_back(history_entry);
        
        // Maintain history size
        if self.message_history.len() > self.config.history_length {
            self.message_history.pop_front();
        }
    }
    
    /// Determine message topic
    fn determine_message_topic(&self, message: &NetworkMessage) -> String {
        match message {
            NetworkMessage::ProofDistribution { .. } => "proofs".to_string(),
            NetworkMessage::ProofRequest { .. } => "proof_requests".to_string(),
            NetworkMessage::ConsensusMessage { .. } => "consensus".to_string(),
            NetworkMessage::Discovery { .. } => "discovery".to_string(),
            NetworkMessage::Heartbeat { .. } => "heartbeat".to_string(),
            NetworkMessage::Relay { .. } => "relay".to_string(),
        }
    }
    
    /// Determine message priority
    fn determine_message_priority(&self, message: &NetworkMessage) -> MessagePriority {
        match message {
            NetworkMessage::ConsensusMessage { .. } => MessagePriority::Critical,
            NetworkMessage::ProofDistribution { .. } => MessagePriority::High,
            NetworkMessage::ProofRequest { .. } => MessagePriority::Normal,
            NetworkMessage::Discovery { .. } => MessagePriority::Low,
            NetworkMessage::Heartbeat { .. } => MessagePriority::Low,
            NetworkMessage::Relay { .. } => MessagePriority::Normal,
        }
    }
    
    /// Serialize gossip message
    fn serialize_gossip_message(&self, message: &GossipMessage) -> Result<Vec<u8>> {
        bincode::serialize(message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))
    }
    
    /// Deserialize gossip message
    fn deserialize_gossip_message(&self, data: &[u8]) -> Result<GossipMessage> {
        bincode::deserialize(data)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))
    }
    
    /// Get all connected peers
    async fn get_all_connected_peers(&self) -> Result<Vec<PeerId>> {
        // TODO: Get actual connected peers from network manager
        Ok(self.peer_scores.keys().cloned().collect())
    }
    
    /// Get local peer ID
    fn get_local_peer_id(&self) -> PeerId {
        // TODO: Get actual local peer ID
        PeerId::random()
    }
    
    /// Get gossip statistics
    pub fn get_gossip_statistics(&self) -> GossipStatistics {
        GossipStatistics {
            cached_messages: self.message_cache.len(),
            known_peers: self.peer_scores.len(),
            topic_subscriptions: self.topic_subscriptions.len(),
            message_history_size: self.message_history.len(),
            bandwidth_usage: self.bandwidth_tracker.current_bandwidth_usage,
        }
    }
}

impl BandwidthTracker {
    fn new(total_limit: u64) -> Self {
        Self {
            peer_bandwidth_usage: HashMap::new(),
            total_bandwidth_limit: total_limit,
            current_bandwidth_usage: 0,
            measurement_window: Duration::from_secs(60),
        }
    }
    
    fn check_bandwidth_limit(&self, message: &GossipMessage) -> Result<()> {
        let message_size = bincode::serialized_size(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))? as u64;
        
        if self.current_bandwidth_usage + message_size > self.total_bandwidth_limit {
            return Err(NetworkError::BandwidthLimitExceeded {
                used: self.current_bandwidth_usage + message_size,
                limit: self.total_bandwidth_limit,
            });
        }
        
        Ok(())
    }
}

/// Gossip protocol statistics
#[derive(Debug, Clone)]
pub struct GossipStatistics {
    pub cached_messages: usize,
    pub known_peers: usize,
    pub topic_subscriptions: usize,
    pub message_history_size: usize,
    pub bandwidth_usage: u64,
}
