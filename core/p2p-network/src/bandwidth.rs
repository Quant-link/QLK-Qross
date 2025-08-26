//! Bandwidth management and congestion control

use crate::{types::*, error::*};
use libp2p::PeerId;
use std::collections::{HashMap, VecDeque};

/// Bandwidth manager for network optimization
pub struct BandwidthManager {
    config: BandwidthConfig,
    peer_bandwidth_usage: HashMap<PeerId, PeerBandwidthTracker>,
    global_bandwidth_tracker: GlobalBandwidthTracker,
    congestion_controller: CongestionController,
    compression_engine: CompressionEngine,
    priority_scheduler: PriorityScheduler,
}

/// Per-peer bandwidth tracking
pub struct PeerBandwidthTracker {
    pub peer_id: PeerId,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub current_rate: f64, // bytes per second
    pub average_rate: f64,
    pub peak_rate: f64,
    pub last_measurement: chrono::DateTime<chrono::Utc>,
    pub rate_history: VecDeque<RateMeasurement>,
}

/// Rate measurement for bandwidth tracking
#[derive(Debug, Clone)]
pub struct RateMeasurement {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub bytes_per_second: f64,
    pub messages_per_second: f64,
}

/// Global bandwidth tracking
pub struct GlobalBandwidthTracker {
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub current_utilization: f64,
    pub peak_utilization: f64,
    pub bandwidth_limit: u64,
    pub utilization_history: VecDeque<UtilizationMeasurement>,
}

/// Utilization measurement
#[derive(Debug, Clone)]
pub struct UtilizationMeasurement {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub utilization: f64,
    pub throughput: f64,
}

/// Congestion controller
pub struct CongestionController {
    config: CongestionControlConfig,
    congestion_window: u32,
    slow_start_threshold: u32,
    rtt_measurements: HashMap<PeerId, RttTracker>,
    congestion_state: CongestionState,
}

/// RTT (Round Trip Time) tracker
#[derive(Debug, Clone)]
pub struct RttTracker {
    pub current_rtt: f64,
    pub smoothed_rtt: f64,
    pub rtt_variance: f64,
    pub min_rtt: f64,
    pub max_rtt: f64,
    pub measurements: VecDeque<f64>,
}

/// Congestion control states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CongestionState {
    SlowStart,
    CongestionAvoidance,
    FastRecovery,
}

/// Compression engine for bandwidth optimization
pub struct CompressionEngine {
    config: CompressionConfig,
    compression_stats: CompressionStatistics,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStatistics {
    pub total_compressed_bytes: u64,
    pub total_uncompressed_bytes: u64,
    pub compression_ratio: f64,
    pub compression_time: f64,
    pub decompression_time: f64,
}

/// Priority scheduler for message prioritization
pub struct PriorityScheduler {
    priority_queues: HashMap<MessagePriority, VecDeque<QueuedMessage>>,
    bandwidth_allocation: HashMap<MessagePriority, f64>,
}

/// Queued message for priority scheduling
#[derive(Debug, Clone)]
pub struct QueuedMessage {
    pub message: NetworkMessage,
    pub peer_id: PeerId,
    pub size: usize,
    pub queued_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl BandwidthManager {
    /// Create a new bandwidth manager
    pub fn new(config: BandwidthConfig) -> Self {
        Self {
            peer_bandwidth_usage: HashMap::new(),
            global_bandwidth_tracker: GlobalBandwidthTracker::new(config.max_bandwidth),
            congestion_controller: CongestionController::new(config.congestion_control.clone()),
            compression_engine: CompressionEngine::new(config.compression_config.clone()),
            priority_scheduler: PriorityScheduler::new(&config.priority_allocation),
            config,
        }
    }
    
    /// Start bandwidth manager
    pub async fn start(&mut self) -> Result<()> {
        // Start periodic bandwidth monitoring
        self.start_bandwidth_monitoring().await?;
        
        tracing::info!("Bandwidth manager started with limit: {} MB/s", 
                      self.config.max_bandwidth / (1024 * 1024));
        
        Ok(())
    }
    
    /// Stop bandwidth manager
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Bandwidth manager stopped");
        Ok(())
    }
    
    /// Check if message can be sent within bandwidth limits
    pub async fn check_bandwidth_available(&mut self, peer_id: &PeerId, message_size: usize) -> Result<bool> {
        // Check global bandwidth limit
        if !self.global_bandwidth_tracker.can_send(message_size as u64) {
            return Ok(false);
        }
        
        // Check per-peer bandwidth
        let peer_tracker = self.peer_bandwidth_usage.entry(*peer_id)
            .or_insert_with(|| PeerBandwidthTracker::new(*peer_id));
        
        if !peer_tracker.can_send(message_size as u64) {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Record bandwidth usage
    pub async fn record_bandwidth_usage(&mut self, peer_id: &PeerId, bytes_sent: u64, bytes_received: u64) -> Result<()> {
        // Update peer tracker
        let peer_tracker = self.peer_bandwidth_usage.entry(*peer_id)
            .or_insert_with(|| PeerBandwidthTracker::new(*peer_id));
        
        peer_tracker.record_usage(bytes_sent, bytes_received);
        
        // Update global tracker
        self.global_bandwidth_tracker.record_usage(bytes_sent, bytes_received);
        
        // Update congestion control
        self.congestion_controller.update_bandwidth_usage(peer_id, bytes_sent).await?;
        
        Ok(())
    }
    
    /// Compress message if beneficial
    pub async fn compress_message(&mut self, message: &NetworkMessage) -> Result<Vec<u8>> {
        self.compression_engine.compress_message(message).await
    }
    
    /// Decompress message
    pub async fn decompress_message(&mut self, compressed_data: &[u8]) -> Result<NetworkMessage> {
        self.compression_engine.decompress_message(compressed_data).await
    }
    
    /// Schedule message based on priority
    pub async fn schedule_message(&mut self, message: NetworkMessage, peer_id: PeerId, priority: MessagePriority) -> Result<()> {
        self.priority_scheduler.schedule_message(message, peer_id, priority).await
    }
    
    /// Get next message to send
    pub async fn get_next_message(&mut self) -> Result<Option<QueuedMessage>> {
        self.priority_scheduler.get_next_message().await
    }
    
    /// Update RTT measurement
    pub async fn update_rtt(&mut self, peer_id: &PeerId, rtt: f64) -> Result<()> {
        self.congestion_controller.update_rtt(peer_id, rtt).await
    }
    
    /// Handle congestion signal
    pub async fn handle_congestion(&mut self, peer_id: &PeerId) -> Result<()> {
        self.congestion_controller.handle_congestion(peer_id).await
    }
    
    /// Get current bandwidth utilization
    pub fn get_current_utilization(&self) -> f64 {
        self.global_bandwidth_tracker.current_utilization
    }
    
    /// Get available bandwidth
    pub async fn get_available_bandwidth(&self) -> Result<f64> {
        Ok(1.0 - self.global_bandwidth_tracker.current_utilization)
    }
    
    /// Start bandwidth monitoring
    async fn start_bandwidth_monitoring(&mut self) -> Result<()> {
        // TODO: Implement periodic bandwidth monitoring
        Ok(())
    }
    
    /// Get bandwidth statistics
    pub fn get_bandwidth_statistics(&self) -> BandwidthStatistics {
        BandwidthStatistics {
            total_bytes_sent: self.global_bandwidth_tracker.total_bytes_sent,
            total_bytes_received: self.global_bandwidth_tracker.total_bytes_received,
            current_utilization: self.global_bandwidth_tracker.current_utilization,
            peak_utilization: self.global_bandwidth_tracker.peak_utilization,
            active_peers: self.peer_bandwidth_usage.len(),
            compression_ratio: self.compression_engine.compression_stats.compression_ratio,
            congestion_window: self.congestion_controller.congestion_window,
        }
    }
}

impl PeerBandwidthTracker {
    fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            current_rate: 0.0,
            average_rate: 0.0,
            peak_rate: 0.0,
            last_measurement: chrono::Utc::now(),
            rate_history: VecDeque::new(),
        }
    }
    
    fn record_usage(&mut self, bytes_sent: u64, bytes_received: u64) {
        self.bytes_sent += bytes_sent;
        self.bytes_received += bytes_received;
        
        let now = chrono::Utc::now();
        let duration = now.signed_duration_since(self.last_measurement);
        let duration_secs = duration.num_milliseconds() as f64 / 1000.0;
        
        if duration_secs > 0.0 {
            self.current_rate = (bytes_sent + bytes_received) as f64 / duration_secs;
            self.peak_rate = self.peak_rate.max(self.current_rate);
            
            // Update average rate
            self.average_rate = (self.average_rate * 0.9) + (self.current_rate * 0.1);
            
            // Record measurement
            let measurement = RateMeasurement {
                timestamp: now,
                bytes_per_second: self.current_rate,
                messages_per_second: 1.0 / duration_secs, // Simplified
            };
            
            self.rate_history.push_back(measurement);
            if self.rate_history.len() > 100 {
                self.rate_history.pop_front();
            }
            
            self.last_measurement = now;
        }
    }
    
    fn can_send(&self, bytes: u64) -> bool {
        // Simple rate limiting based on current rate
        let max_rate = 10 * 1024 * 1024; // 10 MB/s per peer
        self.current_rate + bytes as f64 <= max_rate as f64
    }
}

impl GlobalBandwidthTracker {
    fn new(bandwidth_limit: u64) -> Self {
        Self {
            total_bytes_sent: 0,
            total_bytes_received: 0,
            current_utilization: 0.0,
            peak_utilization: 0.0,
            bandwidth_limit,
            utilization_history: VecDeque::new(),
        }
    }
    
    fn record_usage(&mut self, bytes_sent: u64, bytes_received: u64) {
        self.total_bytes_sent += bytes_sent;
        self.total_bytes_received += bytes_received;
        
        // Calculate current utilization
        let total_bytes = bytes_sent + bytes_received;
        self.current_utilization = total_bytes as f64 / self.bandwidth_limit as f64;
        self.peak_utilization = self.peak_utilization.max(self.current_utilization);
        
        // Record utilization measurement
        let measurement = UtilizationMeasurement {
            timestamp: chrono::Utc::now(),
            utilization: self.current_utilization,
            throughput: total_bytes as f64,
        };
        
        self.utilization_history.push_back(measurement);
        if self.utilization_history.len() > 1000 {
            self.utilization_history.pop_front();
        }
    }
    
    fn can_send(&self, bytes: u64) -> bool {
        let projected_utilization = (self.current_utilization * self.bandwidth_limit as f64 + bytes as f64) / self.bandwidth_limit as f64;
        projected_utilization <= 0.9 // 90% utilization threshold
    }
}

impl CongestionController {
    fn new(config: CongestionControlConfig) -> Self {
        Self {
            congestion_window: config.initial_window,
            slow_start_threshold: config.slow_start_threshold,
            rtt_measurements: HashMap::new(),
            congestion_state: CongestionState::SlowStart,
            config,
        }
    }
    
    async fn update_bandwidth_usage(&mut self, peer_id: &PeerId, bytes_sent: u64) -> Result<()> {
        // Update congestion window based on successful transmission
        match self.congestion_state {
            CongestionState::SlowStart => {
                if self.congestion_window < self.slow_start_threshold {
                    self.congestion_window += 1;
                } else {
                    self.congestion_state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                self.congestion_window += 1 / self.congestion_window;
            }
            CongestionState::FastRecovery => {
                // Stay in fast recovery until all packets are acknowledged
            }
        }
        
        self.congestion_window = self.congestion_window.min(self.config.max_window);
        
        Ok(())
    }
    
    async fn update_rtt(&mut self, peer_id: &PeerId, rtt: f64) -> Result<()> {
        let rtt_tracker = self.rtt_measurements.entry(*peer_id)
            .or_insert_with(|| RttTracker::new());
        
        rtt_tracker.update_rtt(rtt);
        
        Ok(())
    }
    
    async fn handle_congestion(&mut self, _peer_id: &PeerId) -> Result<()> {
        // Reduce congestion window on congestion signal
        self.slow_start_threshold = self.congestion_window / 2;
        self.congestion_window = self.slow_start_threshold;
        self.congestion_state = CongestionState::FastRecovery;
        
        tracing::debug!("Congestion detected, reduced window to {}", self.congestion_window);
        
        Ok(())
    }
}

impl RttTracker {
    fn new() -> Self {
        Self {
            current_rtt: 0.0,
            smoothed_rtt: 0.0,
            rtt_variance: 0.0,
            min_rtt: f64::MAX,
            max_rtt: 0.0,
            measurements: VecDeque::new(),
        }
    }
    
    fn update_rtt(&mut self, rtt: f64) {
        self.current_rtt = rtt;
        self.min_rtt = self.min_rtt.min(rtt);
        self.max_rtt = self.max_rtt.max(rtt);
        
        // Update smoothed RTT using exponential moving average
        if self.smoothed_rtt == 0.0 {
            self.smoothed_rtt = rtt;
            self.rtt_variance = rtt / 2.0;
        } else {
            let alpha = 0.125;
            let beta = 0.25;
            
            self.rtt_variance = (1.0 - beta) * self.rtt_variance + beta * (self.smoothed_rtt - rtt).abs();
            self.smoothed_rtt = (1.0 - alpha) * self.smoothed_rtt + alpha * rtt;
        }
        
        self.measurements.push_back(rtt);
        if self.measurements.len() > 100 {
            self.measurements.pop_front();
        }
    }
}

impl CompressionEngine {
    fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            compression_stats: CompressionStatistics::default(),
        }
    }
    
    async fn compress_message(&mut self, message: &NetworkMessage) -> Result<Vec<u8>> {
        if !self.config.enable_compression {
            return bincode::serialize(message)
                .map_err(|e| NetworkError::SerializationError(e.to_string()));
        }
        
        let start_time = std::time::Instant::now();
        
        // Serialize message first
        let serialized = bincode::serialize(message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))?;
        
        // Check if compression is beneficial
        if serialized.len() < self.config.min_size_threshold {
            return Ok(serialized);
        }
        
        // Compress based on algorithm
        let compressed = match self.config.algorithm {
            CompressionAlgorithm::LZ4 => {
                lz4_flex::compress_prepend_size(&serialized)
            }
            CompressionAlgorithm::Zstd => {
                zstd::bulk::compress(&serialized, self.config.compression_level as i32)
                    .map_err(|e| NetworkError::CompressionError(e.to_string()))?
            }
            _ => serialized, // Fallback to uncompressed
        };
        
        // Update statistics
        self.compression_stats.total_uncompressed_bytes += serialized.len() as u64;
        self.compression_stats.total_compressed_bytes += compressed.len() as u64;
        self.compression_stats.compression_ratio = 
            self.compression_stats.total_compressed_bytes as f64 / self.compression_stats.total_uncompressed_bytes as f64;
        self.compression_stats.compression_time += start_time.elapsed().as_secs_f64();
        
        Ok(compressed)
    }
    
    async fn decompress_message(&mut self, compressed_data: &[u8]) -> Result<NetworkMessage> {
        if !self.config.enable_compression {
            return bincode::deserialize(compressed_data)
                .map_err(|e| NetworkError::SerializationError(e.to_string()));
        }
        
        let start_time = std::time::Instant::now();
        
        // Decompress based on algorithm
        let decompressed = match self.config.algorithm {
            CompressionAlgorithm::LZ4 => {
                lz4_flex::decompress_size_prepended(compressed_data)
                    .map_err(|e| NetworkError::CompressionError(e.to_string()))?
            }
            CompressionAlgorithm::Zstd => {
                zstd::bulk::decompress(compressed_data, 10 * 1024 * 1024) // 10MB max
                    .map_err(|e| NetworkError::CompressionError(e.to_string()))?
            }
            _ => compressed_data.to_vec(), // Fallback to uncompressed
        };
        
        self.compression_stats.decompression_time += start_time.elapsed().as_secs_f64();
        
        // Deserialize message
        bincode::deserialize(&decompressed)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))
    }
}

impl PriorityScheduler {
    fn new(priority_allocation: &HashMap<String, f64>) -> Self {
        let mut bandwidth_allocation = HashMap::new();
        bandwidth_allocation.insert(MessagePriority::Critical, priority_allocation.get("critical").copied().unwrap_or(0.4));
        bandwidth_allocation.insert(MessagePriority::High, priority_allocation.get("high").copied().unwrap_or(0.3));
        bandwidth_allocation.insert(MessagePriority::Normal, priority_allocation.get("normal").copied().unwrap_or(0.2));
        bandwidth_allocation.insert(MessagePriority::Low, priority_allocation.get("low").copied().unwrap_or(0.1));
        
        Self {
            priority_queues: HashMap::new(),
            bandwidth_allocation,
        }
    }
    
    async fn schedule_message(&mut self, message: NetworkMessage, peer_id: PeerId, priority: MessagePriority) -> Result<()> {
        let size = bincode::serialized_size(&message)
            .map_err(|e| NetworkError::SerializationError(e.to_string()))? as usize;
        
        let queued_message = QueuedMessage {
            message,
            peer_id,
            size,
            queued_at: chrono::Utc::now(),
            deadline: None, // TODO: Extract deadline from message
        };
        
        self.priority_queues.entry(priority)
            .or_insert_with(VecDeque::new)
            .push_back(queued_message);
        
        Ok(())
    }
    
    async fn get_next_message(&mut self) -> Result<Option<QueuedMessage>> {
        // Check queues in priority order
        for priority in [MessagePriority::Critical, MessagePriority::High, MessagePriority::Normal, MessagePriority::Low] {
            if let Some(queue) = self.priority_queues.get_mut(&priority) {
                if let Some(message) = queue.pop_front() {
                    return Ok(Some(message));
                }
            }
        }
        
        Ok(None)
    }
}

/// Bandwidth statistics
#[derive(Debug, Clone)]
pub struct BandwidthStatistics {
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub current_utilization: f64,
    pub peak_utilization: f64,
    pub active_peers: usize,
    pub compression_ratio: f64,
    pub congestion_window: u32,
}
