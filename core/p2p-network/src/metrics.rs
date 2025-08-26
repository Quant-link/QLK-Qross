//! Metrics collection for P2P network

use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge,
    register_counter, register_gauge, register_histogram,
    register_int_counter, register_int_gauge
};

/// Network metrics collector
pub struct NetworkMetrics {
    // Connection metrics
    connections_established: IntCounter,
    connections_closed: IntCounter,
    active_connections: IntGauge,
    connection_duration: Histogram,
    
    // Message metrics
    messages_sent: IntCounter,
    messages_received: IntCounter,
    message_size: Histogram,
    message_latency: Histogram,
    
    // Bandwidth metrics
    bytes_sent: IntCounter,
    bytes_received: IntCounter,
    bandwidth_utilization: Gauge,
    
    // Routing metrics
    routes_calculated: IntCounter,
    route_calculation_time: Histogram,
    route_hops: Histogram,
    
    // Gossip metrics
    gossip_messages_sent: IntCounter,
    gossip_messages_received: IntCounter,
    gossip_fanout: Histogram,
    
    // Discovery metrics
    peers_discovered: IntCounter,
    discovery_time: Histogram,
    
    // Relay metrics
    relay_connections_established: IntCounter,
    relay_bytes_transferred: IntCounter,
    relay_latency: Histogram,
    
    // Security metrics
    authentication_attempts: IntCounter,
    authentication_failures: IntCounter,
    rate_limit_violations: IntCounter,
    blacklisted_peers: IntGauge,
    
    // Error metrics
    connection_errors: IntCounter,
    routing_errors: IntCounter,
    gossip_errors: IntCounter,
    discovery_errors: IntCounter,
    relay_errors: IntCounter,
    security_errors: IntCounter,
    
    // Performance metrics
    cpu_usage: Gauge,
    memory_usage: Gauge,
    network_latency: Histogram,
    throughput: Gauge,
}

impl NetworkMetrics {
    /// Create a new network metrics collector
    pub fn new() -> Self {
        Self {
            connections_established: register_int_counter!(
                "p2p_connections_established_total",
                "Total number of connections established"
            ).unwrap(),
            
            connections_closed: register_int_counter!(
                "p2p_connections_closed_total",
                "Total number of connections closed"
            ).unwrap(),
            
            active_connections: register_int_gauge!(
                "p2p_active_connections",
                "Number of currently active connections"
            ).unwrap(),
            
            connection_duration: register_histogram!(
                "p2p_connection_duration_seconds",
                "Duration of connections in seconds",
                vec![1.0, 10.0, 60.0, 300.0, 1800.0, 3600.0]
            ).unwrap(),
            
            messages_sent: register_int_counter!(
                "p2p_messages_sent_total",
                "Total number of messages sent"
            ).unwrap(),
            
            messages_received: register_int_counter!(
                "p2p_messages_received_total",
                "Total number of messages received"
            ).unwrap(),
            
            message_size: register_histogram!(
                "p2p_message_size_bytes",
                "Size of messages in bytes",
                vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
            ).unwrap(),
            
            message_latency: register_histogram!(
                "p2p_message_latency_seconds",
                "Message latency in seconds",
                vec![0.001, 0.01, 0.1, 1.0, 10.0]
            ).unwrap(),
            
            bytes_sent: register_int_counter!(
                "p2p_bytes_sent_total",
                "Total number of bytes sent"
            ).unwrap(),
            
            bytes_received: register_int_counter!(
                "p2p_bytes_received_total",
                "Total number of bytes received"
            ).unwrap(),
            
            bandwidth_utilization: register_gauge!(
                "p2p_bandwidth_utilization_percent",
                "Current bandwidth utilization percentage"
            ).unwrap(),
            
            routes_calculated: register_int_counter!(
                "p2p_routes_calculated_total",
                "Total number of routes calculated"
            ).unwrap(),
            
            route_calculation_time: register_histogram!(
                "p2p_route_calculation_duration_seconds",
                "Time taken to calculate routes",
                vec![0.001, 0.01, 0.1, 1.0, 10.0]
            ).unwrap(),
            
            route_hops: register_histogram!(
                "p2p_route_hops",
                "Number of hops in calculated routes",
                vec![1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
            ).unwrap(),
            
            gossip_messages_sent: register_int_counter!(
                "p2p_gossip_messages_sent_total",
                "Total number of gossip messages sent"
            ).unwrap(),
            
            gossip_messages_received: register_int_counter!(
                "p2p_gossip_messages_received_total",
                "Total number of gossip messages received"
            ).unwrap(),
            
            gossip_fanout: register_histogram!(
                "p2p_gossip_fanout",
                "Gossip fanout (number of peers messaged)",
                vec![1.0, 5.0, 10.0, 20.0, 50.0]
            ).unwrap(),
            
            peers_discovered: register_int_counter!(
                "p2p_peers_discovered_total",
                "Total number of peers discovered"
            ).unwrap(),
            
            discovery_time: register_histogram!(
                "p2p_discovery_duration_seconds",
                "Time taken for peer discovery",
                vec![0.1, 1.0, 10.0, 60.0, 300.0]
            ).unwrap(),
            
            relay_connections_established: register_int_counter!(
                "p2p_relay_connections_established_total",
                "Total number of relay connections established"
            ).unwrap(),
            
            relay_bytes_transferred: register_int_counter!(
                "p2p_relay_bytes_transferred_total",
                "Total number of bytes transferred through relays"
            ).unwrap(),
            
            relay_latency: register_histogram!(
                "p2p_relay_latency_seconds",
                "Relay connection latency",
                vec![0.01, 0.1, 1.0, 10.0, 60.0]
            ).unwrap(),
            
            authentication_attempts: register_int_counter!(
                "p2p_authentication_attempts_total",
                "Total number of authentication attempts"
            ).unwrap(),
            
            authentication_failures: register_int_counter!(
                "p2p_authentication_failures_total",
                "Total number of authentication failures"
            ).unwrap(),
            
            rate_limit_violations: register_int_counter!(
                "p2p_rate_limit_violations_total",
                "Total number of rate limit violations"
            ).unwrap(),
            
            blacklisted_peers: register_int_gauge!(
                "p2p_blacklisted_peers",
                "Number of currently blacklisted peers"
            ).unwrap(),
            
            connection_errors: register_int_counter!(
                "p2p_connection_errors_total",
                "Total number of connection errors"
            ).unwrap(),
            
            routing_errors: register_int_counter!(
                "p2p_routing_errors_total",
                "Total number of routing errors"
            ).unwrap(),
            
            gossip_errors: register_int_counter!(
                "p2p_gossip_errors_total",
                "Total number of gossip errors"
            ).unwrap(),
            
            discovery_errors: register_int_counter!(
                "p2p_discovery_errors_total",
                "Total number of discovery errors"
            ).unwrap(),
            
            relay_errors: register_int_counter!(
                "p2p_relay_errors_total",
                "Total number of relay errors"
            ).unwrap(),
            
            security_errors: register_int_counter!(
                "p2p_security_errors_total",
                "Total number of security errors"
            ).unwrap(),
            
            cpu_usage: register_gauge!(
                "p2p_cpu_usage_percent",
                "Current CPU usage percentage"
            ).unwrap(),
            
            memory_usage: register_gauge!(
                "p2p_memory_usage_bytes",
                "Current memory usage in bytes"
            ).unwrap(),
            
            network_latency: register_histogram!(
                "p2p_network_latency_seconds",
                "Network latency measurements",
                vec![0.001, 0.01, 0.1, 1.0, 10.0]
            ).unwrap(),
            
            throughput: register_gauge!(
                "p2p_throughput_bytes_per_second",
                "Current network throughput in bytes per second"
            ).unwrap(),
        }
    }
    
    // Connection metrics
    pub fn record_connection_established(&self) {
        self.connections_established.inc();
    }
    
    pub fn record_connection_closed(&self, duration: f64) {
        self.connections_closed.inc();
        self.connection_duration.observe(duration);
    }
    
    pub fn set_active_connections(&self, count: i64) {
        self.active_connections.set(count);
    }
    
    // Message metrics
    pub fn record_message_sent(&self) {
        self.messages_sent.inc();
    }
    
    pub fn record_message_received(&self) {
        self.messages_received.inc();
    }
    
    pub fn record_message_size(&self, size: f64) {
        self.message_size.observe(size);
    }
    
    pub fn record_message_latency(&self, latency: f64) {
        self.message_latency.observe(latency);
    }
    
    // Bandwidth metrics
    pub fn record_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.inc_by(bytes);
    }
    
    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.inc_by(bytes);
    }
    
    pub fn set_bandwidth_utilization(&self, utilization: f64) {
        self.bandwidth_utilization.set(utilization * 100.0);
    }
    
    // Routing metrics
    pub fn record_route_calculated(&self) {
        self.routes_calculated.inc();
    }
    
    pub fn record_route_calculation_time(&self, duration: f64) {
        self.route_calculation_time.observe(duration);
    }
    
    pub fn record_route_hops(&self, hops: f64) {
        self.route_hops.observe(hops);
    }
    
    // Gossip metrics
    pub fn record_gossip_message_sent(&self) {
        self.gossip_messages_sent.inc();
    }
    
    pub fn record_gossip_message_received(&self) {
        self.gossip_messages_received.inc();
    }
    
    pub fn record_gossip_fanout(&self, fanout: f64) {
        self.gossip_fanout.observe(fanout);
    }
    
    // Discovery metrics
    pub fn record_peer_discovered(&self) {
        self.peers_discovered.inc();
    }
    
    pub fn record_discovery_time(&self, duration: f64) {
        self.discovery_time.observe(duration);
    }
    
    // Relay metrics
    pub fn record_relay_connection_established(&self) {
        self.relay_connections_established.inc();
    }
    
    pub fn record_relay_bytes_transferred(&self, bytes: u64) {
        self.relay_bytes_transferred.inc_by(bytes);
    }
    
    pub fn record_relay_latency(&self, latency: f64) {
        self.relay_latency.observe(latency);
    }
    
    // Security metrics
    pub fn record_authentication_attempt(&self) {
        self.authentication_attempts.inc();
    }
    
    pub fn record_authentication_failure(&self) {
        self.authentication_failures.inc();
    }
    
    pub fn record_rate_limit_violation(&self) {
        self.rate_limit_violations.inc();
    }
    
    pub fn set_blacklisted_peers(&self, count: i64) {
        self.blacklisted_peers.set(count);
    }
    
    // Error metrics
    pub fn record_connection_error(&self) {
        self.connection_errors.inc();
    }
    
    pub fn record_routing_error(&self) {
        self.routing_errors.inc();
    }
    
    pub fn record_gossip_error(&self) {
        self.gossip_errors.inc();
    }
    
    pub fn record_discovery_error(&self) {
        self.discovery_errors.inc();
    }
    
    pub fn record_relay_error(&self) {
        self.relay_errors.inc();
    }
    
    pub fn record_security_error(&self) {
        self.security_errors.inc();
    }
    
    // Performance metrics
    pub fn set_cpu_usage(&self, usage: f64) {
        self.cpu_usage.set(usage);
    }
    
    pub fn set_memory_usage(&self, usage: f64) {
        self.memory_usage.set(usage);
    }
    
    pub fn record_network_latency(&self, latency: f64) {
        self.network_latency.observe(latency);
    }
    
    pub fn set_throughput(&self, throughput: f64) {
        self.throughput.set(throughput);
    }
    
    // Calculated metrics
    pub fn get_total_messages_sent(&self) -> u64 {
        self.messages_sent.get() as u64
    }
    
    pub fn get_total_messages_received(&self) -> u64 {
        self.messages_received.get() as u64
    }
    
    pub fn get_total_bytes_sent(&self) -> u64 {
        self.bytes_sent.get() as u64
    }
    
    pub fn get_total_bytes_received(&self) -> u64 {
        self.bytes_received.get() as u64
    }
    
    pub fn get_average_latency(&self) -> f64 {
        // TODO: Calculate from histogram
        0.0
    }
    
    pub fn get_success_rate(&self) -> f64 {
        let total_attempts = self.authentication_attempts.get() as f64;
        let failures = self.authentication_failures.get() as f64;
        
        if total_attempts > 0.0 {
            (total_attempts - failures) / total_attempts
        } else {
            1.0
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}
