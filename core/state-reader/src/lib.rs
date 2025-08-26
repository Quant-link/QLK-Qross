//! Quantlink Qross Multi-chain State Reader
//!
//! This module provides specialized RPC connectors and event listeners for different blockchains,
//! enabling the consensus aggregation engine to read state from multiple chains simultaneously.

pub mod connectors;
pub mod types;
pub mod error;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub use error::{StateReaderError, Result};
pub use types::*;

/// Trait for blockchain state readers
#[async_trait]
pub trait StateReader: Send + Sync {
    /// Get the latest block number
    async fn get_latest_block_number(&self) -> Result<u64>;
    
    /// Get block by number
    async fn get_block(&self, block_number: u64) -> Result<Block>;
    
    /// Get transaction by hash
    async fn get_transaction(&self, tx_hash: &str) -> Result<Transaction>;
    
    /// Get account balance
    async fn get_balance(&self, address: &str) -> Result<Balance>;
    
    /// Subscribe to new blocks
    async fn subscribe_blocks(&self) -> Result<Box<dyn BlockSubscription>>;
    
    /// Subscribe to events
    async fn subscribe_events(&self, filter: EventFilter) -> Result<Box<dyn EventSubscription>>;
    
    /// Get chain information
    fn get_chain_info(&self) -> ChainInfo;
}

/// Trait for block subscriptions
#[async_trait]
pub trait BlockSubscription: Send + Sync {
    /// Get the next block
    async fn next_block(&mut self) -> Result<Option<Block>>;
    
    /// Close the subscription
    async fn close(&mut self) -> Result<()>;
}

/// Trait for event subscriptions
#[async_trait]
pub trait EventSubscription: Send + Sync {
    /// Get the next event
    async fn next_event(&mut self) -> Result<Option<Event>>;
    
    /// Close the subscription
    async fn close(&mut self) -> Result<()>;
}

/// Multi-chain state reader manager
pub struct MultiChainStateReader {
    readers: HashMap<ChainId, Box<dyn StateReader>>,
    event_aggregator: EventAggregator,
}

impl MultiChainStateReader {
    /// Create a new multi-chain state reader
    pub fn new() -> Self {
        Self {
            readers: HashMap::new(),
            event_aggregator: EventAggregator::new(),
        }
    }
    
    /// Add a state reader for a specific chain
    pub fn add_reader(&mut self, chain_id: ChainId, reader: Box<dyn StateReader>) {
        self.readers.insert(chain_id, reader);
    }
    
    /// Get state reader for a specific chain
    pub fn get_reader(&self, chain_id: &ChainId) -> Option<&dyn StateReader> {
        self.readers.get(chain_id).map(|r| r.as_ref())
    }
    
    /// Get latest block numbers from all chains
    pub async fn get_all_latest_blocks(&self) -> Result<HashMap<ChainId, u64>> {
        let mut results = HashMap::new();
        
        for (chain_id, reader) in &self.readers {
            match reader.get_latest_block_number().await {
                Ok(block_number) => {
                    results.insert(chain_id.clone(), block_number);
                }
                Err(e) => {
                    tracing::warn!("Failed to get latest block for chain {}: {}", chain_id, e);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Start event aggregation from all chains
    pub async fn start_event_aggregation(&mut self) -> Result<()> {
        for (chain_id, reader) in &self.readers {
            let subscription = reader.subscribe_events(EventFilter::all()).await?;
            self.event_aggregator.add_subscription(chain_id.clone(), subscription).await?;
        }
        
        Ok(())
    }
    
    /// Get aggregated events
    pub async fn get_aggregated_events(&mut self) -> Result<Vec<AggregatedEvent>> {
        self.event_aggregator.get_events().await
    }
}

/// Event aggregator for combining events from multiple chains
pub struct EventAggregator {
    subscriptions: HashMap<ChainId, Box<dyn EventSubscription>>,
    event_buffer: Vec<AggregatedEvent>,
}

impl EventAggregator {
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            event_buffer: Vec::new(),
        }
    }
    
    pub async fn add_subscription(
        &mut self, 
        chain_id: ChainId, 
        subscription: Box<dyn EventSubscription>
    ) -> Result<()> {
        self.subscriptions.insert(chain_id, subscription);
        Ok(())
    }
    
    pub async fn get_events(&mut self) -> Result<Vec<AggregatedEvent>> {
        let mut events = Vec::new();
        
        for (chain_id, subscription) in &mut self.subscriptions {
            while let Some(event) = subscription.next_event().await? {
                events.push(AggregatedEvent {
                    id: Uuid::new_v4(),
                    chain_id: chain_id.clone(),
                    event,
                    timestamp: Utc::now(),
                });
            }
        }
        
        // Sort events by timestamp
        events.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(events)
    }
}

/// Aggregated event from multiple chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedEvent {
    pub id: Uuid,
    pub chain_id: ChainId,
    pub event: Event,
    pub timestamp: DateTime<Utc>,
}

impl Default for MultiChainStateReader {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EventAggregator {
    fn default() -> Self {
        Self::new()
    }
}
