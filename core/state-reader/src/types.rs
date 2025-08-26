//! Core types for the state reader module

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Chain identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChainId {
    Ethereum,
    Polygon,
    BinanceSmartChain,
    Avalanche,
    Polkadot,
    Kusama,
    Cosmos,
    Osmosis,
    Solana,
    Near,
    Custom(String),
}

impl std::fmt::Display for ChainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainId::Ethereum => write!(f, "ethereum"),
            ChainId::Polygon => write!(f, "polygon"),
            ChainId::BinanceSmartChain => write!(f, "bsc"),
            ChainId::Avalanche => write!(f, "avalanche"),
            ChainId::Polkadot => write!(f, "polkadot"),
            ChainId::Kusama => write!(f, "kusama"),
            ChainId::Cosmos => write!(f, "cosmos"),
            ChainId::Osmosis => write!(f, "osmosis"),
            ChainId::Solana => write!(f, "solana"),
            ChainId::Near => write!(f, "near"),
            ChainId::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Chain information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfo {
    pub chain_id: ChainId,
    pub name: String,
    pub network_id: u64,
    pub consensus_type: ConsensusType,
    pub block_time: u64, // in seconds
    pub finality_blocks: u64,
    pub native_token: String,
}

/// Consensus type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusType {
    ProofOfWork,
    ProofOfStake,
    DelegatedProofOfStake,
    NominatedProofOfStake,
    Tendermint,
    ProofOfHistory,
    Custom(String),
}

/// Block representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub hash: String,
    pub number: u64,
    pub parent_hash: String,
    pub timestamp: DateTime<Utc>,
    pub transactions: Vec<String>, // Transaction hashes
    pub state_root: String,
    pub receipts_root: String,
    pub gas_used: Option<u64>,
    pub gas_limit: Option<u64>,
    pub extra_data: HashMap<String, serde_json::Value>,
}

/// Transaction representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub block_hash: String,
    pub block_number: u64,
    pub transaction_index: u64,
    pub from: String,
    pub to: Option<String>,
    pub value: String, // Big integer as string
    pub gas_price: Option<String>,
    pub gas_limit: Option<u64>,
    pub gas_used: Option<u64>,
    pub nonce: u64,
    pub input_data: Vec<u8>,
    pub status: TransactionStatus,
    pub logs: Vec<Log>,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Success,
    Failed,
    Reverted,
}

/// Transaction log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Log {
    pub address: String,
    pub topics: Vec<String>,
    pub data: Vec<u8>,
    pub block_number: u64,
    pub transaction_hash: String,
    pub transaction_index: u64,
    pub log_index: u64,
}

/// Account balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub address: String,
    pub balance: String, // Big integer as string
    pub token_balances: HashMap<String, String>, // token_address -> balance
    pub nonce: u64,
    pub block_number: u64,
}

/// Event representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub block_number: u64,
    pub transaction_hash: String,
    pub log_index: u64,
    pub address: String,
    pub topics: Vec<String>,
    pub data: Vec<u8>,
    pub decoded_data: Option<HashMap<String, serde_json::Value>>,
}

/// Event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Transfer,
    Approval,
    Deposit,
    Withdrawal,
    Swap,
    LiquidityAdded,
    LiquidityRemoved,
    Staking,
    Unstaking,
    Slashing,
    Governance,
    Custom(String),
}

/// Event filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    pub addresses: Option<Vec<String>>,
    pub topics: Option<Vec<Vec<String>>>,
    pub from_block: Option<u64>,
    pub to_block: Option<u64>,
    pub event_types: Option<Vec<EventType>>,
}

impl EventFilter {
    /// Create a filter that matches all events
    pub fn all() -> Self {
        Self {
            addresses: None,
            topics: None,
            from_block: None,
            to_block: None,
            event_types: None,
        }
    }
    
    /// Create a filter for specific addresses
    pub fn for_addresses(addresses: Vec<String>) -> Self {
        Self {
            addresses: Some(addresses),
            topics: None,
            from_block: None,
            to_block: None,
            event_types: None,
        }
    }
    
    /// Create a filter for specific event types
    pub fn for_event_types(event_types: Vec<EventType>) -> Self {
        Self {
            addresses: None,
            topics: None,
            from_block: None,
            to_block: None,
            event_types: Some(event_types),
        }
    }
}

/// RPC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcConfig {
    pub url: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub rate_limit_per_second: u32,
    pub websocket_url: Option<String>,
    pub api_key: Option<String>,
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            timeout_seconds: 30,
            max_retries: 3,
            retry_delay_ms: 1000,
            rate_limit_per_second: 10,
            websocket_url: None,
            api_key: None,
        }
    }
}
