//! Ethereum-compatible blockchain state reader

use crate::{
    StateReader, BlockSubscription, EventSubscription, Result, StateReaderError,
    ChainId, ChainInfo, ConsensusType, Block, Transaction, Balance, Event, EventFilter,
    RpcConfig, TransactionStatus, Log, EventType
};
use async_trait::async_trait;
use ethers::{
    providers::{Provider, Http, Ws, Middleware, StreamExt},
    types::{Block as EthBlock, Transaction as EthTransaction, H256, U64, Address, Filter},
    core::types::BlockNumber,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Ethereum state reader implementation
pub struct EthereumStateReader {
    chain_id: ChainId,
    provider: Arc<Provider<Http>>,
    ws_provider: Option<Arc<Provider<Ws>>>,
    config: RpcConfig,
    chain_info: ChainInfo,
}

impl EthereumStateReader {
    /// Create a new Ethereum state reader
    pub async fn new(chain_id: ChainId, config: RpcConfig) -> Result<Self> {
        let provider = Provider::<Http>::try_from(&config.url)
            .map_err(|e| StateReaderError::Configuration(e.to_string()))?;
        let provider = Arc::new(provider);
        
        // Try to establish WebSocket connection if URL is provided
        let ws_provider = if let Some(ws_url) = &config.websocket_url {
            match Provider::<Ws>::connect(ws_url).await {
                Ok(ws) => Some(Arc::new(ws)),
                Err(e) => {
                    tracing::warn!("Failed to connect to WebSocket: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        let chain_info = Self::get_chain_info_for_id(&chain_id);
        
        Ok(Self {
            chain_id,
            provider,
            ws_provider,
            config,
            chain_info,
        })
    }
    
    fn get_chain_info_for_id(chain_id: &ChainId) -> ChainInfo {
        match chain_id {
            ChainId::Ethereum => ChainInfo {
                chain_id: chain_id.clone(),
                name: "Ethereum".to_string(),
                network_id: 1,
                consensus_type: ConsensusType::ProofOfStake,
                block_time: 12,
                finality_blocks: 32,
                native_token: "ETH".to_string(),
            },
            ChainId::Polygon => ChainInfo {
                chain_id: chain_id.clone(),
                name: "Polygon".to_string(),
                network_id: 137,
                consensus_type: ConsensusType::ProofOfStake,
                block_time: 2,
                finality_blocks: 128,
                native_token: "MATIC".to_string(),
            },
            ChainId::BinanceSmartChain => ChainInfo {
                chain_id: chain_id.clone(),
                name: "Binance Smart Chain".to_string(),
                network_id: 56,
                consensus_type: ConsensusType::ProofOfStake,
                block_time: 3,
                finality_blocks: 15,
                native_token: "BNB".to_string(),
            },
            ChainId::Avalanche => ChainInfo {
                chain_id: chain_id.clone(),
                name: "Avalanche C-Chain".to_string(),
                network_id: 43114,
                consensus_type: ConsensusType::ProofOfStake,
                block_time: 2,
                finality_blocks: 1,
                native_token: "AVAX".to_string(),
            },
            _ => ChainInfo {
                chain_id: chain_id.clone(),
                name: "Unknown".to_string(),
                network_id: 0,
                consensus_type: ConsensusType::Custom("Unknown".to_string()),
                block_time: 12,
                finality_blocks: 12,
                native_token: "ETH".to_string(),
            },
        }
    }
    
    /// Convert ethers Block to our Block type
    fn convert_block(&self, eth_block: EthBlock<H256>) -> Block {
        Block {
            hash: format!("{:?}", eth_block.hash.unwrap_or_default()),
            number: eth_block.number.unwrap_or_default().as_u64(),
            parent_hash: format!("{:?}", eth_block.parent_hash),
            timestamp: DateTime::from_timestamp(eth_block.timestamp.as_u64() as i64, 0)
                .unwrap_or_else(Utc::now),
            transactions: eth_block.transactions.iter()
                .map(|tx| format!("{:?}", tx))
                .collect(),
            state_root: format!("{:?}", eth_block.state_root),
            receipts_root: format!("{:?}", eth_block.receipts_root),
            gas_used: eth_block.gas_used.map(|g| g.as_u64()),
            gas_limit: Some(eth_block.gas_limit.as_u64()),
            extra_data: {
                let mut extra = HashMap::new();
                extra.insert("difficulty".to_string(), 
                    serde_json::Value::String(eth_block.difficulty.to_string()));
                extra.insert("total_difficulty".to_string(), 
                    serde_json::Value::String(
                        eth_block.total_difficulty.unwrap_or_default().to_string()
                    ));
                extra.insert("nonce".to_string(), 
                    serde_json::Value::String(format!("{:?}", eth_block.nonce.unwrap_or_default())));
                extra
            },
        }
    }
    
    /// Convert ethers Transaction to our Transaction type
    fn convert_transaction(&self, eth_tx: EthTransaction, receipt: Option<ethers::types::TransactionReceipt>) -> Transaction {
        let status = if let Some(ref receipt) = receipt {
            if receipt.status == Some(1.into()) {
                TransactionStatus::Success
            } else {
                TransactionStatus::Failed
            }
        } else {
            TransactionStatus::Pending
        };
        
        let logs = receipt.map(|r| r.logs.into_iter().map(|log| Log {
            address: format!("{:?}", log.address),
            topics: log.topics.iter().map(|t| format!("{:?}", t)).collect(),
            data: log.data.to_vec(),
            block_number: log.block_number.unwrap_or_default().as_u64(),
            transaction_hash: format!("{:?}", log.transaction_hash.unwrap_or_default()),
            transaction_index: log.transaction_index.unwrap_or_default().as_u64(),
            log_index: log.log_index.unwrap_or_default().as_u64(),
        }).collect()).unwrap_or_default();
        
        Transaction {
            hash: format!("{:?}", eth_tx.hash),
            block_hash: format!("{:?}", eth_tx.block_hash.unwrap_or_default()),
            block_number: eth_tx.block_number.unwrap_or_default().as_u64(),
            transaction_index: eth_tx.transaction_index.unwrap_or_default().as_u64(),
            from: format!("{:?}", eth_tx.from),
            to: eth_tx.to.map(|addr| format!("{:?}", addr)),
            value: eth_tx.value.to_string(),
            gas_price: eth_tx.gas_price.map(|gp| gp.to_string()),
            gas_limit: Some(eth_tx.gas.as_u64()),
            gas_used: None, // Will be filled from receipt
            nonce: eth_tx.nonce.as_u64(),
            input_data: eth_tx.input.to_vec(),
            status,
            logs,
        }
    }
}

#[async_trait]
impl StateReader for EthereumStateReader {
    async fn get_latest_block_number(&self) -> Result<u64> {
        let block_number = self.provider.get_block_number().await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?;
        Ok(block_number.as_u64())
    }
    
    async fn get_block(&self, block_number: u64) -> Result<Block> {
        let eth_block = self.provider
            .get_block_with_txs(BlockNumber::Number(U64::from(block_number)))
            .await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?
            .ok_or_else(|| StateReaderError::BlockNotFound(block_number))?;
        
        Ok(self.convert_block(eth_block))
    }
    
    async fn get_transaction(&self, tx_hash: &str) -> Result<Transaction> {
        let hash = tx_hash.parse::<H256>()
            .map_err(|e| StateReaderError::Parse(e.to_string()))?;
        
        let eth_tx = self.provider.get_transaction(hash).await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?
            .ok_or_else(|| StateReaderError::TransactionNotFound(tx_hash.to_string()))?;
        
        let receipt = self.provider.get_transaction_receipt(hash).await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?;
        
        Ok(self.convert_transaction(eth_tx, receipt))
    }
    
    async fn get_balance(&self, address: &str) -> Result<Balance> {
        let addr = address.parse::<Address>()
            .map_err(|e| StateReaderError::InvalidAddress(e.to_string()))?;
        
        let balance = self.provider.get_balance(addr, None).await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?;
        
        let nonce = self.provider.get_transaction_count(addr, None).await
            .map_err(|e| StateReaderError::Rpc(e.to_string()))?;
        
        let block_number = self.get_latest_block_number().await?;
        
        Ok(Balance {
            address: address.to_string(),
            balance: balance.to_string(),
            token_balances: HashMap::new(), // TODO: Implement token balance queries
            nonce: nonce.as_u64(),
            block_number,
        })
    }
    
    async fn subscribe_blocks(&self) -> Result<Box<dyn BlockSubscription>> {
        if let Some(ws_provider) = &self.ws_provider {
            let subscription = EthereumBlockSubscription::new(ws_provider.clone(), self.clone()).await?;
            Ok(Box::new(subscription))
        } else {
            Err(StateReaderError::Configuration("WebSocket not available".to_string()))
        }
    }
    
    async fn subscribe_events(&self, filter: EventFilter) -> Result<Box<dyn EventSubscription>> {
        if let Some(ws_provider) = &self.ws_provider {
            let subscription = EthereumEventSubscription::new(ws_provider.clone(), filter).await?;
            Ok(Box::new(subscription))
        } else {
            Err(StateReaderError::Configuration("WebSocket not available".to_string()))
        }
    }
    
    fn get_chain_info(&self) -> ChainInfo {
        self.chain_info.clone()
    }
}

impl Clone for EthereumStateReader {
    fn clone(&self) -> Self {
        Self {
            chain_id: self.chain_id.clone(),
            provider: self.provider.clone(),
            ws_provider: self.ws_provider.clone(),
            config: self.config.clone(),
            chain_info: self.chain_info.clone(),
        }
    }
}

/// Ethereum block subscription
pub struct EthereumBlockSubscription {
    ws_provider: Arc<Provider<Ws>>,
    reader: EthereumStateReader,
    stream: Option<ethers::providers::SubscriptionStream<'static, Ws, EthBlock<H256>>>,
}

impl EthereumBlockSubscription {
    async fn new(ws_provider: Arc<Provider<Ws>>, reader: EthereumStateReader) -> Result<Self> {
        Ok(Self {
            ws_provider,
            reader,
            stream: None,
        })
    }
}

#[async_trait]
impl BlockSubscription for EthereumBlockSubscription {
    async fn next_block(&mut self) -> Result<Option<Block>> {
        if self.stream.is_none() {
            let stream = self.ws_provider.subscribe_blocks().await
                .map_err(|e| StateReaderError::WebSocket(e.to_string()))?;
            self.stream = Some(stream);
        }
        
        if let Some(stream) = &mut self.stream {
            if let Some(eth_block) = stream.next().await {
                return Ok(Some(self.reader.convert_block(eth_block)));
            }
        }
        
        Ok(None)
    }
    
    async fn close(&mut self) -> Result<()> {
        self.stream = None;
        Ok(())
    }
}

/// Ethereum event subscription
pub struct EthereumEventSubscription {
    ws_provider: Arc<Provider<Ws>>,
    filter: EventFilter,
    stream: Option<ethers::providers::SubscriptionStream<'static, Ws, ethers::types::Log>>,
}

impl EthereumEventSubscription {
    async fn new(ws_provider: Arc<Provider<Ws>>, filter: EventFilter) -> Result<Self> {
        Ok(Self {
            ws_provider,
            filter,
            stream: None,
        })
    }
    
    fn convert_log_to_event(&self, log: ethers::types::Log) -> Event {
        Event {
            id: Uuid::new_v4(),
            event_type: EventType::Custom("Unknown".to_string()), // TODO: Decode event type
            block_number: log.block_number.unwrap_or_default().as_u64(),
            transaction_hash: format!("{:?}", log.transaction_hash.unwrap_or_default()),
            log_index: log.log_index.unwrap_or_default().as_u64(),
            address: format!("{:?}", log.address),
            topics: log.topics.iter().map(|t| format!("{:?}", t)).collect(),
            data: log.data.to_vec(),
            decoded_data: None, // TODO: Implement event decoding
        }
    }
}

#[async_trait]
impl EventSubscription for EthereumEventSubscription {
    async fn next_event(&mut self) -> Result<Option<Event>> {
        if self.stream.is_none() {
            let eth_filter = Filter::new(); // TODO: Convert our filter to ethers filter
            let stream = self.ws_provider.subscribe_logs(&eth_filter).await
                .map_err(|e| StateReaderError::WebSocket(e.to_string()))?;
            self.stream = Some(stream);
        }
        
        if let Some(stream) = &mut self.stream {
            if let Some(log) = stream.next().await {
                return Ok(Some(self.convert_log_to_event(log)));
            }
        }
        
        Ok(None)
    }
    
    async fn close(&mut self) -> Result<()> {
        self.stream = None;
        Ok(())
    }
}
