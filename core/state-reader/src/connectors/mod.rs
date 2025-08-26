//! Blockchain connectors for different chains

pub mod ethereum;
pub mod substrate;
pub mod cosmos;

use crate::{StateReader, Result, ChainId, RpcConfig};
use std::sync::Arc;

/// Factory for creating state readers
pub struct StateReaderFactory;

impl StateReaderFactory {
    /// Create a state reader for the specified chain
    pub async fn create_reader(
        chain_id: ChainId,
        config: RpcConfig,
    ) -> Result<Box<dyn StateReader>> {
        match chain_id {
            ChainId::Ethereum | 
            ChainId::Polygon | 
            ChainId::BinanceSmartChain | 
            ChainId::Avalanche => {
                let reader = ethereum::EthereumStateReader::new(chain_id, config).await?;
                Ok(Box::new(reader))
            }
            ChainId::Polkadot | ChainId::Kusama => {
                let reader = substrate::SubstrateStateReader::new(chain_id, config).await?;
                Ok(Box::new(reader))
            }
            ChainId::Cosmos | ChainId::Osmosis => {
                let reader = cosmos::CosmosStateReader::new(chain_id, config).await?;
                Ok(Box::new(reader))
            }
            ChainId::Solana => {
                // TODO: Implement Solana connector
                Err(crate::StateReaderError::UnsupportedChain("Solana not yet implemented".to_string()))
            }
            ChainId::Near => {
                // TODO: Implement Near connector
                Err(crate::StateReaderError::UnsupportedChain("Near not yet implemented".to_string()))
            }
            ChainId::Custom(name) => {
                Err(crate::StateReaderError::UnsupportedChain(format!("Custom chain {} not supported", name)))
            }
        }
    }
}
