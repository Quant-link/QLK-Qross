//! Price oracle for asset pricing

use crate::{types::*, error::*};

/// Price oracle
pub struct PriceOracle {
    config: OracleConfig,
    // TODO: Add implementation
}

impl PriceOracle {
    pub fn new(config: OracleConfig) -> Self {
        Self { config }
    }
    
    pub async fn start(&mut self) -> Result<()> {
        Ok(())
    }
}
