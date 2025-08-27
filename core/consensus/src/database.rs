use sqlx::{SqlitePool, Row};
use crate::{ValidatorId, Validator, ValidatorStatus, SlashingEvent, BlockProductionStats};
use rust_decimal::Decimal;
use std::str::FromStr;

pub struct ValidatorDatabase {
    pool: SqlitePool,
}

impl ValidatorDatabase {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePool::connect(database_url).await?;
        
        // Create tables
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS validators (
                id TEXT PRIMARY KEY,
                stake TEXT NOT NULL,
                delegated_stake TEXT NOT NULL,
                status TEXT NOT NULL,
                last_block_produced INTEGER,
                missed_blocks INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS slashing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validator_id TEXT NOT NULL,
                block_height INTEGER NOT NULL,
                slash_type TEXT NOT NULL,
                penalty_amount TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (validator_id) REFERENCES validators (id)
            )
            "#,
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS production_stats (
                validator_id TEXT PRIMARY KEY,
                blocks_produced INTEGER NOT NULL,
                blocks_missed INTEGER NOT NULL,
                last_block_produced INTEGER,
                consecutive_misses INTEGER NOT NULL,
                FOREIGN KEY (validator_id) REFERENCES validators (id)
            )
            "#,
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    pub async fn save_validator(&self, validator: &Validator) -> Result<(), sqlx::Error> {
        let status_str = match &validator.status {
            ValidatorStatus::Active => "Active".to_string(),
            ValidatorStatus::Inactive => "Inactive".to_string(),
            ValidatorStatus::Slashed { until_block } => format!("Slashed:{}", until_block),
        };

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO validators 
            (id, stake, delegated_stake, status, last_block_produced, missed_blocks, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            "#,
        )
        .bind(validator.id.0.to_string())
        .bind(validator.stake.to_string())
        .bind(validator.delegated_stake.to_string())
        .bind(status_str)
        .bind(validator.last_block_produced.map(|b| b as i64))
        .bind(validator.missed_blocks as i64)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn load_validator(&self, validator_id: &ValidatorId) -> Result<Option<Validator>, sqlx::Error> {
        let row = sqlx::query(
            "SELECT id, stake, delegated_stake, status, last_block_produced, missed_blocks FROM validators WHERE id = ?"
        )
        .bind(validator_id.0.to_string())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let status_str: String = row.get("status");
            let status = if status_str == "Active" {
                ValidatorStatus::Active
            } else if status_str == "Inactive" {
                ValidatorStatus::Inactive
            } else if status_str.starts_with("Slashed:") {
                let until_block: u64 = status_str.strip_prefix("Slashed:").unwrap().parse().unwrap();
                ValidatorStatus::Slashed { until_block }
            } else {
                ValidatorStatus::Inactive
            };

            let validator = Validator {
                id: *validator_id,
                stake: Decimal::from_str(&row.get::<String, _>("stake")).unwrap(),
                delegated_stake: Decimal::from_str(&row.get::<String, _>("delegated_stake")).unwrap(),
                status,
                last_block_produced: row.get::<Option<i64>, _>("last_block_produced").map(|b| b as u64),
                missed_blocks: row.get::<i64, _>("missed_blocks") as u64,
                stake_history: Vec::new(), // Load separately if needed
            };

            Ok(Some(validator))
        } else {
            Ok(None)
        }
    }

    pub async fn save_slashing_event(&self, event: &SlashingEvent) -> Result<(), sqlx::Error> {
        let slash_type_str = match &event.slash_type {
            crate::SlashingType::MissedBlocks { count } => format!("MissedBlocks:{}", count),
            crate::SlashingType::Inactivity => "Inactivity".to_string(),
            crate::SlashingType::ExcessiveMisses => "ExcessiveMisses".to_string(),
        };

        sqlx::query(
            r#"
            INSERT INTO slashing_events 
            (validator_id, block_height, slash_type, penalty_amount, timestamp)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(event.validator_id.0.to_string())
        .bind(event.block_height as i64)
        .bind(slash_type_str)
        .bind(event.penalty_amount.to_string())
        .bind(event.timestamp)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn save_production_stats(&self, validator_id: &ValidatorId, stats: &BlockProductionStats) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO production_stats 
            (validator_id, blocks_produced, blocks_missed, last_block_produced, consecutive_misses)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(validator_id.0.to_string())
        .bind(stats.blocks_produced as i64)
        .bind(stats.blocks_missed as i64)
        .bind(stats.last_block_produced.map(|b| b as i64))
        .bind(stats.consecutive_misses as i64)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn load_production_stats(&self, validator_id: &ValidatorId) -> Result<Option<BlockProductionStats>, sqlx::Error> {
        let row = sqlx::query(
            "SELECT blocks_produced, blocks_missed, last_block_produced, consecutive_misses FROM production_stats WHERE validator_id = ?"
        )
        .bind(validator_id.0.to_string())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let stats = BlockProductionStats {
                blocks_produced: row.get::<i64, _>("blocks_produced") as u64,
                blocks_missed: row.get::<i64, _>("blocks_missed") as u64,
                last_block_produced: row.get::<Option<i64>, _>("last_block_produced").map(|b| b as u64),
                consecutive_misses: row.get::<i64, _>("consecutive_misses") as u64,
            };
            Ok(Some(stats))
        } else {
            Ok(None)
        }
    }

    pub async fn get_all_validators(&self) -> Result<Vec<ValidatorId>, sqlx::Error> {
        let rows = sqlx::query("SELECT id FROM validators")
            .fetch_all(&self.pool)
            .await?;

        let validator_ids = rows
            .into_iter()
            .map(|row| {
                let id_str: String = row.get("id");
                ValidatorId(uuid::Uuid::parse_str(&id_str).unwrap())
            })
            .collect();

        Ok(validator_ids)
    }
}
