use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeSet};
use std::str::FromStr;
use rust_decimal::Decimal;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValidatorId(pub Uuid);

impl ValidatorId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Validator {
    pub id: ValidatorId,
    pub stake: Decimal,
    pub delegated_stake: Decimal,
    pub status: ValidatorStatus,
    pub last_block_produced: Option<u64>,
    pub missed_blocks: u64,
    pub stake_history: Vec<StakeChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeChange {
    pub amount: Decimal,
    pub change_type: StakeChangeType,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub delegator: Option<ValidatorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StakeChangeType {
    Stake,
    Unstake,
    Delegate,
    Undelegate,
    Slash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockProductionSchedule {
    pub assignments: HashMap<u64, ValidatorId>, // block_height -> validator_id
    pub round_duration: u64, // blocks per selection round
    pub start_block: u64,
    pub end_block: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockProductionStats {
    pub blocks_produced: u64,
    pub blocks_missed: u64,
    pub last_block_produced: Option<u64>,
    pub consecutive_misses: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    pub validator_id: ValidatorId,
    pub block_height: u64,
    pub slash_type: SlashingType,
    pub penalty_amount: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlashingType {
    MissedBlocks { count: u64 },
    Inactivity,
    ExcessiveMisses,
    // Future: DoubleSign, InvalidBlock, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidatorStatus {
    Active,
    Inactive,
    Slashed { until_block: u64 },
}

#[derive(Debug, Clone)]
pub struct ValidatorSet {
    validators: HashMap<ValidatorId, Validator>,
    total_stake: Decimal,
    active_validators: BTreeSet<ValidatorId>,
    production_stats: HashMap<ValidatorId, BlockProductionStats>,
    current_schedule: Option<BlockProductionSchedule>,
    slashing_events: Vec<SlashingEvent>,
    slashed_stake_pool: Decimal, // Accumulated slashed stake
}

pub struct PersistentValidatorSet {
    validator_set: ValidatorSet,
    database: Option<crate::database::ValidatorDatabase>,
}

impl ValidatorSet {
    pub fn new() -> Self {
        Self {
            validators: HashMap::new(),
            total_stake: Decimal::ZERO,
            active_validators: BTreeSet::new(),
            production_stats: HashMap::new(),
            current_schedule: None,
            slashing_events: Vec::new(),
            slashed_stake_pool: Decimal::ZERO,
        }
    }

    pub fn minimum_stake() -> Decimal {
        Decimal::from(1000) // Minimum 1000 tokens to become validator
    }

    pub fn maximum_delegation_ratio() -> Decimal {
        Decimal::from(10) // Max 10:1 delegation ratio
    }

    pub fn add_validator(&mut self, validator: Validator) -> Result<(), String> {
        if self.validators.contains_key(&validator.id) {
            return Err("Validator already exists".to_string());
        }

        if validator.stake <= Decimal::ZERO {
            return Err("Validator stake must be positive".to_string());
        }

        let id = validator.id;
        let stake = validator.stake;
        let is_active = matches!(validator.status, ValidatorStatus::Active);

        self.validators.insert(id, validator);
        self.total_stake += stake;

        if is_active {
            self.active_validators.insert(id);
        }

        Ok(())
    }

    pub fn get_validator(&self, id: &ValidatorId) -> Option<&Validator> {
        self.validators.get(id)
    }

    pub fn update_validator_status(&mut self, id: &ValidatorId, status: ValidatorStatus) -> Result<(), String> {
        let validator = self.validators.get_mut(id)
            .ok_or("Validator not found")?;

        let was_active = matches!(validator.status, ValidatorStatus::Active);
        let is_active = matches!(status, ValidatorStatus::Active);

        validator.status = status;

        match (was_active, is_active) {
            (true, false) => { self.active_validators.remove(id); },
            (false, true) => { self.active_validators.insert(*id); },
            _ => {}
        }

        Ok(())
    }

    pub fn get_active_validators(&self) -> Vec<&Validator> {
        self.active_validators
            .iter()
            .filter_map(|id| self.validators.get(id))
            .collect()
    }

    pub fn total_stake(&self) -> Decimal {
        self.total_stake
    }

    pub fn active_stake(&self) -> Decimal {
        self.get_active_validators()
            .iter()
            .map(|v| v.stake + v.delegated_stake)
            .sum()
    }

    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    pub fn active_validator_count(&self) -> usize {
        self.active_validators.len()
    }

    /// Add stake to a validator
    pub fn add_stake(&mut self, validator_id: &ValidatorId, amount: Decimal) -> Result<(), String> {
        if amount <= Decimal::ZERO {
            return Err("Stake amount must be positive".to_string());
        }

        let validator = self.validators.get_mut(validator_id)
            .ok_or("Validator not found")?;

        validator.stake += amount;
        self.total_stake += amount;

        // Record stake change
        validator.stake_history.push(StakeChange {
            amount,
            change_type: StakeChangeType::Stake,
            timestamp: chrono::Utc::now(),
            delegator: None,
        });

        Ok(())
    }

    /// Remove stake from a validator
    pub fn remove_stake(&mut self, validator_id: &ValidatorId, amount: Decimal) -> Result<(), String> {
        if amount <= Decimal::ZERO {
            return Err("Stake amount must be positive".to_string());
        }

        let validator = self.validators.get_mut(validator_id)
            .ok_or("Validator not found")?;

        if validator.stake < amount {
            return Err("Insufficient stake to remove".to_string());
        }

        validator.stake -= amount;
        self.total_stake -= amount;

        // Check if validator still meets minimum stake requirement
        if validator.stake < Self::minimum_stake() {
            validator.status = ValidatorStatus::Inactive;
            self.active_validators.remove(validator_id);
        }

        // Record stake change
        validator.stake_history.push(StakeChange {
            amount,
            change_type: StakeChangeType::Unstake,
            timestamp: chrono::Utc::now(),
            delegator: None,
        });

        Ok(())
    }

    /// Delegate stake to a validator
    pub fn delegate_stake(&mut self, validator_id: &ValidatorId, delegator_id: ValidatorId, amount: Decimal) -> Result<(), String> {
        if amount <= Decimal::ZERO {
            return Err("Delegation amount must be positive".to_string());
        }

        let validator = self.validators.get_mut(validator_id)
            .ok_or("Validator not found")?;

        // Check delegation ratio limit
        let max_delegation = validator.stake * Self::maximum_delegation_ratio();

        if validator.delegated_stake + amount > max_delegation {
            return Err("Delegation would exceed maximum ratio".to_string());
        }

        validator.delegated_stake += amount;
        self.total_stake += amount;

        // Record delegation
        validator.stake_history.push(StakeChange {
            amount,
            change_type: StakeChangeType::Delegate,
            timestamp: chrono::Utc::now(),
            delegator: Some(delegator_id),
        });

        Ok(())
    }

    /// Undelegate stake from a validator
    pub fn undelegate_stake(&mut self, validator_id: &ValidatorId, delegator_id: ValidatorId, amount: Decimal) -> Result<(), String> {
        if amount <= Decimal::ZERO {
            return Err("Undelegation amount must be positive".to_string());
        }

        let validator = self.validators.get_mut(validator_id)
            .ok_or("Validator not found")?;

        if validator.delegated_stake < amount {
            return Err("Insufficient delegated stake to remove".to_string());
        }

        validator.delegated_stake -= amount;
        self.total_stake -= amount;

        // Record undelegation
        validator.stake_history.push(StakeChange {
            amount,
            change_type: StakeChangeType::Undelegate,
            timestamp: chrono::Utc::now(),
            delegator: Some(delegator_id),
        });

        Ok(())
    }

    /// Get total stake for a validator (own + delegated)
    pub fn get_total_validator_stake(&self, validator_id: &ValidatorId) -> Option<Decimal> {
        self.validators.get(validator_id)
            .map(|v| v.stake + v.delegated_stake)
    }

    /// Select a single validator based on stake weight using deterministic randomness
    pub fn select_validator(&self, block_height: u64, seed: u64) -> Option<ValidatorId> {
        let active_validators = self.get_active_validators();
        if active_validators.is_empty() {
            return None;
        }

        // Calculate total active stake for selection
        let total_active_stake: Decimal = active_validators
            .iter()
            .map(|v| v.stake + v.delegated_stake)
            .sum();

        if total_active_stake == Decimal::ZERO {
            return None;
        }

        // Generate deterministic random value
        let random_value = self.generate_deterministic_random(block_height, seed);
        let selection_point = Decimal::from(random_value) % total_active_stake;

        // Find validator using cumulative stake ranges
        let mut cumulative_stake = Decimal::ZERO;
        for validator in &active_validators {
            let validator_total_stake = validator.stake + validator.delegated_stake;
            cumulative_stake += validator_total_stake;

            if selection_point < cumulative_stake {
                return Some(validator.id);
            }
        }

        // Fallback to first validator (should not happen with correct math)
        active_validators.first().map(|v| v.id)
    }

    /// Select multiple validators without repetition
    pub fn select_validators(&self, block_height: u64, count: usize) -> Vec<ValidatorId> {
        let active_validators = self.get_active_validators();
        if active_validators.is_empty() || count == 0 {
            return Vec::new();
        }

        let mut selected = Vec::new();
        let mut excluded_ids = std::collections::HashSet::new();

        for i in 0..count {
            if let Some(validator_id) = self.select_validator_excluding(
                block_height,
                i as u64,
                &excluded_ids
            ) {
                selected.push(validator_id);
                excluded_ids.insert(validator_id);
            } else {
                break; // No more validators available
            }
        }

        selected
    }

    /// Select validator excluding specific validators (for multiple selection)
    fn select_validator_excluding(
        &self,
        block_height: u64,
        seed: u64,
        excluded: &std::collections::HashSet<ValidatorId>
    ) -> Option<ValidatorId> {
        let active_validators: Vec<&Validator> = self.get_active_validators()
            .into_iter()
            .filter(|v| !excluded.contains(&v.id))
            .collect();

        if active_validators.is_empty() {
            return None;
        }

        // Calculate total stake of non-excluded validators
        let total_stake: Decimal = active_validators
            .iter()
            .map(|v| v.stake + v.delegated_stake)
            .sum();

        if total_stake == Decimal::ZERO {
            return None;
        }

        // Generate deterministic random value
        let random_value = self.generate_deterministic_random(block_height, seed);
        let selection_point = Decimal::from(random_value) % total_stake;

        // Find validator using cumulative stake ranges
        let mut cumulative_stake = Decimal::ZERO;
        for validator in active_validators {
            let validator_total_stake = validator.stake + validator.delegated_stake;
            cumulative_stake += validator_total_stake;

            if selection_point < cumulative_stake {
                return Some(validator.id);
            }
        }

        None
    }

    /// Generate deterministic random number from block height and seed
    fn generate_deterministic_random(&self, block_height: u64, seed: u64) -> u64 {
        // Simple deterministic hash function
        // In production, would use a cryptographic hash like SHA-256
        let mut hash = block_height.wrapping_mul(0x9e3779b97f4a7c15);
        hash ^= seed.wrapping_mul(0x85ebca6b);
        hash ^= hash >> 32;
        hash = hash.wrapping_mul(0xc2b2ae35);
        hash ^= hash >> 16;
        hash
    }

    /// Verify that a validator selection is correct for given parameters
    pub fn verify_selection(&self, block_height: u64, seed: u64, claimed_validator: ValidatorId) -> bool {
        if let Some(selected) = self.select_validator(block_height, seed) {
            selected == claimed_validator
        } else {
            false
        }
    }

    /// Get selection probability for a validator (for testing/analysis)
    pub fn get_selection_probability(&self, validator_id: &ValidatorId) -> Option<f64> {
        let validator = self.validators.get(validator_id)?;

        if !matches!(validator.status, ValidatorStatus::Active) {
            return Some(0.0);
        }

        let validator_total_stake = validator.stake + validator.delegated_stake;
        let total_active_stake = self.active_stake();

        if total_active_stake == Decimal::ZERO {
            return Some(0.0);
        }

        // Convert to f64 for probability calculation
        let validator_stake_f64 = validator_total_stake.to_string().parse::<f64>().unwrap_or(0.0);
        let total_stake_f64 = total_active_stake.to_string().parse::<f64>().unwrap_or(1.0);

        Some(validator_stake_f64 / total_stake_f64)
    }

    /// Create a block production schedule for a given range
    pub fn create_production_schedule(&mut self, start_block: u64, round_duration: u64) -> BlockProductionSchedule {
        let end_block = start_block + round_duration - 1;
        let mut assignments = HashMap::new();

        // Assign each block in the round
        for block_height in start_block..=end_block {
            if let Some(validator_id) = self.select_validator(block_height, 0) {
                assignments.insert(block_height, validator_id);
            }
        }

        let schedule = BlockProductionSchedule {
            assignments,
            round_duration,
            start_block,
            end_block,
        };

        self.current_schedule = Some(schedule.clone());
        schedule
    }

    /// Get the validator assigned to produce a specific block
    pub fn get_block_producer(&self, block_height: u64) -> Option<ValidatorId> {
        // First check current schedule
        if let Some(schedule) = &self.current_schedule {
            if block_height >= schedule.start_block && block_height <= schedule.end_block {
                return schedule.assignments.get(&block_height).copied();
            }
        }

        // Fallback to direct selection if no schedule or block outside range
        self.select_validator(block_height, 0)
    }

    /// Record successful block production
    pub fn record_block_produced(&mut self, validator_id: ValidatorId, block_height: u64) {
        // Update validator's last block produced
        if let Some(validator) = self.validators.get_mut(&validator_id) {
            validator.last_block_produced = Some(block_height);
        }

        // Update production stats
        let stats = self.production_stats.entry(validator_id).or_insert(BlockProductionStats {
            blocks_produced: 0,
            blocks_missed: 0,
            last_block_produced: None,
            consecutive_misses: 0,
        });

        stats.blocks_produced += 1;
        stats.last_block_produced = Some(block_height);
        stats.consecutive_misses = 0; // Reset consecutive misses on successful production
    }

    /// Record missed block production
    pub fn record_missed_block(&mut self, validator_id: ValidatorId, block_height: u64) {
        // Update validator's missed blocks count
        if let Some(validator) = self.validators.get_mut(&validator_id) {
            validator.missed_blocks += 1;
        }

        // Update production stats
        let stats = self.production_stats.entry(validator_id).or_insert(BlockProductionStats {
            blocks_produced: 0,
            blocks_missed: 0,
            last_block_produced: None,
            consecutive_misses: 0,
        });

        stats.blocks_missed += 1;
        stats.consecutive_misses += 1;

        // Store consecutive misses count for slashing logic
        let consecutive_misses = stats.consecutive_misses;

        // Apply slashing for excessive consecutive misses
        if consecutive_misses >= 5 {
            let slash_result = self.apply_slashing(
                validator_id,
                SlashingType::ExcessiveMisses,
                block_height
            );
            if slash_result.is_ok() {
                self.update_validator_status(&validator_id, ValidatorStatus::Inactive).ok();
            }
        } else if consecutive_misses >= 3 {
            // Apply lighter slashing for 3+ consecutive misses
            self.apply_slashing(
                validator_id,
                SlashingType::MissedBlocks { count: consecutive_misses },
                block_height
            ).ok();
        }
    }

    /// Get block production statistics for a validator
    pub fn get_production_stats(&self, validator_id: &ValidatorId) -> Option<&BlockProductionStats> {
        self.production_stats.get(validator_id)
    }

    /// Get production efficiency (blocks produced / total assigned)
    pub fn get_production_efficiency(&self, validator_id: &ValidatorId) -> Option<f64> {
        let stats = self.production_stats.get(validator_id)?;
        let total_assigned = stats.blocks_produced + stats.blocks_missed;

        if total_assigned == 0 {
            return Some(1.0); // No blocks assigned yet, assume perfect efficiency
        }

        Some(stats.blocks_produced as f64 / total_assigned as f64)
    }

    /// Check if a validator is currently assigned to produce a block
    pub fn is_validator_assigned(&self, validator_id: &ValidatorId, block_height: u64) -> bool {
        self.get_block_producer(block_height) == Some(*validator_id)
    }

    /// Get all validators assigned in the current schedule
    pub fn get_scheduled_validators(&self) -> Vec<ValidatorId> {
        if let Some(schedule) = &self.current_schedule {
            let mut validators: Vec<ValidatorId> = schedule.assignments.values().copied().collect();
            validators.sort();
            validators.dedup();
            validators
        } else {
            Vec::new()
        }
    }

    /// Update schedule if needed (e.g., validator set changes)
    pub fn refresh_schedule_if_needed(&mut self, current_block: u64, round_duration: u64) {
        let needs_refresh = if let Some(schedule) = &self.current_schedule {
            current_block > schedule.end_block
        } else {
            true
        };

        if needs_refresh {
            let next_start = ((current_block / round_duration) + 1) * round_duration;
            self.create_production_schedule(next_start, round_duration);
        }
    }

    /// Apply slashing penalty to a validator
    pub fn apply_slashing(&mut self, validator_id: ValidatorId, slash_type: SlashingType, block_height: u64) -> Result<SlashingEvent, String> {
        // First, calculate penalty without borrowing validator mutably
        let penalty_amount = {
            let validator = self.validators.get(&validator_id)
                .ok_or("Validator not found")?;
            self.calculate_slashing_penalty(validator, &slash_type)?
        };

        // Now get mutable reference to apply the penalty
        let validator = self.validators.get_mut(&validator_id)
            .ok_or("Validator not found")?;

        if penalty_amount > validator.stake {
            return Err("Penalty exceeds validator stake".to_string());
        }

        // Apply the penalty
        validator.stake -= penalty_amount;
        self.total_stake -= penalty_amount;
        self.slashed_stake_pool += penalty_amount;

        // Update validator status if stake falls below minimum
        if validator.stake < Self::minimum_stake() {
            validator.status = ValidatorStatus::Slashed { until_block: block_height + 1000 }; // Slashed for 1000 blocks
            self.active_validators.remove(&validator_id);
        }

        // Record the slashing event
        let slashing_event = SlashingEvent {
            validator_id,
            block_height,
            slash_type: slash_type.clone(),
            penalty_amount,
            timestamp: chrono::Utc::now(),
        };

        // Add to stake history
        validator.stake_history.push(StakeChange {
            amount: penalty_amount,
            change_type: StakeChangeType::Slash,
            timestamp: chrono::Utc::now(),
            delegator: None,
        });

        // Store slashing event
        self.slashing_events.push(slashing_event.clone());

        Ok(slashing_event)
    }

    /// Calculate slashing penalty based on slash type and validator stake
    fn calculate_slashing_penalty(&self, validator: &Validator, slash_type: &SlashingType) -> Result<Decimal, String> {
        let base_stake = validator.stake + validator.delegated_stake;

        let penalty_percentage = match slash_type {
            SlashingType::MissedBlocks { count } => {
                // Escalating penalties: 0.5% for 3 misses, 1% for 4 misses, 2% for 5+ misses
                match count {
                    3 => Decimal::from_str("0.005").unwrap(), // 0.5%
                    4 => Decimal::from_str("0.01").unwrap(),  // 1%
                    _ => Decimal::from_str("0.02").unwrap(),  // 2%
                }
            },
            SlashingType::ExcessiveMisses => {
                Decimal::from_str("0.05").unwrap() // 5% for excessive consecutive misses
            },
            SlashingType::Inactivity => {
                Decimal::from_str("0.01").unwrap() // 1% for general inactivity
            },
        };

        let penalty = base_stake * penalty_percentage;

        // Ensure penalty doesn't exceed validator's own stake (don't slash delegated stake directly)
        let max_penalty = validator.stake;
        Ok(penalty.min(max_penalty))
    }

    /// Get slashing events for a validator
    pub fn get_slashing_events(&self, validator_id: &ValidatorId) -> Vec<&SlashingEvent> {
        self.slashing_events
            .iter()
            .filter(|event| event.validator_id == *validator_id)
            .collect()
    }

    /// Get total slashed stake in the pool
    pub fn get_slashed_stake_pool(&self) -> Decimal {
        self.slashed_stake_pool
    }

    /// Check if validator is currently slashed
    pub fn is_validator_slashed(&self, validator_id: &ValidatorId, current_block: u64) -> bool {
        if let Some(validator) = self.validators.get(validator_id) {
            match validator.status {
                ValidatorStatus::Slashed { until_block } => current_block < until_block,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Restore validator from slashed status if penalty period has ended
    pub fn restore_slashed_validator(&mut self, validator_id: &ValidatorId, current_block: u64) -> Result<(), String> {
        let validator = self.validators.get_mut(validator_id)
            .ok_or("Validator not found")?;

        match validator.status {
            ValidatorStatus::Slashed { until_block } => {
                if current_block >= until_block && validator.stake >= Self::minimum_stake() {
                    validator.status = ValidatorStatus::Active;
                    self.active_validators.insert(*validator_id);
                    Ok(())
                } else if current_block >= until_block {
                    validator.status = ValidatorStatus::Inactive; // Still below minimum stake
                    Ok(())
                } else {
                    Err("Slashing period not yet ended".to_string())
                }
            },
            _ => Err("Validator is not slashed".to_string()),
        }
    }

    /// Get slashing statistics
    pub fn get_slashing_statistics(&self) -> (usize, Decimal, Decimal) {
        let total_events = self.slashing_events.len();
        let total_slashed = self.slashed_stake_pool;
        let avg_penalty = if total_events > 0 {
            total_slashed / Decimal::from(total_events)
        } else {
            Decimal::ZERO
        };

        (total_events, total_slashed, avg_penalty)
    }
}

impl PersistentValidatorSet {
    pub fn new() -> Self {
        Self {
            validator_set: ValidatorSet::new(),
            database: None,
        }
    }

    pub async fn with_database(database_url: &str) -> Result<Self, sqlx::Error> {
        let database = crate::database::ValidatorDatabase::new(database_url).await?;
        Ok(Self {
            validator_set: ValidatorSet::new(),
            database: Some(database),
        })
    }

    pub async fn add_validator(&mut self, validator: Validator) -> Result<(), String> {
        // Add to in-memory set
        self.validator_set.add_validator(validator.clone())?;

        // Persist to database if available
        if let Some(db) = &self.database {
            db.save_validator(&validator).await.map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    pub async fn record_block_produced(&mut self, validator_id: ValidatorId, block_height: u64) -> Result<(), String> {
        // Update in-memory state
        self.validator_set.record_block_produced(validator_id, block_height);

        // Persist to database if available
        if let Some(db) = &self.database {
            if let Some(validator) = self.validator_set.get_validator(&validator_id) {
                db.save_validator(validator).await.map_err(|e| e.to_string())?;
            }

            if let Some(stats) = self.validator_set.get_production_stats(&validator_id) {
                db.save_production_stats(&validator_id, stats).await.map_err(|e| e.to_string())?;
            }
        }

        Ok(())
    }

    pub async fn apply_slashing(&mut self, validator_id: ValidatorId, slash_type: SlashingType, block_height: u64) -> Result<SlashingEvent, String> {
        // Apply slashing in-memory
        let event = self.validator_set.apply_slashing(validator_id, slash_type, block_height)?;

        // Persist to database if available
        if let Some(db) = &self.database {
            db.save_slashing_event(&event).await.map_err(|e| e.to_string())?;

            if let Some(validator) = self.validator_set.get_validator(&validator_id) {
                db.save_validator(validator).await.map_err(|e| e.to_string())?;
            }
        }

        Ok(event)
    }

    pub async fn load_from_database(&mut self) -> Result<(), String> {
        if let Some(db) = &self.database {
            let validator_ids = db.get_all_validators().await.map_err(|e| e.to_string())?;

            for validator_id in validator_ids {
                if let Some(validator) = db.load_validator(&validator_id).await.map_err(|e| e.to_string())? {
                    self.validator_set.validators.insert(validator_id, validator.clone());
                    self.validator_set.total_stake += validator.stake + validator.delegated_stake;

                    if matches!(validator.status, ValidatorStatus::Active) {
                        self.validator_set.active_validators.insert(validator_id);
                    }
                }

                if let Some(stats) = db.load_production_stats(&validator_id).await.map_err(|e| e.to_string())? {
                    self.validator_set.production_stats.insert(validator_id, stats);
                }
            }
        }

        Ok(())
    }

    // Delegate methods to inner ValidatorSet
    pub fn select_validator(&self, block_height: u64, seed: u64) -> Option<ValidatorId> {
        self.validator_set.select_validator(block_height, seed)
    }

    pub fn get_validator(&self, id: &ValidatorId) -> Option<&Validator> {
        self.validator_set.get_validator(id)
    }

    pub fn get_active_validators(&self) -> Vec<&Validator> {
        self.validator_set.get_active_validators()
    }

    pub fn total_stake(&self) -> Decimal {
        self.validator_set.total_stake()
    }

    pub fn active_validator_count(&self) -> usize {
        self.validator_set.active_validator_count()
    }
}

impl Default for ValidatorSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator(id: ValidatorId, stake: u64, status: ValidatorStatus) -> Validator {
        Validator {
            id,
            stake: Decimal::from(stake),
            delegated_stake: Decimal::ZERO,
            status,
            last_block_produced: None,
            missed_blocks: 0,
            stake_history: Vec::new(),
        }
    }

    #[test]
    fn test_validator_creation() {
        let id = ValidatorId::new();
        let validator = Validator {
            id,
            stake: Decimal::from(100),
            delegated_stake: Decimal::ZERO,
            status: ValidatorStatus::Active,
            last_block_produced: None,
            missed_blocks: 0,
            stake_history: Vec::new(),
        };

        assert_eq!(validator.id, id);
        assert_eq!(validator.stake, Decimal::from(100));
        assert_eq!(validator.status, ValidatorStatus::Active);
    }

    #[test]
    fn test_validator_set_add() {
        let mut validator_set = ValidatorSet::new();
        let validator = Validator {
            id: ValidatorId::new(),
            stake: Decimal::from(100),
            delegated_stake: Decimal::ZERO,
            status: ValidatorStatus::Active,
            last_block_produced: None,
            missed_blocks: 0,
            stake_history: Vec::new(),
        };

        let result = validator_set.add_validator(validator.clone());
        assert!(result.is_ok());
        assert_eq!(validator_set.validator_count(), 1);
        assert_eq!(validator_set.active_validator_count(), 1);
        assert_eq!(validator_set.total_stake(), Decimal::from(100));
    }

    #[test]
    fn test_validator_set_duplicate() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 100, ValidatorStatus::Active);

        validator_set.add_validator(validator.clone()).unwrap();
        let result = validator_set.add_validator(validator);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Validator already exists");
    }

    #[test]
    fn test_validator_set_zero_stake() {
        let mut validator_set = ValidatorSet::new();
        let validator = create_test_validator(ValidatorId::new(), 0, ValidatorStatus::Active);

        let result = validator_set.add_validator(validator);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Validator stake must be positive");
    }

    #[test]
    fn test_validator_status_update() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 100, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        assert_eq!(validator_set.active_validator_count(), 1);

        validator_set.update_validator_status(&id, ValidatorStatus::Inactive).unwrap();
        assert_eq!(validator_set.active_validator_count(), 0);

        validator_set.update_validator_status(&id, ValidatorStatus::Active).unwrap();
        assert_eq!(validator_set.active_validator_count(), 1);
    }

    #[test]
    fn test_active_stake_calculation() {
        let mut validator_set = ValidatorSet::new();

        let validator1 = create_test_validator(ValidatorId::new(), 100, ValidatorStatus::Active);
        let validator2 = create_test_validator(ValidatorId::new(), 200, ValidatorStatus::Inactive);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        assert_eq!(validator_set.total_stake(), Decimal::from(300));
        assert_eq!(validator_set.active_stake(), Decimal::from(100));
    }

    #[test]
    fn test_add_stake() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 100, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Add stake
        validator_set.add_stake(&id, Decimal::from(50)).unwrap();

        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.stake, Decimal::from(150));
        assert_eq!(validator_set.total_stake(), Decimal::from(150));
        assert_eq!(validator.stake_history.len(), 1);
        assert!(matches!(validator.stake_history[0].change_type, StakeChangeType::Stake));
    }

    #[test]
    fn test_remove_stake() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 2000, ValidatorStatus::Active); // Above minimum

        validator_set.add_validator(validator).unwrap();

        // Remove stake
        validator_set.remove_stake(&id, Decimal::from(500)).unwrap();

        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.stake, Decimal::from(1500));
        assert_eq!(validator_set.total_stake(), Decimal::from(1500));
        assert_eq!(validator.stake_history.len(), 1);
        assert!(matches!(validator.stake_history[0].change_type, StakeChangeType::Unstake));
    }

    #[test]
    fn test_remove_stake_below_minimum() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1200, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        assert_eq!(validator_set.active_validator_count(), 1);

        // Remove stake below minimum
        validator_set.remove_stake(&id, Decimal::from(500)).unwrap();

        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.stake, Decimal::from(700)); // Below 1000 minimum
        assert_eq!(validator.status, ValidatorStatus::Inactive);
        assert_eq!(validator_set.active_validator_count(), 0);
    }

    #[test]
    fn test_delegate_stake() {
        let mut validator_set = ValidatorSet::new();
        let validator_id = ValidatorId::new();
        let delegator_id = ValidatorId::new();
        let validator = create_test_validator(validator_id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Delegate stake
        validator_set.delegate_stake(&validator_id, delegator_id, Decimal::from(500)).unwrap();

        let validator = validator_set.get_validator(&validator_id).unwrap();
        assert_eq!(validator.delegated_stake, Decimal::from(500));
        assert_eq!(validator_set.total_stake(), Decimal::from(1500));
        assert_eq!(validator.stake_history.len(), 1);
        assert!(matches!(validator.stake_history[0].change_type, StakeChangeType::Delegate));
    }

    #[test]
    fn test_delegate_stake_exceeds_ratio() {
        let mut validator_set = ValidatorSet::new();
        let validator_id = ValidatorId::new();
        let delegator_id = ValidatorId::new();
        let validator = create_test_validator(validator_id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Try to delegate more than 10:1 ratio (max 10,000 for 1,000 stake)
        let result = validator_set.delegate_stake(&validator_id, delegator_id, Decimal::from(15000));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Delegation would exceed maximum ratio");
    }

    #[test]
    fn test_undelegate_stake() {
        let mut validator_set = ValidatorSet::new();
        let validator_id = ValidatorId::new();
        let delegator_id = ValidatorId::new();
        let validator = create_test_validator(validator_id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        validator_set.delegate_stake(&validator_id, delegator_id, Decimal::from(500)).unwrap();

        // Undelegate stake
        validator_set.undelegate_stake(&validator_id, delegator_id, Decimal::from(200)).unwrap();

        let validator = validator_set.get_validator(&validator_id).unwrap();
        assert_eq!(validator.delegated_stake, Decimal::from(300));
        assert_eq!(validator_set.total_stake(), Decimal::from(1300));
        assert_eq!(validator.stake_history.len(), 2); // Delegate + Undelegate
        assert!(matches!(validator.stake_history[1].change_type, StakeChangeType::Undelegate));
    }

    #[test]
    fn test_get_total_validator_stake() {
        let mut validator_set = ValidatorSet::new();
        let validator_id = ValidatorId::new();
        let delegator_id = ValidatorId::new();
        let validator = create_test_validator(validator_id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        validator_set.delegate_stake(&validator_id, delegator_id, Decimal::from(500)).unwrap();

        let total_stake = validator_set.get_total_validator_stake(&validator_id).unwrap();
        assert_eq!(total_stake, Decimal::from(1500)); // 1000 + 500
    }

    #[test]
    fn test_select_validator_deterministic() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 1000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 3000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        // Same block height and seed should always return same validator
        let selected1 = validator_set.select_validator(100, 42);
        let selected2 = validator_set.select_validator(100, 42);
        assert_eq!(selected1, selected2);

        // Different seed should potentially return different validator
        let selected3 = validator_set.select_validator(100, 43);
        assert!(selected3.is_some());
    }

    #[test]
    fn test_select_validator_no_active() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Inactive);

        validator_set.add_validator(validator).unwrap();

        let selected = validator_set.select_validator(100, 42);
        assert!(selected.is_none());
    }

    #[test]
    fn test_select_multiple_validators() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();
        let id3 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 1000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 2000, ValidatorStatus::Active);
        let validator3 = create_test_validator(id3, 3000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();
        validator_set.add_validator(validator3).unwrap();

        // Select 2 validators
        let selected = validator_set.select_validators(100, 2);
        assert_eq!(selected.len(), 2);

        // Should not have duplicates
        assert_ne!(selected[0], selected[1]);

        // Select more than available
        let selected_all = validator_set.select_validators(100, 5);
        assert_eq!(selected_all.len(), 3); // Only 3 active validators
    }

    #[test]
    fn test_verify_selection() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 1000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 2000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        let selected = validator_set.select_validator(100, 42).unwrap();

        // Verification should pass for correct selection
        assert!(validator_set.verify_selection(100, 42, selected));

        // Verification should fail for incorrect selection
        let other_id = if selected == id1 { id2 } else { id1 };
        assert!(!validator_set.verify_selection(100, 42, other_id));
    }

    #[test]
    fn test_selection_probability() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 1000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 3000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        let prob1 = validator_set.get_selection_probability(&id1).unwrap();
        let prob2 = validator_set.get_selection_probability(&id2).unwrap();

        // Validator 1: 1000/4000 = 0.25
        // Validator 2: 3000/4000 = 0.75
        assert!((prob1 - 0.25).abs() < 0.001);
        assert!((prob2 - 0.75).abs() < 0.001);
        assert!((prob1 + prob2 - 1.0).abs() < 0.001); // Should sum to 1.0
    }

    #[test]
    fn test_selection_with_delegation() {
        let mut validator_set = ValidatorSet::new();
        let validator_id = ValidatorId::new();
        let delegator_id = ValidatorId::new();

        let validator = create_test_validator(validator_id, 1000, ValidatorStatus::Active);
        validator_set.add_validator(validator).unwrap();

        // Add delegation
        validator_set.delegate_stake(&validator_id, delegator_id, Decimal::from(2000)).unwrap();

        // Selection should consider total stake (1000 + 2000 = 3000)
        let prob = validator_set.get_selection_probability(&validator_id).unwrap();
        assert!((prob - 1.0).abs() < 0.001); // Should be 100% since only validator

        // Verify selection still works
        let selected = validator_set.select_validator(100, 42);
        assert_eq!(selected, Some(validator_id));
    }

    #[test]
    fn test_create_production_schedule() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 1000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 2000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        // Create schedule for 5 blocks starting at block 100
        let schedule = validator_set.create_production_schedule(100, 5);

        assert_eq!(schedule.start_block, 100);
        assert_eq!(schedule.end_block, 104);
        assert_eq!(schedule.round_duration, 5);
        assert_eq!(schedule.assignments.len(), 5);

        // All assignments should be to active validators
        for validator_id in schedule.assignments.values() {
            assert!(validator_id == &id1 || validator_id == &id2);
        }
    }

    #[test]
    fn test_get_block_producer() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        validator_set.create_production_schedule(100, 3);

        // Should return validator for blocks in schedule
        assert_eq!(validator_set.get_block_producer(100), Some(id));
        assert_eq!(validator_set.get_block_producer(101), Some(id));
        assert_eq!(validator_set.get_block_producer(102), Some(id));

        // Should fallback to selection for blocks outside schedule
        let producer = validator_set.get_block_producer(200);
        assert!(producer.is_some());
    }

    #[test]
    fn test_record_block_produced() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Record block production
        validator_set.record_block_produced(id, 100);

        // Check validator state updated
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.last_block_produced, Some(100));

        // Check production stats
        let stats = validator_set.get_production_stats(&id).unwrap();
        assert_eq!(stats.blocks_produced, 1);
        assert_eq!(stats.blocks_missed, 0);
        assert_eq!(stats.last_block_produced, Some(100));
        assert_eq!(stats.consecutive_misses, 0);
    }

    #[test]
    fn test_record_missed_block() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Record missed block
        validator_set.record_missed_block(id, 100);

        // Check validator state updated
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.missed_blocks, 1);

        // Check production stats
        let stats = validator_set.get_production_stats(&id).unwrap();
        assert_eq!(stats.blocks_produced, 0);
        assert_eq!(stats.blocks_missed, 1);
        assert_eq!(stats.consecutive_misses, 1);
    }

    #[test]
    fn test_consecutive_misses_deactivation() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        assert_eq!(validator_set.active_validator_count(), 1);

        // Record 5 consecutive misses
        for i in 100..105 {
            validator_set.record_missed_block(id, i);
        }

        // Validator should be deactivated after 5 consecutive misses
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.status, ValidatorStatus::Inactive);
        assert_eq!(validator_set.active_validator_count(), 0);
    }

    #[test]
    fn test_production_efficiency() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Initially should be 100% (no blocks assigned)
        let efficiency = validator_set.get_production_efficiency(&id).unwrap_or(1.0);
        assert!((efficiency - 1.0).abs() < 0.001);

        // Record some production and misses
        validator_set.record_block_produced(id, 100);
        validator_set.record_block_produced(id, 101);
        validator_set.record_missed_block(id, 102);

        // Efficiency should be 2/3 = 0.667
        let efficiency = validator_set.get_production_efficiency(&id).unwrap();
        assert!((efficiency - 0.6666666666666666).abs() < 0.001);
    }

    #[test]
    fn test_is_validator_assigned() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();
        validator_set.create_production_schedule(100, 3);

        // Should be assigned for blocks in schedule
        assert!(validator_set.is_validator_assigned(&id, 100));
        assert!(validator_set.is_validator_assigned(&id, 101));
        assert!(validator_set.is_validator_assigned(&id, 102));
    }

    #[test]
    fn test_refresh_schedule_if_needed() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Create initial schedule
        validator_set.create_production_schedule(100, 5);
        let initial_end = validator_set.current_schedule.as_ref().unwrap().end_block;
        assert_eq!(initial_end, 104);

        // Refresh when current block is beyond schedule
        validator_set.refresh_schedule_if_needed(110, 5);
        let new_start = validator_set.current_schedule.as_ref().unwrap().start_block;
        assert_eq!(new_start, 115); // Next round starts at ((110/5)+1)*5 = 115
    }

    #[test]
    fn test_apply_slashing_missed_blocks() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 2000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Apply slashing for 3 missed blocks (0.5% penalty)
        let slash_event = validator_set.apply_slashing(
            id,
            SlashingType::MissedBlocks { count: 3 },
            100
        ).unwrap();

        // Check slashing event
        assert_eq!(slash_event.validator_id, id);
        assert_eq!(slash_event.slash_type, SlashingType::MissedBlocks { count: 3 });
        assert_eq!(slash_event.penalty_amount, Decimal::from(10)); // 0.5% of 2000 = 10

        // Check validator stake reduced
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.stake, Decimal::from(1990)); // 2000 - 10

        // Check slashed stake pool
        assert_eq!(validator_set.get_slashed_stake_pool(), Decimal::from(10));
    }

    #[test]
    fn test_apply_slashing_excessive_misses() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 2000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Apply slashing for excessive misses (5% penalty)
        let slash_event = validator_set.apply_slashing(
            id,
            SlashingType::ExcessiveMisses,
            100
        ).unwrap();

        // Check penalty amount (5% of 2000 = 100)
        assert_eq!(slash_event.penalty_amount, Decimal::from(100));

        // Check validator stake reduced
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.stake, Decimal::from(1900)); // 2000 - 100

        // Validator should still be active since 1900 > 1000 minimum
        assert_eq!(validator.status, ValidatorStatus::Active);
    }

    #[test]
    fn test_slashing_below_minimum_stake() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1050, ValidatorStatus::Active); // Just above minimum

        validator_set.add_validator(validator).unwrap();

        // Apply slashing that brings stake below minimum
        validator_set.apply_slashing(
            id,
            SlashingType::ExcessiveMisses, // 5% penalty = 52.5, bringing stake to 997.5
            100
        ).unwrap();

        let validator = validator_set.get_validator(&id).unwrap();

        // Should be slashed since below minimum stake (1000)
        assert!(matches!(validator.status, ValidatorStatus::Slashed { .. }));
        assert_eq!(validator_set.active_validator_count(), 0);
    }

    #[test]
    fn test_consecutive_misses_trigger_slashing() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 2000, ValidatorStatus::Active);

        validator_set.add_validator(validator).unwrap();

        // Record 3 consecutive misses (should trigger slashing)
        validator_set.record_missed_block(id, 100);
        validator_set.record_missed_block(id, 101);
        validator_set.record_missed_block(id, 102);

        // Check that slashing was applied
        let events = validator_set.get_slashing_events(&id);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].slash_type, SlashingType::MissedBlocks { count: 3 }));

        // Validator stake should be reduced
        let validator = validator_set.get_validator(&id).unwrap();
        assert!(validator.stake < Decimal::from(2000));
    }

    #[test]
    fn test_is_validator_slashed() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1050, ValidatorStatus::Active); // Will be below minimum after 5% slash

        validator_set.add_validator(validator).unwrap();

        // Apply slashing that brings below minimum
        validator_set.apply_slashing(id, SlashingType::ExcessiveMisses, 100).unwrap();

        // Should be slashed until block 1100
        assert!(validator_set.is_validator_slashed(&id, 500));
        assert!(validator_set.is_validator_slashed(&id, 1099));
        assert!(!validator_set.is_validator_slashed(&id, 1100));
    }

    #[test]
    fn test_restore_slashed_validator() {
        let mut validator_set = ValidatorSet::new();
        let id = ValidatorId::new();
        let validator = create_test_validator(id, 1050, ValidatorStatus::Active); // Will be below minimum after slash

        validator_set.add_validator(validator).unwrap();

        // Apply slashing that brings below minimum
        validator_set.apply_slashing(id, SlashingType::ExcessiveMisses, 100).unwrap();

        // Should be slashed
        assert!(validator_set.is_validator_slashed(&id, 500));

        // Add stake back to meet minimum requirement
        validator_set.add_stake(&id, Decimal::from(100)).unwrap();

        // Restore after slashing period
        validator_set.restore_slashed_validator(&id, 1100).unwrap();

        // Should be active again (has sufficient stake)
        let validator = validator_set.get_validator(&id).unwrap();
        assert_eq!(validator.status, ValidatorStatus::Active);
        assert_eq!(validator_set.active_validator_count(), 1);
    }

    #[test]
    fn test_slashing_statistics() {
        let mut validator_set = ValidatorSet::new();
        let id1 = ValidatorId::new();
        let id2 = ValidatorId::new();

        let validator1 = create_test_validator(id1, 2000, ValidatorStatus::Active);
        let validator2 = create_test_validator(id2, 3000, ValidatorStatus::Active);

        validator_set.add_validator(validator1).unwrap();
        validator_set.add_validator(validator2).unwrap();

        // Apply different slashing penalties
        validator_set.apply_slashing(id1, SlashingType::MissedBlocks { count: 3 }, 100).unwrap(); // 10 penalty
        validator_set.apply_slashing(id2, SlashingType::ExcessiveMisses, 101).unwrap(); // 150 penalty

        let (total_events, total_slashed, avg_penalty) = validator_set.get_slashing_statistics();
        assert_eq!(total_events, 2);
        assert_eq!(total_slashed, Decimal::from(160)); // 10 + 150
        assert_eq!(avg_penalty, Decimal::from(80)); // 160 / 2
    }

    #[tokio::test]
    async fn test_persistent_validator_set() {
        let mut persistent_set = PersistentValidatorSet::with_database(":memory:").await.unwrap();

        let id = ValidatorId::new();
        let validator = create_test_validator(id, 2000, ValidatorStatus::Active);

        // Add validator
        persistent_set.add_validator(validator).await.unwrap();

        // Verify it's in memory
        assert_eq!(persistent_set.active_validator_count(), 1);
        assert_eq!(persistent_set.total_stake(), Decimal::from(2000));

        // Record block production
        persistent_set.record_block_produced(id, 100).await.unwrap();

        // Apply slashing
        let event = persistent_set.apply_slashing(id, SlashingType::MissedBlocks { count: 3 }, 101).await.unwrap();
        assert_eq!(event.penalty_amount, Decimal::from(10)); // 0.5% of 2000

        // Create new instance and load from database
        let mut new_persistent_set = PersistentValidatorSet::with_database(":memory:").await.unwrap();
        new_persistent_set.load_from_database().await.unwrap();

        // Note: In-memory database doesn't persist between connections
        // In a real test, we'd use a file-based database
    }
}
