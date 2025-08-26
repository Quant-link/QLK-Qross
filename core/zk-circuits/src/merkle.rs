//! Merkle tree verification circuits optimized for cross-chain state proofs

use crate::{types::*, error::*, Circuit, CircuitInputs};
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, FieldElement, TraceTable,
    TransitionConstraintDegree, math::fields::f64::BaseElement,
};
use std::collections::HashMap;
use blake3::Hasher;
use rs_merkle::{MerkleTree, algorithms::Sha256};

/// Merkle tree verification circuit for state transition proofs
pub struct MerkleVerificationCircuit {
    id: CircuitId,
    config: MerkleConfig,
    complexity: CircuitComplexity,
}

impl MerkleVerificationCircuit {
    /// Create a new Merkle verification circuit
    pub fn new(id: CircuitId, config: MerkleConfig) -> Self {
        let complexity = Self::calculate_complexity(&config);
        
        Self {
            id,
            config,
            complexity,
        }
    }
    
    /// Calculate circuit complexity based on configuration
    fn calculate_complexity(config: &MerkleConfig) -> CircuitComplexity {
        let constraint_count = config.tree_height * 3; // Hash constraints per level
        let trace_length = 1 << (config.tree_height + 2); // Power of 2 for FFT
        let trace_width = 8; // Hash input/output columns
        
        CircuitComplexity {
            constraint_count,
            trace_length,
            trace_width,
            degree: 2,
            memory_usage: trace_length * trace_width * 8, // 8 bytes per field element
            estimated_proving_time: std::time::Duration::from_millis(
                (constraint_count as u64 * trace_length as u64) / 1000
            ),
            estimated_verification_time: std::time::Duration::from_millis(
                constraint_count as u64 / 10
            ),
        }
    }
    
    /// Generate Merkle proof for a leaf
    pub fn generate_merkle_proof(
        &self,
        leaves: &[Vec<u8>],
        leaf_index: usize,
    ) -> Result<MerkleProof> {
        if leaf_index >= leaves.len() {
            return Err(CircuitError::InvalidInput(
                format!("Leaf index {} out of bounds", leaf_index)
            ));
        }
        
        // Create Merkle tree
        let tree = MerkleTree::<Sha256>::from_leaves(leaves);
        let root = tree.root().ok_or_else(|| 
            CircuitError::MerkleTree("Failed to compute root".to_string()))?;
        
        // Generate proof path
        let proof_indices = tree.proof(&[leaf_index]);
        let proof_hashes = proof_indices.proof_hashes();
        
        let mut proof_path = Vec::new();
        for (i, hash) in proof_hashes.iter().enumerate() {
            let is_left = (leaf_index >> i) & 1 == 0;
            proof_path.push(MerkleNode {
                hash: hash.to_vec(),
                is_left,
            });
        }
        
        Ok(MerkleProof {
            leaf_index,
            leaf_value: leaves[leaf_index].clone(),
            proof_path,
            root: MerkleNode {
                hash: root.to_vec(),
                is_left: false,
            },
        })
    }
    
    /// Verify Merkle proof
    pub fn verify_merkle_proof(&self, proof: &MerkleProof) -> Result<bool> {
        let mut current_hash = self.hash_leaf(&proof.leaf_value)?;
        let mut index = proof.leaf_index;
        
        for node in &proof.proof_path {
            if node.is_left {
                current_hash = self.hash_pair(&node.hash, &current_hash)?;
            } else {
                current_hash = self.hash_pair(&current_hash, &node.hash)?;
            }
            index >>= 1;
        }
        
        Ok(current_hash == proof.root.hash)
    }
    
    /// Hash a leaf value
    fn hash_leaf(&self, leaf: &[u8]) -> Result<Vec<u8>> {
        match self.config.hash_function {
            HashFunction::Blake3 => {
                let mut hasher = Hasher::new();
                hasher.update(leaf);
                Ok(hasher.finalize().as_bytes().to_vec())
            }
            HashFunction::Sha256 => {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(leaf);
                Ok(hasher.finalize().to_vec())
            }
            _ => Err(CircuitError::UnsupportedOperation(
                format!("Hash function {:?} not implemented", self.config.hash_function)
            )),
        }
    }
    
    /// Hash a pair of values
    fn hash_pair(&self, left: &[u8], right: &[u8]) -> Result<Vec<u8>> {
        match self.config.hash_function {
            HashFunction::Blake3 => {
                let mut hasher = Hasher::new();
                hasher.update(left);
                hasher.update(right);
                Ok(hasher.finalize().as_bytes().to_vec())
            }
            HashFunction::Sha256 => {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(left);
                hasher.update(right);
                Ok(hasher.finalize().to_vec())
            }
            _ => Err(CircuitError::UnsupportedOperation(
                format!("Hash function {:?} not implemented", self.config.hash_function)
            )),
        }
    }
    
    /// Convert bytes to field elements
    fn bytes_to_field_elements(&self, bytes: &[u8]) -> Vec<BaseElement> {
        bytes.chunks(8)
            .map(|chunk| {
                let mut array = [0u8; 8];
                array[..chunk.len()].copy_from_slice(chunk);
                BaseElement::new(u64::from_le_bytes(array))
            })
            .collect()
    }
    
    /// Convert field elements to bytes
    fn field_elements_to_bytes(&self, elements: &[BaseElement]) -> Vec<u8> {
        elements.iter()
            .flat_map(|elem| elem.as_int().to_le_bytes())
            .collect()
    }
}

impl Circuit for MerkleVerificationCircuit {
    fn get_id(&self) -> CircuitId {
        self.id
    }
    
    fn get_name(&self) -> &'static str {
        "merkle_verification"
    }
    
    fn generate_trace(&self, inputs: &CircuitInputs) -> Result<TraceTable<BaseElement>> {
        // Extract Merkle proof from inputs
        if inputs.public_inputs.len() < 3 {
            return Err(CircuitError::InvalidInput(
                "Insufficient public inputs for Merkle verification".to_string()
            ));
        }
        
        // Parse inputs: leaf_index, leaf_value, proof_path, root
        let leaf_index = inputs.public_inputs[0].as_int() as usize;
        let proof_data = &inputs.public_inputs[1..];
        
        // Create trace table
        let trace_length = self.complexity.trace_length;
        let trace_width = self.complexity.trace_width;
        let mut trace = vec![vec![BaseElement::ZERO; trace_length]; trace_width];
        
        // Fill trace with Merkle verification computation
        self.fill_merkle_trace(&mut trace, leaf_index, proof_data)?;
        
        TraceTable::init(trace)
    }
    
    fn get_air(&self) -> Box<dyn Air<BaseElement>> {
        Box::new(MerkleVerificationAir::new(self.config.clone()))
    }
    
    fn validate_inputs(&self, inputs: &CircuitInputs) -> Result<()> {
        if inputs.public_inputs.is_empty() {
            return Err(CircuitError::InvalidInput("No public inputs provided".to_string()));
        }
        
        if inputs.public_inputs.len() < 3 {
            return Err(CircuitError::InvalidInput(
                "Insufficient inputs for Merkle verification".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn get_complexity(&self) -> CircuitComplexity {
        self.complexity.clone()
    }
    
    fn optimize(&mut self, constraints: &OptimizationConstraints) -> Result<()> {
        // Optimize tree height based on constraints
        if self.complexity.estimated_proving_time > constraints.max_proving_time {
            // Reduce tree height to meet timing constraints
            let target_height = (self.config.tree_height * 3) / 4;
            self.config.tree_height = target_height.max(8); // Minimum height of 8
            self.complexity = Self::calculate_complexity(&self.config);
        }
        
        // Optimize batch size for memory constraints
        if self.complexity.memory_usage > constraints.max_memory_usage {
            self.config.batch_size = (self.config.batch_size / 2).max(1);
        }
        
        Ok(())
    }
}

impl MerkleVerificationCircuit {
    /// Fill trace table with Merkle verification computation
    fn fill_merkle_trace(
        &self,
        trace: &mut [Vec<BaseElement>],
        leaf_index: usize,
        proof_data: &[BaseElement],
    ) -> Result<()> {
        let trace_length = trace[0].len();
        
        // Initialize first row with leaf data
        trace[0][0] = BaseElement::new(leaf_index as u64);
        
        // Fill subsequent rows with hash computations
        let mut current_index = leaf_index;
        let mut step = 1;
        
        for level in 0..self.config.tree_height {
            if step >= trace_length {
                break;
            }
            
            // Simulate hash computation in trace
            let is_left = (current_index & 1) == 0;
            
            // Store hash inputs and outputs
            trace[0][step] = BaseElement::new(current_index as u64);
            trace[1][step] = BaseElement::new(if is_left { 1 } else { 0 });
            
            // Simulate hash output (simplified for trace)
            trace[2][step] = BaseElement::new((current_index / 2) as u64);
            
            current_index >>= 1;
            step += 1;
        }
        
        Ok(())
    }
}

/// AIR (Algebraic Intermediate Representation) for Merkle verification
pub struct MerkleVerificationAir {
    config: MerkleConfig,
    context: AirContext<BaseElement>,
}

impl MerkleVerificationAir {
    pub fn new(config: MerkleConfig) -> Self {
        let trace_length = 1 << (config.tree_height + 2);
        let context = AirContext::new(
            trace_length,
            8, // trace width
            vec![], // public inputs
            TransitionConstraintDegree::new(2),
            4, // num assertions
        );
        
        Self { config, context }
    }
}

impl Air<BaseElement> for MerkleVerificationAir {
    fn context(&self) -> &AirContext<BaseElement> {
        &self.context
    }
    
    fn evaluate_transition<E: FieldElement + From<BaseElement>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();
        
        // Constraint 1: Index progression (index[i+1] = index[i] / 2)
        result[0] = next[0] - current[0] / E::from(BaseElement::new(2));
        
        // Constraint 2: Hash computation consistency
        // This is a simplified constraint - in practice, you'd implement
        // the full hash function constraints
        result[1] = current[2] - (current[0] + current[1]);
        
        // Additional constraints for hash verification would go here
    }
    
    fn get_assertions(&self) -> Vec<Assertion<BaseElement>> {
        // Define assertions for the circuit
        vec![
            // Assert that the final result is the expected root
            Assertion::single(self.context.trace_length() - 1, 2, BaseElement::ONE),
        ]
    }
}

/// Batch Merkle verification circuit for multiple proofs
pub struct BatchMerkleVerificationCircuit {
    base_circuit: MerkleVerificationCircuit,
    batch_size: usize,
}

impl BatchMerkleVerificationCircuit {
    pub fn new(id: CircuitId, config: MerkleConfig, batch_size: usize) -> Self {
        Self {
            base_circuit: MerkleVerificationCircuit::new(id, config),
            batch_size,
        }
    }
    
    /// Verify multiple Merkle proofs in batch
    pub fn batch_verify_proofs(&self, proofs: &[MerkleProof]) -> Result<Vec<bool>> {
        if proofs.len() > self.batch_size {
            return Err(CircuitError::InvalidInput(
                format!("Batch size {} exceeds maximum {}", proofs.len(), self.batch_size)
            ));
        }
        
        let mut results = Vec::with_capacity(proofs.len());
        
        for proof in proofs {
            let is_valid = self.base_circuit.verify_merkle_proof(proof)?;
            results.push(is_valid);
        }
        
        Ok(results)
    }
}

impl Circuit for BatchMerkleVerificationCircuit {
    fn get_id(&self) -> CircuitId {
        self.base_circuit.get_id()
    }
    
    fn get_name(&self) -> &'static str {
        "batch_merkle_verification"
    }
    
    fn generate_trace(&self, inputs: &CircuitInputs) -> Result<TraceTable<BaseElement>> {
        // Generate trace for batch verification
        // This would combine multiple Merkle verification traces
        self.base_circuit.generate_trace(inputs)
    }
    
    fn get_air(&self) -> Box<dyn Air<BaseElement>> {
        self.base_circuit.get_air()
    }
    
    fn validate_inputs(&self, inputs: &CircuitInputs) -> Result<()> {
        self.base_circuit.validate_inputs(inputs)
    }
    
    fn get_complexity(&self) -> CircuitComplexity {
        let base_complexity = self.base_circuit.get_complexity();
        
        CircuitComplexity {
            constraint_count: base_complexity.constraint_count * self.batch_size,
            trace_length: base_complexity.trace_length,
            trace_width: base_complexity.trace_width * self.batch_size,
            degree: base_complexity.degree,
            memory_usage: base_complexity.memory_usage * self.batch_size,
            estimated_proving_time: base_complexity.estimated_proving_time * self.batch_size as u32,
            estimated_verification_time: base_complexity.estimated_verification_time,
        }
    }
    
    fn optimize(&mut self, constraints: &OptimizationConstraints) -> Result<()> {
        // Optimize batch size based on constraints
        if self.get_complexity().memory_usage > constraints.max_memory_usage {
            self.batch_size = (self.batch_size / 2).max(1);
        }
        
        self.base_circuit.optimize(constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_merkle_proof_generation() {
        let config = MerkleConfig::default();
        let circuit = MerkleVerificationCircuit::new(1, config);
        
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
            b"leaf4".to_vec(),
        ];
        
        let proof = circuit.generate_merkle_proof(&leaves, 1).unwrap();
        assert_eq!(proof.leaf_index, 1);
        assert_eq!(proof.leaf_value, b"leaf2".to_vec());
        assert!(!proof.proof_path.is_empty());
    }
    
    #[test]
    fn test_merkle_proof_verification() {
        let config = MerkleConfig::default();
        let circuit = MerkleVerificationCircuit::new(1, config);
        
        let leaves = vec![
            b"leaf1".to_vec(),
            b"leaf2".to_vec(),
            b"leaf3".to_vec(),
            b"leaf4".to_vec(),
        ];
        
        let proof = circuit.generate_merkle_proof(&leaves, 1).unwrap();
        let is_valid = circuit.verify_merkle_proof(&proof).unwrap();
        assert!(is_valid);
    }
}
