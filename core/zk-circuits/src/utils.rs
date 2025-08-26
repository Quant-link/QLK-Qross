//! Utility functions for zk-STARK circuits

use crate::{types::*, error::*};
use math::fields::f64::BaseElement;
use std::collections::HashMap;

/// Circuit optimization utilities
pub struct CircuitOptimizer;

impl CircuitOptimizer {
    /// Optimize circuit for specific constraints
    pub fn optimize_for_constraints(
        complexity: &CircuitComplexity,
        constraints: &OptimizationConstraints,
    ) -> Result<OptimizationResult> {
        let original_complexity = complexity.clone();
        let mut optimized_complexity = complexity.clone();
        let start_time = std::time::Instant::now();
        
        let mut improvements = HashMap::new();
        
        // Optimize proving time
        if optimized_complexity.estimated_proving_time > constraints.max_proving_time {
            let reduction_factor = constraints.max_proving_time.as_secs_f64() / 
                                 optimized_complexity.estimated_proving_time.as_secs_f64();
            
            optimized_complexity.constraint_count = 
                (optimized_complexity.constraint_count as f64 * reduction_factor) as usize;
            optimized_complexity.trace_length = 
                (optimized_complexity.trace_length as f64 * reduction_factor.sqrt()) as usize;
            
            optimized_complexity.estimated_proving_time = constraints.max_proving_time;
            
            improvements.insert("proving_time_reduction".to_string(), 1.0 - reduction_factor);
        }
        
        // Optimize memory usage
        if optimized_complexity.memory_usage > constraints.max_memory_usage {
            let reduction_factor = constraints.max_memory_usage as f64 / 
                                 optimized_complexity.memory_usage as f64;
            
            optimized_complexity.trace_width = 
                (optimized_complexity.trace_width as f64 * reduction_factor.sqrt()) as usize;
            optimized_complexity.memory_usage = constraints.max_memory_usage;
            
            improvements.insert("memory_reduction".to_string(), 1.0 - reduction_factor);
        }
        
        // Optimize verification time
        if optimized_complexity.estimated_verification_time > constraints.max_verification_time {
            let reduction_factor = constraints.max_verification_time.as_secs_f64() / 
                                 optimized_complexity.estimated_verification_time.as_secs_f64();
            
            optimized_complexity.degree = (optimized_complexity.degree as f64 * reduction_factor) as usize;
            optimized_complexity.estimated_verification_time = constraints.max_verification_time;
            
            improvements.insert("verification_time_reduction".to_string(), 1.0 - reduction_factor);
        }
        
        let optimization_time = start_time.elapsed();
        
        Ok(OptimizationResult {
            original_complexity,
            optimized_complexity,
            optimization_time,
            improvements,
        })
    }
    
    /// Calculate optimal batch size for circuit operations
    pub fn calculate_optimal_batch_size(
        circuit_complexity: &CircuitComplexity,
        available_memory: usize,
        target_latency: std::time::Duration,
    ) -> usize {
        let memory_per_circuit = circuit_complexity.memory_usage;
        let max_batch_by_memory = available_memory / memory_per_circuit.max(1);
        
        let time_per_circuit = circuit_complexity.estimated_proving_time;
        let max_batch_by_time = if time_per_circuit.as_millis() > 0 {
            (target_latency.as_millis() / time_per_circuit.as_millis()) as usize
        } else {
            1000
        };
        
        max_batch_by_memory.min(max_batch_by_time).max(1)
    }
    
    /// Estimate circuit resource requirements
    pub fn estimate_resources(
        circuit_complexity: &CircuitComplexity,
        batch_size: usize,
    ) -> ResourceEstimate {
        ResourceEstimate {
            memory_bytes: circuit_complexity.memory_usage * batch_size,
            cpu_time: circuit_complexity.estimated_proving_time * batch_size as u32,
            verification_time: circuit_complexity.estimated_verification_time,
            parallelization_factor: Self::calculate_parallelization_factor(circuit_complexity),
        }
    }
    
    /// Calculate optimal parallelization factor
    fn calculate_parallelization_factor(complexity: &CircuitComplexity) -> usize {
        let cpu_count = num_cpus::get();
        let memory_factor = complexity.memory_usage / (1024 * 1024 * 1024); // GB
        
        // Balance CPU cores with memory requirements
        (cpu_count / memory_factor.max(1)).max(1).min(cpu_count)
    }
}

/// Resource estimation result
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    pub memory_bytes: usize,
    pub cpu_time: std::time::Duration,
    pub verification_time: std::time::Duration,
    pub parallelization_factor: usize,
}

/// Field element utilities
pub struct FieldUtils;

impl FieldUtils {
    /// Convert bytes to field elements
    pub fn bytes_to_field_elements(bytes: &[u8]) -> Vec<BaseElement> {
        bytes.chunks(8)
            .map(|chunk| {
                let mut array = [0u8; 8];
                array[..chunk.len()].copy_from_slice(chunk);
                BaseElement::new(u64::from_le_bytes(array))
            })
            .collect()
    }
    
    /// Convert field elements to bytes
    pub fn field_elements_to_bytes(elements: &[BaseElement]) -> Vec<u8> {
        elements.iter()
            .flat_map(|elem| elem.as_int().to_le_bytes())
            .collect()
    }
    
    /// Generate random field element
    pub fn random_field_element() -> BaseElement {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        BaseElement::new(rng.next_u64())
    }
    
    /// Generate random field elements
    pub fn random_field_elements(count: usize) -> Vec<BaseElement> {
        (0..count).map(|_| Self::random_field_element()).collect()
    }
    
    /// Compute polynomial from roots
    pub fn polynomial_from_roots(roots: &[BaseElement]) -> Vec<BaseElement> {
        if roots.is_empty() {
            return vec![BaseElement::ONE];
        }
        
        let mut poly = vec![BaseElement::ONE];
        
        for root in roots {
            let mut new_poly = vec![BaseElement::ZERO; poly.len() + 1];
            
            // Multiply by (x - root)
            for i in 0..poly.len() {
                new_poly[i + 1] = new_poly[i + 1] + poly[i];
                new_poly[i] = new_poly[i] - poly[i] * *root;
            }
            
            poly = new_poly;
        }
        
        poly
    }
    
    /// Evaluate polynomial at point
    pub fn evaluate_polynomial(coefficients: &[BaseElement], point: BaseElement) -> BaseElement {
        if coefficients.is_empty() {
            return BaseElement::ZERO;
        }
        
        let mut result = coefficients[coefficients.len() - 1];
        for i in (0..coefficients.len() - 1).rev() {
            result = result * point + coefficients[i];
        }
        
        result
    }
    
    /// Compute modular inverse
    pub fn mod_inverse(a: BaseElement) -> Option<BaseElement> {
        if a == BaseElement::ZERO {
            None
        } else {
            Some(a.inverse())
        }
    }
}

/// Proof serialization utilities
pub struct ProofSerializer;

impl ProofSerializer {
    /// Serialize proof to bytes
    pub fn serialize_proof(proof: &ZkStarkProof) -> Result<Vec<u8>> {
        serde_json::to_vec(proof).map_err(|e| CircuitError::Serialization(e))
    }
    
    /// Deserialize proof from bytes
    pub fn deserialize_proof(bytes: &[u8]) -> Result<ZkStarkProof> {
        serde_json::from_slice(bytes).map_err(|e| CircuitError::Serialization(e))
    }
    
    /// Compress proof data
    pub fn compress_proof(proof: &ZkStarkProof) -> Result<Vec<u8>> {
        let serialized = Self::serialize_proof(proof)?;
        
        // TODO: Implement actual compression (e.g., using zstd)
        // For now, return uncompressed data
        Ok(serialized)
    }
    
    /// Decompress proof data
    pub fn decompress_proof(compressed: &[u8]) -> Result<ZkStarkProof> {
        // TODO: Implement actual decompression
        // For now, assume uncompressed data
        Self::deserialize_proof(compressed)
    }
    
    /// Calculate proof hash
    pub fn hash_proof(proof: &ZkStarkProof) -> Result<Vec<u8>> {
        use blake3::Hasher;
        
        let serialized = Self::serialize_proof(proof)?;
        let mut hasher = Hasher::new();
        hasher.update(&serialized);
        Ok(hasher.finalize().as_bytes().to_vec())
    }
}

/// Circuit testing utilities
pub struct CircuitTester;

impl CircuitTester {
    /// Generate test inputs for a circuit
    pub fn generate_test_inputs(circuit_id: CircuitId, input_size: usize) -> CircuitInputs {
        CircuitInputs {
            public_inputs: FieldUtils::random_field_elements(input_size),
            private_inputs: FieldUtils::random_field_elements(input_size / 2),
            auxiliary_inputs: {
                let mut aux = HashMap::new();
                aux.insert("test_data".to_string(), FieldUtils::random_field_elements(10));
                aux
            },
        }
    }
    
    /// Validate circuit implementation
    pub fn validate_circuit_implementation(
        circuit: &dyn crate::Circuit,
        test_cases: usize,
    ) -> Result<ValidationResult> {
        let mut successful_tests = 0;
        let mut failed_tests = 0;
        let mut total_proving_time = std::time::Duration::ZERO;
        let mut total_verification_time = std::time::Duration::ZERO;
        
        for _ in 0..test_cases {
            let inputs = Self::generate_test_inputs(circuit.get_id(), 100);
            
            // Validate inputs
            match circuit.validate_inputs(&inputs) {
                Ok(_) => {
                    // Generate trace
                    let start_time = std::time::Instant::now();
                    match circuit.generate_trace(&inputs) {
                        Ok(_trace) => {
                            total_proving_time += start_time.elapsed();
                            successful_tests += 1;
                        }
                        Err(_) => failed_tests += 1,
                    }
                }
                Err(_) => failed_tests += 1,
            }
        }
        
        Ok(ValidationResult {
            total_tests: test_cases,
            successful_tests,
            failed_tests,
            success_rate: successful_tests as f64 / test_cases as f64,
            average_proving_time: total_proving_time / test_cases as u32,
            average_verification_time: total_verification_time / test_cases as u32,
        })
    }
}

/// Circuit validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub total_tests: usize,
    pub successful_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub average_proving_time: std::time::Duration,
    pub average_verification_time: std::time::Duration,
}

/// Performance benchmarking utilities
pub struct PerformanceBenchmark;

impl PerformanceBenchmark {
    /// Benchmark circuit performance
    pub fn benchmark_circuit(
        circuit: &dyn crate::Circuit,
        iterations: usize,
    ) -> BenchmarkResult {
        let mut proving_times = Vec::new();
        let mut memory_usage = Vec::new();
        
        for _ in 0..iterations {
            let inputs = CircuitTester::generate_test_inputs(circuit.get_id(), 100);
            
            let start_time = std::time::Instant::now();
            let start_memory = Self::get_memory_usage();
            
            if let Ok(_trace) = circuit.generate_trace(&inputs) {
                let proving_time = start_time.elapsed();
                let end_memory = Self::get_memory_usage();
                
                proving_times.push(proving_time);
                memory_usage.push(end_memory - start_memory);
            }
        }
        
        BenchmarkResult {
            iterations,
            average_proving_time: Self::calculate_average_duration(&proving_times),
            min_proving_time: proving_times.iter().min().copied().unwrap_or_default(),
            max_proving_time: proving_times.iter().max().copied().unwrap_or_default(),
            average_memory_usage: memory_usage.iter().sum::<usize>() / memory_usage.len().max(1),
            complexity: circuit.get_complexity(),
        }
    }
    
    /// Get current memory usage (placeholder implementation)
    fn get_memory_usage() -> usize {
        // TODO: Implement actual memory usage measurement
        0
    }
    
    /// Calculate average duration
    fn calculate_average_duration(durations: &[std::time::Duration]) -> std::time::Duration {
        if durations.is_empty() {
            return std::time::Duration::ZERO;
        }
        
        let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
        std::time::Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub iterations: usize,
    pub average_proving_time: std::time::Duration,
    pub min_proving_time: std::time::Duration,
    pub max_proving_time: std::time::Duration,
    pub average_memory_usage: usize,
    pub complexity: CircuitComplexity,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_field_element_conversion() {
        let bytes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let elements = FieldUtils::bytes_to_field_elements(&bytes);
        let converted_back = FieldUtils::field_elements_to_bytes(&elements);
        
        // Should be equal up to padding
        assert_eq!(&converted_back[..bytes.len()], &bytes);
    }
    
    #[test]
    fn test_polynomial_evaluation() {
        let coefficients = vec![
            BaseElement::ONE,
            BaseElement::new(2),
            BaseElement::new(3),
        ];
        let point = BaseElement::new(2);
        
        // 1 + 2*2 + 3*4 = 17
        let result = FieldUtils::evaluate_polynomial(&coefficients, point);
        assert_eq!(result, BaseElement::new(17));
    }
}
