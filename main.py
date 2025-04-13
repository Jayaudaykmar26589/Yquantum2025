import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli
import math

def quantum_hash(input_bytes: bytes) -> bytes:
    """
    A quantum hash function that satisfies the requirements for the Superquantum challenge.
    
    Args:
        input_bytes: Input byte array of size 2^N where N >= 5
        
    Returns:
        A byte array of the same size as the input
    """
    # Validate input length
    input_len = len(input_bytes)
    if input_len < 32 or (input_len & (input_len - 1)) != 0:  # Check if power of 2 and >= 32
        raise ValueError(f"Input length must be a power of 2 and at least 32 bytes. Got {input_len} bytes.")
    
    # Calculate number of qubits needed (capped at 20 to maintain feasibility)
    n = min(20, int(math.log2(input_len)))
    
    # Initialize the circuit
    qc = QuantumCircuit(n)
    
    # Apply Hadamard gates to create initial superposition
    for i in range(n):
        qc.h(i)
    
    # Apply parameterized gates in multiple layers for complexity
    num_layers = 3
    
    for layer in range(num_layers):
        # Apply parameterized rotation gates
        for i in range(n):
            # Use different bytes from input to determine angles
            # Scale byte values to be between 0 and 2π
            byte_idx = (i + layer * n) % input_len
            angle_x = (input_bytes[byte_idx] / 128.0) * np.pi
            
            # Apply RX gate
            qc.rx(angle_x, i)
            
            # Use another byte for RZ
            byte_idx = (i + layer * n + n) % input_len
            angle_z = (input_bytes[byte_idx] / 128.0) * np.pi
            qc.rz(angle_z, i)
        
        # Apply controlled operations for entanglement in each layer
        for i in range(n-1):
            qc.cx(i, (i+1) % n)
        
        # In odd layers, apply controlled operations in reverse order for improved mixing
        if layer % 2 == 1:
            for i in range(n-1, 0, -1):
                qc.cx(i, (i-1) % n)
                
        # Add a barrier for clarity
        qc.barrier()
        
        # Apply additional controlled phase gates for nonlinearity
        for i in range(0, n, 2):
            control = i
            target = (i + 1) % n
            # Use another byte to determine phase angle
            byte_idx = (i + layer * n + 2*n) % input_len
            angle_p = (input_bytes[byte_idx] / 256.0) * 2 * np.pi
            qc.cp(angle_p, control, target)
        
        # Apply a final permutation of qubits through swap gates to mix information
        for i in range(n//2):
            # Use another byte from input as a condition to apply swap
            byte_idx = (i + layer * n + 3*n) % input_len
            if input_bytes[byte_idx] % 2 == 1:  # Apply swap only if the byte is odd
                qc.swap(i, n-i-1)
    
    # Measure expectation values in different bases for richness
    sv = Statevector.from_instruction(qc)
    
    # Generate output by measuring in multiple bases (X, Y, Z)
    # to capture full quantum state information
    exp_vals_z = [sv.expectation_value(Pauli("Z"), [i]).real for i in range(n)]
    exp_vals_x = [sv.expectation_value(Pauli("X"), [i]).real for i in range(n)]
    exp_vals_y = [sv.expectation_value(Pauli("Y"), [i]).real for i in range(n)]
    
    # Combine measurements and stretch to fill output size
    all_exp_vals = exp_vals_z + exp_vals_x + exp_vals_y
    
    # Stretch expectation values to match input size
    stretched_vals = []
    for i in range(input_len):
        # Mix measurements from different bases using input bytes as weights
        idx = i % len(all_exp_vals)
        next_idx = (i + 1) % len(all_exp_vals)
        weight = input_bytes[i % input_len] / 255.0
        
        # Create a mixture of two expectation values
        mixed_val = all_exp_vals[idx] * weight + all_exp_vals[next_idx] * (1 - weight)
        
        # Map from [-1, 1] to [0, 255]
        byte_val = int(((mixed_val + 1) / 2) * 255)
        stretched_vals.append(byte_val)
    
    # Final mixing step to ensure avalanche effect
    output_bytes = bytearray(input_len)
    for i in range(input_len):
        # Each output byte depends on multiple input bytes
        idx1 = i % input_len
        idx2 = (i + n) % input_len
        idx3 = (i * 7 + 3) % input_len  # Non-linear index
        
        # XOR combined with rotation for better mixing
        mixed_val = (stretched_vals[idx1] ^ 
                     ((stretched_vals[idx2] << 3) | (stretched_vals[idx2] >> 5)) ^ 
                     ~stretched_vals[idx3]) & 0xFF
        
        output_bytes[i] = mixed_val
    
    return bytes(output_bytes)


def test_quantum_hash():
    """Test the quantum hash function with a simple example."""
    # Create test input (2^5 = 32 bytes)
    test_input = bytes(range(32))
    
    # Compute hash
    hash_output = quantum_hash(test_input)
    
    print(f"Input length: {len(test_input)} bytes")
    print(f"Output length: {len(hash_output)} bytes")
    print(f"Hash output: {list(hash_output)}")
    
    # Test determinism
    hash_output2 = quantum_hash(test_input)
    print(f"Determinism check: {hash_output == hash_output2}")
    
    # Test small change in input
    test_input_modified = bytearray(test_input)
    test_input_modified[0] = (test_input_modified[0] + 1) % 256
    hash_output_modified = quantum_hash(bytes(test_input_modified))
    
    # Count number of different bytes (avalanche effect)
    diff_count = sum(a != b for a, b in zip(hash_output, hash_output_modified))
    
    print(f"Bytes changed in output after 1 byte change in input: {diff_count}/{len(hash_output)}")
    print(f"Percentage changed: {diff_count/len(hash_output)*100:.2f}%")


if __name__ == "__main__":
    test_quantum_hash()
