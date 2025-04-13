import math
import numpy as np
# Updated import for Aer
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
import random
import time

# --- Helper Functions ---

def calculate_hamming_distance(bytes1, bytes2):
    """Calculates the Hamming distance between two byte strings."""
    distance = 0
    for b1, b2 in zip(bytes1, bytes2):
        xor_val = b1 ^ b2
        distance += bin(xor_val).count('1')
    # Account for length difference if any (shouldn't happen in this context)
    distance += abs(len(bytes1) - len(bytes2)) * 8
    return distance

def estimate_avalanche_effect(hash_func, input_size_bytes=32, num_samples=50):
    """Estimates the avalanche effect of the hash function."""
    print(f"\n--- Estimating Avalanche Effect (Input Size: {input_size_bytes} bytes, Samples: {num_samples}) ---")
    total_distance = 0
    # Correct calculation for total bits considering each sample involves two hashes
    # but the comparison is based on the output size which matches input_size_bytes
    total_bits_in_output = input_size_bytes * 8
    total_comparisons = num_samples

    start_time = time.time()
    for i in range(num_samples):
        # Generate random base input
        base_input = bytes([random.randint(0, 255) for _ in range(input_size_bytes)])
        base_hash = hash_func(base_input)

        # Flip one random bit
        bit_to_flip = random.randrange(input_size_bytes * 8)
        byte_index = bit_to_flip // 8
        bit_index = bit_to_flip % 8

        flipped_input_list = list(base_input)
        flipped_input_list[byte_index] ^= (1 << bit_index)
        flipped_input = bytes(flipped_input_list)

        flipped_hash = hash_func(flipped_input)

        # Ensure hashes have the same length for comparison
        # Since the function ensures output length == input length, min_len isn't strictly needed
        # but good practice if the function signature changed.
        min_len = min(len(base_hash), len(flipped_hash))
        distance = calculate_hamming_distance(base_hash[:min_len], flipped_hash[:min_len])
        total_distance += distance

        if (i + 1) % 10 == 0:
            print(f"Sample {i+1}/{num_samples} processed...")

    end_time = time.time()
    avg_distance = total_distance / total_comparisons
    # Avalanche percentage is the average fraction of bits flipped in the output
    avalanche_percentage = (avg_distance / total_bits_in_output) * 100
    print(f"Average Hamming Distance per change: {avg_distance:.2f} bits (out of {total_bits_in_output})")
    print(f"Avalanche Effect: {avalanche_percentage:.2f}% (Ideal is ~50%)")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    return avalanche_percentage

# --- Main Hash Function ---

def advanced_quantum_hash(input_bytes: bytes) -> bytes:
    """
    Advanced Quantum hash function for the YQuantum 2025 challenge.

    Uses HEA, multi-basis measurements, complex parameter mapping, and mixing.

    Args:
        input_bytes (bytes): The input byte array (min 32 bytes).

    Returns:
        bytes: The resulting hash value as a byte array (same size as input).
    """
    # 1. Input Validation and Qubit Calculation
    num_input_bytes = len(input_bytes)
    if num_input_bytes < 32: # 2**5 = 32
        raise ValueError("Input must be at least 32 bytes")

    # Determine number of qubits, capped at 20
    # More qubits generally allow for more complexity, but simulation cost increases
    # Using a function of input size up to the cap
    num_qubits = min(20, max(8, int(math.log2(num_input_bytes)) + 4)) # Example: Start at 8, increase slowly, cap at 20

    # 2. Parameter Generation (More Complex Mapping)
    parameters = []
    num_layers = 4 # Increased layers for more complexity
    num_params_per_layer = num_qubits * 3 # RX, RY, RZ per qubit

    param_values_list = []
    input_len = len(input_bytes)
    param_idx = 0
    for _ in range(num_layers * num_params_per_layer):
        # Combine two bytes for more complex parameter derivation
        byte1_idx = param_idx % input_len
        byte2_idx = (param_idx + input_len // 2) % input_len # Offset index
        byte1 = input_bytes[byte1_idx]
        byte2 = input_bytes[byte2_idx]

        # Combine bytes and scale to an angle (e.g., 0 to 2*pi)
        combined_value = (byte1 << 8) | byte2 # Combine into a 16-bit value
        angle = (combined_value / 65535.0) * 2 * math.pi
        param_values_list.append(angle)
        param_idx += 1

    # 3. Quantum Circuit Construction (Hardware Efficient Ansatz)
    qc = QuantumCircuit(num_qubits)
    qiskit_params = []
    param_iter = iter(param_values_list)

    for layer in range(num_layers):
        # Single-qubit rotation layer
        for qubit in range(num_qubits):
            # Use unique Parameter objects for binding
            p_rx = Parameter(f'rx_{layer}_{qubit}')
            p_ry = Parameter(f'ry_{layer}_{qubit}')
            p_rz = Parameter(f'rz_{layer}_{qubit}')
            qiskit_params.extend([p_rx, p_ry, p_rz])
            qc.rx(p_rx, qubit)
            qc.ry(p_ry, qubit)
            qc.rz(p_rz, qubit)

        # Entanglement layer (e.g., linear or circular CNOTs)
        if num_qubits > 1:
            for i in range(0, num_qubits - 1, 2): # Pairwise CNOTs
                 qc.cx(i, i + 1)
            # Connect remaining qubits in a circular fashion for better entanglement
            if num_qubits > 2:
                 qc.cx(num_qubits -1, 0) # Connect last to first
                 if num_qubits % 2 != 0: # If odd number of qubits, connect last pair
                     qc.cx(num_qubits - 2, num_qubits - 1)

        qc.barrier() # Separate layers visually and potentially for transpiler

    # Create the parameter binding dictionary
    param_binding = {p: val for p, val in zip(qiskit_params, param_values_list)}

    # Bind parameters
    bound_qc = qc.assign_parameters(param_binding)

    # 4. Quantum Simulation
    # Use statevector simulator for exact expectation values
    simulator = Aer.get_backend('statevector_simulator')
    # Transpile for the simulator with optimization
    transpiled_qc = transpile(bound_qc, simulator, optimization_level=2)
    result = simulator.run(transpiled_qc).result()
    statevector = result.get_statevector()

    # 5. Measurement and Post-Processing (Multi-Basis & Mixing)
    mixed_expectation_values = []
    paulis = {'X': Pauli('X'), 'Y': Pauli('Y'), 'Z': Pauli('Z')}

    for i in range(num_qubits):
        exp_x = statevector.expectation_value(paulis['X'], [i]).real
        exp_y = statevector.expectation_value(paulis['Y'], [i]).real
        exp_z = statevector.expectation_value(paulis['Z'], [i]).real

        # Combine expectation values (example: simple non-linear mix)
        # Squaring adds non-linearity, coefficients can be tuned
        # Ensure the weights sum to 1 or scale appropriately later
        mixed_val = 0.5 * exp_x**2 + 0.3 * exp_y**2 + 0.2 * exp_z**2
        # The result of this mix is expected to be in [0, 1] since exp_pauli^2 is in [0, 1]
        mixed_expectation_values.append(mixed_val)

    # Scale mixed values from [0, 1] to [0, 255]
    scaled_values = []
    for val in mixed_expectation_values:
         scaled = max(0, min(255, int(val * 255.0)))
         scaled_values.append(scaled)

    # Simple output mixing (e.g., XOR adjacent values with rotation)
    mixed_output_values = list(scaled_values) # Copy
    num_vals = len(mixed_output_values)
    if num_vals > 1:
        temp_val = mixed_output_values[0] # Store first value for wrap-around
        for i in range(num_vals - 1):
             # XOR with the next value
             mixed_output_values[i] ^= mixed_output_values[i+1]
        # XOR last value with the original first value
        mixed_output_values[num_vals-1] ^= temp_val

    # Convert potentially fewer scaled values (num_qubits) into target byte size
    output_bytes = bytearray()
    current_byte = 0
    bits_in_byte = 0
    value_idx = 0
    num_scaled_values = len(mixed_output_values)

    # Generate enough bits to fill the output byte array
    # We'll cycle through the mixed_output_values as needed
    total_bits_needed = num_input_bytes * 8
    generated_bits = 0

    while generated_bits < total_bits_needed:
        value_to_use = mixed_output_values[value_idx % num_scaled_values]
        # Determine how many bits to take from this value (up to 8)
        bits_to_take = min(8, 8 - bits_in_byte)

        # Extract bits (take lower bits first)
        mask = (1 << bits_to_take) - 1
        extracted_bits = (value_to_use >> (value_idx // num_scaled_values * 8)) & mask # Use different bits on reuse

        current_byte |= (extracted_bits << bits_in_byte)
        bits_in_byte += bits_to_take
        generated_bits += bits_to_take

        if bits_in_byte == 8:
            output_bytes.append(current_byte)
            current_byte = 0
            bits_in_byte = 0

        value_idx += 1 # Move to the next value/bit source

    # Ensure exact output length
    return bytes(output_bytes)


# --- Example Usage and Basic Analysis ---
if __name__ == '__main__':
    print("--- Running Advanced Quantum Hash Example ---")
    # Example usage
    input_data = b'This is a different test string for the advanced quantum hash function implementation.' * 5 # Ensure > 32 bytes
    print(f"Input data length: {len(input_data)} bytes")

    start_hash_time = time.time()
    hashed_data = advanced_quantum_hash(input_data)
    end_hash_time = time.time()

    print(f"Hashed data (hex): {hashed_data.hex()}")
    print(f"Hashed data length: {len(hashed_data)} bytes")
    print(f"Hashing time: {end_hash_time - start_hash_time:.4f} seconds")

    # Check determinism
    print("\n--- Determinism Check ---")
    hashed_data_2 = advanced_quantum_hash(input_data)
    print(f"Deterministic check passed: {hashed_data == hashed_data_2}")

    # Basic entropy check (Chi-squared)
    print("\n--- Basic Entropy Check (Chi-squared) ---")
    num_entropy_samples = 200 # Reduced samples for faster testing
    entropy_input_size = 32   # Use minimum input size for test
    byte_counts = {}
    total_output_bytes = 0
    entropy_start_time = time.time()
    for i in range(num_entropy_samples):
        random_data = bytes([random.randint(0, 255) for _ in range(entropy_input_size)])
        try:
            hashed_value = advanced_quantum_hash(random_data)
            total_output_bytes += len(hashed_value)
            for byte in hashed_value:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1
        except Exception as e:
            print(f"Error during entropy sample {i+1}: {e}")
            # Optionally continue or break
            continue
        if (i + 1) % 50 == 0:
             print(f"Entropy sample {i+1}/{num_entropy_samples} processed...")
    entropy_end_time = time.time()

    if total_output_bytes > 0 and len(byte_counts) > 0:
        expected_count = total_output_bytes / 256.0
        # Calculate Chi-squared only for observed byte values
        chi2 = sum((count - expected_count)**2 / expected_count for count in byte_counts.values())
        # Add contributions for unobserved byte values (count=0)
        chi2 += (256 - len(byte_counts)) * (expected_count**2 / expected_count) if expected_count > 0 else 0

        print(f"Chi-squared test result: {chi2:.2f} (Lower is better, indicates more uniform distribution)")
        print(f"Number of unique byte values observed: {len(byte_counts)} / 256")
    else:
        print("Not enough valid data generated for Chi-squared test.")
    print(f"Entropy test time: {entropy_end_time - entropy_start_time:.2f} seconds")

    # Estimate Avalanche Effect
    # Note: This can be slow as it runs the hash function 2*num_samples times
    estimate_avalanche_effect(advanced_quantum_hash, input_size_bytes=32, num_samples=50) # Reduced samples
