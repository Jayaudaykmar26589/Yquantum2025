# main.py
# Contains the core quantum hash function for the YQuantum 2025 challenge.

import math
import numpy as np
# Import for Aer simulator
from qiskit_aer import Aer
# Core Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
import time # Optional: for timing in example

# --- Core Quantum Hash Function ---

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
    num_qubits = min(20, max(8, int(math.log2(num_input_bytes)) + 4))

    # 2. Parameter Generation (More Complex Mapping)
    num_layers = 4 # Number of layers in the HEA circuit
    num_params_per_layer = num_qubits * 3 # RX, RY, RZ per qubit
    total_params_needed = num_layers * num_params_per_layer

    param_values_list = []
    input_len = len(input_bytes)
    param_idx = 0
    # Generate parameters based on input bytes
    for i in range(total_params_needed):
        byte1_idx = (param_idx * 2) % input_len # Use different indexing
        byte2_idx = (param_idx * 2 + 1) % input_len
        # Ensure indices are different if possible, handle short inputs
        if byte1_idx == byte2_idx:
             byte2_idx = (byte1_idx + 1) % input_len

        byte1 = input_bytes[byte1_idx]
        byte2 = input_bytes[byte2_idx]

        combined_value = (byte1 << 8) | byte2 # Combine into a 16-bit value
        # Scale angle, ensure non-zero rotation for non-zero input
        angle = (combined_value / 65535.0) * 2 * math.pi + (0.01 if combined_value > 0 else 0)
        param_values_list.append(angle)
        param_idx += 1

    # 3. Quantum Circuit Construction (Hardware Efficient Ansatz)
    qc = QuantumCircuit(num_qubits)
    qiskit_params = [] # List to hold Qiskit Parameter objects

    # Build the circuit layer by layer
    for layer in range(num_layers):
        # Single-qubit rotation layer
        for qubit in range(num_qubits):
            p_rx = Parameter(f'rx_{layer}_{qubit}')
            p_ry = Parameter(f'ry_{layer}_{qubit}')
            p_rz = Parameter(f'rz_{layer}_{qubit}')
            qiskit_params.extend([p_rx, p_ry, p_rz])
            qc.rx(p_rx, qubit)
            qc.ry(p_ry, qubit)
            qc.rz(p_rz, qubit)

        # Entanglement layer (circular CNOTs for strong entanglement)
        if num_qubits > 1:
            for i in range(num_qubits):
                 qc.cx(i, (i + 1) % num_qubits) # Circular entanglement

        # Add barrier for visual separation and potential transpiler hints
        if layer < num_layers - 1:
            qc.barrier()

    # Create the parameter binding dictionary mapping Parameter objects to values
    param_binding = {p: val for p, val in zip(qiskit_params, param_values_list)}

    # Bind parameters to the circuit
    bound_qc = qc.assign_parameters(param_binding)

    # 4. Quantum Simulation
    # Use statevector simulator for exact expectation values
    simulator = Aer.get_backend('statevector_simulator')
    # Transpile the circuit for the simulator backend with optimization
    # Optimization level 2 provides good balance of optimization and compile time
    transpiled_qc = transpile(bound_qc, simulator, optimization_level=2)
    # Execute the circuit
    result = simulator.run(transpiled_qc).result()
    statevector = result.get_statevector()

    # 5. Measurement and Post-Processing (Multi-Basis & Mixing)
    mixed_expectation_values = []
    # Define Pauli operators for expectation value calculation
    paulis = {'X': Pauli('X'), 'Y': Pauli('Y'), 'Z': Pauli('Z')}

    # Calculate expectation values for each qubit in X, Y, Z bases
    for i in range(num_qubits):
        exp_x = statevector.expectation_value(paulis['X'], [i]).real
        exp_y = statevector.expectation_value(paulis['Y'], [i]).real
        exp_z = statevector.expectation_value(paulis['Z'], [i]).real

        # Combine expectation values using a non-linear function
        # Weights ensure the result is roughly bounded; squaring adds non-linearity
        mixed_val = 0.5 * exp_x**2 + 0.3 * exp_y**2 + 0.2 * exp_z**2
        # Result is expected to be in [0, 1]
        mixed_expectation_values.append(mixed_val)

    # Scale mixed values from [0, 1] range to [0, 255] byte range
    scaled_values = []
    for val in mixed_expectation_values:
         # Clamp values just in case of floating point inaccuracies
         scaled = max(0, min(255, int(val * 255.0)))
         scaled_values.append(scaled)

    # Simple output mixing using XOR for diffusion
    mixed_output_values = list(scaled_values) # Create a copy
    num_vals = len(mixed_output_values)
    if num_vals > 1:
        temp_val = mixed_output_values[0] # Store for wrap-around XOR
        for i in range(num_vals - 1):
             mixed_output_values[i] ^= mixed_output_values[i+1] # XOR with next element
        mixed_output_values[num_vals-1] ^= temp_val # XOR last with original first

    # Convert the list of mixed integer values (0-255) into the final byte output
    # Ensure the output length matches the input length by cycling/repeating bits
    output_bytes = bytearray()
    value_idx = 0
    num_scaled_values = len(mixed_output_values)

    # Generate exactly num_input_bytes bytes for the output
    for _ in range(num_input_bytes):
        current_byte = 0
        # Collect 8 bits for the current output byte
        for bit_idx in range(8):
            # Select the value source, cycling through mixed_output_values
            source_val_idx = (value_idx * 8 + bit_idx) % num_scaled_values
            source_val = mixed_output_values[source_val_idx]

            # Select which bit to extract from the source value
            # Cycle through bits of the source value as well
            source_bit_idx = ((value_idx * 8 + bit_idx) // num_scaled_values) % 8
            bit = (source_val >> source_bit_idx) & 1

            # Add the extracted bit to the current byte
            current_byte |= (bit << bit_idx)

        output_bytes.append(current_byte)
        value_idx += 1 # Move to the next index for sourcing bits

    # Return the final hash as bytes
    return bytes(output_bytes)


# --- Example Usage (Optional - for testing main.py directly) ---
if __name__ == '__main__':
    print("--- Running Quantum Hash Function (main.py) ---")

    # Example input data (must be >= 32 bytes)
    input_data = b"This is a test input string for the quantum hash function." * 2
    if len(input_data) < 32:
         input_data += b" " * (32 - len(input_data)) # Pad if needed for example

    print(f"Input data length: {len(input_data)} bytes")
    # print(f"Input data (repr): {input_data!r}") # Uncomment to see input

    start_time = time.time()
    try:
        # Calculate the hash
        hashed_data = advanced_quantum_hash(input_data)
        end_time = time.time()

        # Print the results
        print(f"Output hash length: {len(hashed_data)} bytes")
        print(f"Output hash (hex): {hashed_data.hex()}")
        print(f"Hashing time: {end_time - start_time:.4f} seconds")

        # Basic determinism check
        print("\n--- Determinism Check ---")
        hashed_data_2 = advanced_quantum_hash(input_data)
        if hashed_data == hashed_data_2:
            print("Check Passed: Function is deterministic for the same input.")
        else:
            print("Check Failed: Function produced different outputs for the same input.")
            print(f"Output 1 (hex): {hashed_data.hex()}")
            print(f"Output 2 (hex): {hashed_data_2.hex()}")

    except ValueError as ve:
        print(f"Error: {ve}")
    except ImportError as ie:
         print(f"Import Error: {ie}. Please ensure Qiskit and Qiskit Aer are installed (`pip install qiskit qiskit-aer`).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

