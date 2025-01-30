from utils import fidelity, percentage_difference, percentage_contribution, single_gate_counts

"""
Qiskit packages for quantum simulations
"""
import qiskit
from qiskit import *
from qiskit import pulse, QuantumCircuit, QuantumRegister, execute, Aer, transpile, assemble
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import library, DriveChannel, ControlChannel, Schedule, Play, Waveform, Delay, Acquire, MemorySlot, AcquireChannel
from qiskit.pulse.library import Gaussian, Drag, drag, GaussianSquare
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.providers.aer.noise.errors import thermal_relaxation_error, coherent_unitary_error
from qiskit.providers.fake_provider import FakeArmonk, FakeValencia, FakeHanoi

print("Qiskit version:", qiskit.__version__)
print("Qiskit Aer version:", qiskit.providers.aer.__version__)

"""Tensorflow libraries for the deep learning."""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import math
import csv
import os


"""
The backends required for running the simulations on.
"""
realistic_backend = FakeValencia()
idealistic_backend = Aer.get_backend('qasm_simulator')

"""
Break down generic hadamard gate, to get generic Qiskit parameters
"""
original_hadamard = QuantumCircuit(1)
original_hadamard.h(0)
original_hadamard_transpile = transpile(original_hadamard, realistic_backend)
original_hadamard_sched = schedule(original_hadamard_transpile, realistic_backend)
original_hadamard_sched


def custom_hadamard_circuit(backend):
    """
    Create a custom Hadamard circuit with both realistic and idealized implementations.
    Parameters:
    backend (Backend): The backend used for pulse-based operations and circuit execution.
    Returns:
    tuple: A tuple containing three elements:
        - realistic_circ (QuantumCircuit): The realistic version of the Hadamard circuit.
        - idealistic_circ (QuantumCircuit): The ideal version of the Hadamard circuit.
        - data (list): A list containing the amplitude used in the realistic gate.
    """
    numOfQubits = 1
    realistic_circ = QuantumCircuit(numOfQubits)
    idealistic_circ = QuantumCircuit(numOfQubits)
    amp = random.uniform(0.000001, 0.999999)
    with pulse.build(backend, name='Hadamard') as h_q0:
        pulse.shift_phase(-1.5707963268, pulse.drive_channel(0))
        pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=-0.35835396095069005, angle=0.008783280252964184), pulse.drive_channel(0))
        pulse.shift_phase(1.57, pulse.drive_channel(0))
    data = [amp]
    custom_h_gate = Gate('Hadamard', 1, [])
    realistic_circ.append(custom_h_gate, [0])
    realistic_circ.add_calibration(custom_h_gate, [0], h_q0)
    idealistic_circ.h(0)
    with pulse.build(backend, name='Measure') as measure:
        pulse.acquire(120, pulse.acquire_channel(0), MemorySlot(0))
    custom_m_gate = Gate('Measure', 1, [])
    realistic_circ.append(custom_m_gate, [0])
    realistic_circ.add_calibration(custom_m_gate, [0], measure)
    return realistic_circ, idealistic_circ, data

custom_hadamard_circuit(realistic_backend)

"""Transpile and schedule the custom Hadamard gate."""

had_circ = custom_hadamard_circuit(realistic_backend)
random_h_circ_transpile = transpile(had_circ[0], realistic_backend)
random_h_circ_sched = schedule(random_h_circ_transpile, realistic_backend)
random_h_circ_sched.draw()

"""Run the custom Hadamard gate and time it for analysing the how long it takes to run of different devices. Then show the count of states in a histogram."""

job = realistic_backend.run(random_h_circ_sched, shots=1024)
start_time = time.time()
h_counts = job.result().get_counts()
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

plot_histogram(h_counts)

"""Put counts of states in an array, then convert it to a percentage."""

realistic_hadamard_array = single_gate_counts(h_counts)
print(realistic_hadamard_array)

real_percent_hadamard = percentage_contribution(realistic_hadamard_array)
print(real_percent_hadamard)

ideal_h_counts = [512, 512]

ideal_percent_hadamard = percentage_contribution(ideal_h_counts)
print(ideal_percent_hadamard)

"""Calculate fidelity between the custom Hadamard gate count and the ideal gate count."""

hadamard_fidelity_value = fidelity(ideal_percent_hadamard, real_percent_hadamard)
print("Hadamard fidelity:", hadamard_fidelity_value)

"""The whole process in one function that can be used to gather data."""

def custom_hadamard_process():
    """
    Execute and analyze a custom Hadamard gate process using a realistic quantum circuit.
    The process involves:
    1. Creating realistic and ideal Hadamard circuits.
    2. Transpiling and scheduling the realistic circuit.
    3. Running the scheduled circuit on the specified backend.
    4. Collecting and analyzing the results to calculate fidelity between ideal and 
       realistic results.
    Returns:
    list: A list containing:
        - realistic_data (list): The amplitude used in the realistic Hadamard gate.
        - process_data (list): A list containing the fidelity value between ideal 
          and realistic implementations of the Hadamard gate.
    """
    customHad = custom_hadamard_circuit(realistic_backend)
    realistic_circ = customHad[0]
    idealistic_circ = customHad[1]
    realistic_data = customHad[2]
    hadamard_circ_transpile = transpile(realistic_circ, realistic_backend)
    hadamard_circ_sched = schedule(hadamard_circ_transpile, realistic_backend)
    job = realistic_backend.run(hadamard_circ_sched, shots=1024)
    start_time = time.time()
    counts = job.result().get_counts()
    end_time = time.time()
    hadamard_time = end_time - start_time
    realistic_hadamard_array = single_gate_counts(counts)
    real_percent_hadamard = percentage_contribution(realistic_hadamard_array)
    ideal_h_counts = [512, 512]
    real_percent_hadamard = percentage_contribution(realistic_hadamard_array)
    hadamard_fidelity_value = fidelity(ideal_percent_hadamard, real_percent_hadamard)
    process_data = [realistic_data, [hadamard_fidelity_value]]
    return process_data

custom_hadamard_process()

"""Gather Data"""
num_runs = 1000000
folder = './data'
filename = 'hadamard_data.csv'
file_path = os.path.join(folder, filename)
os.makedirs(folder, exist_ok=True)
with open(file_path, 'a', newline='') as f:
    write = csv.writer(f)
    for i in range(num_runs):
        array = custom_hadamard_process()
        flattened_array = [str(i) for sublist in array for i in sublist]
        write.writerow(flattened_array)
