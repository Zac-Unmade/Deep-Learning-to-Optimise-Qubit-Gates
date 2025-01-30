from utils import fidelity, percentage_difference, percentage_contribution, single_gate_counts

"""
Qiskit packages for quantum simulations
"""
import qiskit
from qiskit import pulse, QuantumCircuit, QuantumRegister, execute, Aer, transpile, assemble
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import library, DriveChannel, ControlChannel, Schedule, Play, Waveform, Delay, Acquire, MemorySlot, AcquireChannel
from qiskit.pulse.library import Gaussian, Drag, drag, GaussianSquare
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.fake_provider import FakeValencia

print("Qiskit version:", qiskit.__version__)
print("Qiskit Aer version:", qiskit.providers.aer.__version__)

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
A break down of the original Pauli-X gate, such that we can use the same parameters as Qiskit other than the amplitude.
"""

original_x = QuantumCircuit(1)
original_x.x(0)
original_x_transpile = transpile(original_x, realistic_backend)
original_x_sched = schedule(original_x_transpile, realistic_backend)
original_x_sched

"""Define the custom Pauli-X gate."""

def custom_X_gate_circuit(backend):
    numOfQubits = 1
    realistic_circ = QuantumCircuit(numOfQubits)
    idealistic_circ = QuantumCircuit(numOfQubits)
    amp = random.uniform(0.000001, 0.999999)
    with pulse.build(backend, name='X Gate') as x_q0:
        pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=-0.25388969010654494, angle=0.0), pulse.drive_channel(0))
    data = [amp]
    custom_x_gate = Gate('X Gate', 1, [])
    realistic_circ.append(custom_x_gate, [0])
    realistic_circ.add_calibration(custom_x_gate, [0], x_q0)
    idealistic_circ.x(0)
    with pulse.build(backend, name='Measure') as measure:
        pulse.acquire(120, pulse.acquire_channel(0), MemorySlot(0))
    custom_m_gate = Gate('Measure', 1, [])
    realistic_circ.append(custom_m_gate, [0])
    realistic_circ.add_calibration(custom_m_gate, [0], measure)
    return realistic_circ, idealistic_circ, data

custom_X_gate_circuit(realistic_backend)

"""Transpile and schedule the custom Pauli-X gate."""
x_circuit = custom_X_gate_circuit(realistic_backend)
random_x_circ_transpile = transpile(x_circuit[0], realistic_backend)
random_x_circ_sched = schedule(random_x_circ_transpile, realistic_backend)
random_x_circ_sched.draw()

"""Run the custom Pauli-X gate and time it for analysing the how long it takes to run of different devices. Then show the count of states in a histogram."""
x_job = realistic_backend.run(random_x_circ_sched, shots=1024)
x_start_time = time.time()
x_counts = x_job.result().get_counts()
x_end_time = time.time()
x_elapsed_time = x_end_time - x_start_time
print(x_elapsed_time)

plot_histogram(x_counts)

"""Put counts of states in an array, then convert it to a percentage."""

realistic_x_gate_array = single_gate_counts(x_counts)
print(realistic_x_gate_array)

x_gate_real_percent_array = percentage_contribution(realistic_x_gate_array)
print(x_gate_real_percent_array)

ideal_x_counts = [0, 1024]

x_gate_ideal_percent_array = percentage_contribution(ideal_x_counts)
print(x_gate_ideal_percent_array)

"""Calculate fidelity between the custom Pauli-X gate count and the ideal gate count."""

x_gate_fidelity_value = fidelity(x_gate_ideal_percent_array, x_gate_real_percent_array)
print("Hadamard fidelity:", x_gate_fidelity_value)

"""The whole process in one function that can be used to gather data."""

def custom_x_gate_process():
    customX = custom_X_gate_circuit(realistic_backend)
    realistic_circ = customX[0]
    realistic_data = customX[2]
    x_circ_transpile = transpile(realistic_circ, realistic_backend)
    x_circ_sched = schedule(x_circ_transpile, realistic_backend)
    job = realistic_backend.run(x_circ_sched, shots=1024)
    x_counts = job.result().get_counts()
    realistic_x_gate_array = single_gate_counts(x_counts)
    x_gate_real_percent_array = percentage_contribution(realistic_x_gate_array)
    ideal_x_counts = [0, 1024]
    x_gate_ideal_percent_array = percentage_contribution(ideal_x_counts)
    x_gate_fidelity_value = fidelity(x_gate_ideal_percent_array, x_gate_real_percent_array)
    data = [realistic_data, [x_gate_fidelity_value]]
    return data

custom_x_gate_process()

"""## Gather Data"""

num_runs = 1000000
folder = '/home/lunet/phzf/Documents'
filename = 'x_gate_data.csv'
file_path = os.path.join(folder, filename)
with open(file_path, 'a', newline='') as f:
    write = csv.writer(f)
    for i in range(num_runs):
        array = custom_x_gate_process()
        flattened_array = [str(i) for sublist in array for i in sublist]
        write.writerow(flattened_array)
