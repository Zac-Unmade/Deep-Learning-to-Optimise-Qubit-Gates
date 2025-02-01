from utils import fidelity, percentage_difference, percentage_contribution, single_gate_counts

import qiskit
from qiskit import pulse, QuantumCircuit, QuantumRegister, execute, Aer, transpile, assemble
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import library, DriveChannel, ControlChannel, Schedule, Play, Waveform, Delay, Acquire, MemorySlot, AcquireChannel
from qiskit.pulse.library import Gaussian, Drag, drag, GaussianSquare
from qiskit.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.fake_provider import FakeValencia

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import math
import csv
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
from tensorflow import keras


"""Preprocess Data
"""

os.chdir("/home/lunet/phzf/Downloads")
cnot_df = pd.read_csv('cnot_data.csv')
cnot_amplitude_1 = cnot_df.iloc[:14000, 0]
cnot_amplitude_2 = cnot_df.iloc[:14000, 1]
cnot_amplitude_3 = cnot_df.iloc[:14000, 2]
cnot_fidelity = cnot_df.iloc[:14000, 3]
cnot_dataset = pd.DataFrame({'Amplitude 1': cnot_amplitude_1,
                          'Amplitude 2': cnot_amplitude_2,
                          'Amplitude 3': cnot_amplitude_3,
                          'Fidelity': cnot_fidelity})
cnot_dataset.head()

cnot_dataset.isna().sum()

cnot_dataset = cnot_dataset.dropna()

data_names = ['Amplitude 1','Amplitude 2','Amplitude 3','Fidelity']
sns.pairplot(cnot_dataset[data_names], diag_kind='kde')

X = cnot_dataset[['Amplitude 1','Amplitude 2','Amplitude 3']]
y = cnot_dataset['Fidelity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""###  Train and Test"""

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(128, activation='tanh'),
    layers.Dense(256, activation='tanh'),
    layers.Dense(128, activation='tanh'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_scaled, y_train, epochs=60, batch_size=32, verbose=1, validation_split=0.2)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse:.4f}")

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
plt.figure(figsize=(10,5))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

amp1 = np.arange(0,1,0.1)
amp2 = np.arange(0,1,0.1)
amp3 = np.arange(0,1,0.1)

max_prediction = -float('inf')
best_combination = None

for element1 in amp1:
    for element2 in amp2:
        for element3 in amp3:
            amp_vector = np.array([[element1, element2, element3]])
            prediction = model.predict(scaler.transform(amp_vector))[0][0]
            if prediction > max_prediction:
                max_prediction = prediction
                best_combination = (element1, element2, element3)

print("Best Combination:", best_combination)
print("Max Prediction:", max_prediction)

amp1_refined = np.arange(best_combination[0]-0.1,best_combination[0]+0.1,0.01)
amp2_refined = np.arange(best_combination[1]-0.1,best_combination[1]+0.1,0.01)
amp3_refined = np.arange(best_combination[2]-0.1,best_combination[2]+0.1,0.01)

refined_max_prediction = -float('inf')
best_refined_combination = None

for element1 in amp1_refined:
    for element2 in amp2_refined:
        for element3 in amp3_refined:
            amp_vector = np.array([[element1, element2, element3]])
            prediction = model.predict(scaler.transform(amp_vector))[0][0]
            if prediction > refined_max_prediction:
                refined_max_prediction = prediction
                best_refined_combination = (element1, element2, element3)

print("Best Combination:", best_refined_combination)
print("Max Prediction:", refined_max_prediction)

"""### Check results"""

def check_CNOT_circuit(backend, amplitudes):
    numOfQubits = 2
    realistic_circ = QuantumCircuit(numOfQubits)

    amp1 = amplitudes[0]
    amp2 = amplitudes[1]
    amp3 = amplitudes[2]
    with pulse.build(backend, name='CNOT') as cnot_gate:
        pulse.shift_phase(1.5707963267948966, pulse.drive_channel(0))
        pulse.play(Drag(duration=160, sigma=40, beta=-0.25388969010654494, amp=amp1, angle=-1.5707963267948968), pulse.drive_channel(0))#0.19290084722113582
        pulse.play(Drag(duration=160, sigma=40, beta=-0.5196057292826135, amp=amp2, angle=0.007898523847627245), pulse.drive_channel(1))#0.0746414463895804
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=0.06740617502322209, angle=0.023765720478682046), pulse.drive_channel(1))
        pulse.delay(duration=160, channel=ControlChannel(0))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=0.37716020465737493, angle=0.4451938954606395), ControlChannel(0))
        pulse.delay(duration=432, channel=pulse.drive_channel(0))
        pulse.play(Drag(duration=160, sigma=40, beta=-0.25388969010654494, amp=amp3, angle=0.0), pulse.drive_channel(0))#0.19290084722113582
        pulse.delay(duration=160, channel=pulse.drive_channel(1))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=0.06740617502322209, angle=-3.1178269331111115), pulse.drive_channel(1))
        pulse.delay(duration=160, channel=ControlChannel(0))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=0.3771602046573749, angle=-2.696398758129154), ControlChannel(0))
    data = [amp1, amp2, amp3]

    custom_C_gate = Gate('CNOT', 2, [])
    realistic_circ.append(custom_C_gate, [0,1])
    realistic_circ.add_calibration(custom_C_gate, [0,1], cnot_gate)

    with pulse.build(backend, name='Measure') as measure:
        pulse.acquire(120, pulse.acquire_channel(0), MemorySlot(0))
        pulse.acquire(120, pulse.acquire_channel(1), MemorySlot(1))
    custom_m_gate = Gate('Measure', 2, [])
    realistic_circ.append(custom_m_gate, [0,1])
    realistic_circ.add_calibration(custom_m_gate, [0,1], measure)
    return realistic_circ, data

check_CNOT_circuit(realistic_backend, best_refined_combination)

cnot_check = check_CNOT_circuit(realistic_backend, best_refined_combination)
CNOT_check_transpile = transpile(cnot_check[0], realistic_backend)
CNOT_check_sched = schedule(CNOT_check_transpile, realistic_backend)
CNOT_check_sched.draw()

check_CNOT_job = realistic_backend.run(CNOT_check_sched, shots=1024)
check_CNOT_counts = check_CNOT_job.result().get_counts()
plot_histogram(check_CNOT_counts)
