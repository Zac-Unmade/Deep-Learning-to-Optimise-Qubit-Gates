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

"""Preprocess Data"""
folder = './data' 
filename = 'x_gate_data.csv'
file_path = os.path.join(folder, filename)
x_df = pd.read_csv(file_path)
x_amplitude = x_df.iloc[:15000, 0]
x_fidelity = x_df.iloc[:15000, 1]
x_dataset = pd.DataFrame({'Amplitude': x_amplitude, 'Fidelity': x_fidelity})
x_dataset.head()

x_dataset.isna().sum()
x_dataset = x_dataset.dropna()

"""Inspect the dataset: view the joint distribution of paired data."""

data_names = ['Amplitude','Fidelity']
sns.pairplot(x_dataset[data_names], diag_kind='kde')

"""Defining the independent, X, and dependent, y, data parameters. Next splitting the data. Then standardize the features."""

X = x_dataset[['Amplitude']]
y = x_dataset['Fidelity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""### Train and Test

Build a DNN model. Compile model. Train the model and evalute it.
"""

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
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

"""Plot the training and validation curves for loss vs epochs."""

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

"""R-Squared calculation for the curves above."""

mean_training_loss = np.mean(training_loss)
mean_validation_loss = np.mean(validation_loss)
ssr_training = np.sum((training_loss - mean_training_loss)**2)
ssr_validation = np.sum((validation_loss - mean_validation_loss)**2)
sst = ssr_training + ssr_validation
r_squared = 1 - (ssr_validation/sst)
print(f"R-squared: {r_squared:.4f}")

"""### Predictions

Create an array of amplidudes and then create an array of predicted values for each amplitude using our DNN model.
"""

amp = np.arange(0,1,0.001)
fid = []
for i in range(len(amp)):
    value = amp[i]
    a = np.array([[value]])
    f = model.predict(scaler.transform(a))
    fid.append(f[0][0])

"""Plot the predicted fidelity against amplitude."""

plt.plot(amp,fid)
plt.grid()
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("Amplitude vs Fidelity", fontdict = font1)
plt.xlabel("Amplitude", fontdict = font2)
plt.ylabel("Fidelity", fontdict = font2)
plt.show()

"""Select the highest predicted fideliy and find the corresponding amplitude."""

highest_fid = max(fid)
amp_with_highest_fid = amp[fid.index(highest_fid)]
print(amp_with_highest_fid)

"""### Fine-Tuning for a More Precise Amplitude

Refine the range that the amplitude values take, to be around the previously found value and make the steps finer. Then find the the highest predicted fidelity for these amplitudes, and then take the corresponding amplitude to that fidelity as the optimum.
"""

x_lower_bound = amp_with_highest_fid - 0.001
x_high_bound = amp_with_highest_fid + 0.001
finer_amp = np.arange(x_lower_bound,x_high_bound, 0.000001)
finer_fid = []
for i in range(len(finer_amp)):
    value = finer_amp[i]
    a = np.array([[value]])
    f = model.predict(scaler.transform(a))
    finer_fid.append(f[0][0])

x_highest_finer_fid = max(finer_fid)
x_finer_amp_with_highest_fid = finer_amp[finer_fid.index(highest_finer_fid)]
print(x_finer_amp_with_highest_fid)

"""### Check Result

Use the refined ampitude obtained above to check that it is a valid and optimized result.
"""

realistic_backend = FakeValencia()
idealistic_backend = Aer.get_backend('qasm_simulator')

def check_X_gate_circuit(backend, x_amplitude):
    numOfQubits = 1
    realistic_circ = QuantumCircuit(numOfQubits)

    with pulse.build(backend, name='X Gate') as x_q0:
        pulse.play(Drag(duration=160, amp=x_amplitude, sigma=40, beta=-0.25388969010654494, angle=0.0), pulse.drive_channel(0))
    data = [amp]

    custom_x_gate = Gate('X Gate', 1, [])
    realistic_circ.append(custom_x_gate, [0])
    realistic_circ.add_calibration(custom_x_gate, [0], x_q0)

    with pulse.build(backend, name='Measure') as measure:
        pulse.acquire(120, pulse.acquire_channel(0), MemorySlot(0))
    custom_m_gate = Gate('Measure', 1, [])
    realistic_circ.append(custom_m_gate, [0])
    realistic_circ.add_calibration(custom_m_gate, [0], measure)

    return realistic_circ, data

check_X_gate_circuit(realistic_backend, x_finer_amp_with_highest_fid)

check_x_circuit = check_X_gate_circuit(realistic_backend, amp_with_highest_fid)
check_x_circ_transpile = transpile(x_circuit[0], realistic_backend)
check_x_circ_sched = schedule(random_x_circ_transpile, realistic_backend)
check_x_circ_sched.draw()

x_job = realistic_backend.run(random_x_circ_sched, shots=1024)
x_counts = x_job.result().get_counts()
plot_histogram(x_counts)
