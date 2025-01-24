
"""Qiskit packages for quantum simulations"""
import qiskit
from qiskit import *
from qiskit import pulse, QuantumCircuit, QuantumRegister
from qiskit import execute, Aer, transpile, assemble
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

"""General libraries"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import math
import csv
import os

"""Common Functions"""

def percentage_difference(num1,num2):
    perc = abs((num1-num2)/num1)*100
    return perc

def single_gate_counts(counts_dict):
    state0 = 0
    state1 = 0
    if '0' in counts_dict:
        state0 = counts_dict['0']
    if '1' in counts_dict:
        state1 = counts_dict['1']
    return [state0, state1]

def percentage_contribution(array):
    total_sum = sum(array)
    percentages = [element / total_sum for element in array]
    return percentages

def fidelity(ideal_probs, realistic_probs):
    psi_ideal = np.sqrt(ideal_probs)
    psi_realistic = np.sqrt(realistic_probs)
    fidelity_value = abs(np.dot(psi_ideal, psi_realistic))**2
    return fidelity_value

"""Backends required for running the simulations on."""
realistic_backend = FakeValencia()
idealistic_backend = Aer.get_backend('qasm_simulator')

"""Hadamard Gate 

Simulation

A break down of the original hadamard gate, such that we can use the same parameters as Qiskit other than the amplitude.
"""

original_hadamard = QuantumCircuit(1)
original_hadamard.h(0)
original_hadamard_transpile = transpile(original_hadamard, realistic_backend)
original_hadamard_sched = schedule(original_hadamard_transpile, realistic_backend)
original_hadamard_sched

"""Define the custom Hadamard gate."""

def custom_hadamard_circuit(backend):
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

"""Run the custom Hadamard gate and time it for analysing the how long it takes to run of different devices. 
Then show the count of states in a histogram."""

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

"""## Gather Data"""

num_runs = 1000000
folder = '/home/lunet/phzf/Documents'
filename = 'hadamard_data.csv'
file_path = os.path.join(folder, filename)
with open(file_path, 'a', newline='') as f:
    write = csv.writer(f)
    for i in range(num_runs):
        array = custom_hadamard_process()
        flattened_array = [str(i) for sublist in array for i in sublist]
        write.writerow(flattened_array)

"""## Regression with Deep Neural Network

### Preprocess Data

Load and prepare Hadamard gate data. Make sure the file "hadamard_data.csv" is kept in the same folder as this Jupyter Notebook.
"""

from google.colab import drive
drive.mount('/content/drive')

"""### Old"""

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/hadamard_data.csv')
Amplitude = df.iloc[:10000, 0]
Fidelity = df.iloc[:10000, 1]
hadamard_dataset = pd.DataFrame({'Amplitude': Amplitude, 'Fidelity': Fidelity})
hadamard_dataset.head()

hadamard_dataset.isna().sum()

hadamard_dataset = hadamard_dataset.dropna()

"""Inspect the dataset: view the joint distribution of paired data."""

data_names = ['Amplitude','Fidelity']
sns.pairplot(hadamard_dataset[data_names], diag_kind='kde')

"""Defining the independent, X, and dependent, y, data parameters. Next splitting the data. Then standardize the features."""

X = hadamard_dataset[['Amplitude']]
y = hadamard_dataset['Fidelity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""### New (H & X Gates)"""

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/hadamard_data.csv')
# df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/x_gate_data.csv')

# Extracting the first 10000 rows for Amplitude and Fidelity
Amplitude = df.iloc[:10000, 0]
Fidelity = df.iloc[:10000, 1]

# Create a DataFrame with Amplitude and Fidelity
hadamard_dataset = pd.DataFrame({'Amplitude': Amplitude, 'Fidelity': Fidelity})

# Splitting initial validation set
validation_data = hadamard_dataset.iloc[:2000]
remaining_data = hadamard_dataset.iloc[2000:2125]

# Splitting remaining data into training and testing sets
X_remaining = remaining_data[['Amplitude']]
y_remaining = remaining_data['Fidelity']
X_train, X_test, y_train, y_test = train_test_split(X_remaining, y_remaining, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(validation_data[['Amplitude']])

amplitude = remaining_data['Amplitude']
fidelity = remaining_data['Fidelity']
plt.figure(figsize=(8, 6))
plt.scatter(amplitude, fidelity, color='blue', alpha=0.5)
plt.title('Amplitude vs Fidelity')
plt.xlabel('Amplitude')
plt.ylabel('Fidelity')
plt.grid(True)
plt.show()

"""### New (CNOT Gate)"""

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/cnot_data.csv')

# Extracting the first 10000 rows for Amplitude and Fidelity
Amplitude_1 = df.iloc[:10000, 0]
Amplitude_2 = df.iloc[:10000, 1]
Amplitude_3 = df.iloc[:10000, 2]
Fidelity = df.iloc[:10000, 3]

# Create a DataFrame with Amplitude and Fidelity
# hadamard_dataset = pd.DataFrame({'Amplitude': Amplitude, 'Fidelity': Fidelity})
cnot_dataset = pd.DataFrame({'Amplitude 1': Amplitude_1,
                          'Amplitude 2': Amplitude_2,
                          'Amplitude 3': Amplitude_3,
                          'Fidelity': Fidelity})

# Splitting initial validation set
validation_data = cnot_dataset.iloc[:2000]
remaining_data = cnot_dataset.iloc[5500:6000]

# Splitting remaining data into training and testing sets
X_remaining = remaining_data[['Amplitude 1','Amplitude 2','Amplitude 3']]
y_remaining = remaining_data['Fidelity']
X_train, X_test, y_train, y_test = train_test_split(X_remaining, y_remaining, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validation_scaled = scaler.transform(validation_data[['Amplitude 1','Amplitude 2','Amplitude 3']])

amplitude = remaining_data['Amplitude 1']
fidelity = remaining_data['Fidelity']
plt.figure(figsize=(8, 6))
plt.scatter(amplitude, fidelity, color='blue', alpha=0.5)
plt.title('Amplitude vs Fidelity')
plt.xlabel('Amplitude')
plt.ylabel('Fidelity')
plt.grid(True)
plt.show()

"""### Train and Test

### Old

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

"""### New"""

# Model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),
    layers.Dense(128, activation='tanh'),
    layers.Dense(256, activation='tanh'),
    layers.Dense(128, activation='tanh'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# model = keras.Sequential([
#     layers.Dense(16, activation='relu', input_shape=(3,)),
#     layers.Dense(32, activation='tanh'),
#     # layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='linear')  # Using linear activation for regression task
# ])

# Model compilation
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Model training
history = model.fit(X_train_scaled, y_train, epochs=60, batch_size=32, verbose=1, validation_split=0.2, validation_data=(X_validation_scaled, validation_data['Fidelity']))

# Model evaluation
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

from sklearn.metrics import r2_score

# Calculate R-squared
r_squared = r2_score(training_loss, validation_loss)
print(f"R-squared: {r_squared:.4f}")

from scipy.stats import spearmanr

# Calculate Spearman's correlation coefficient
spearman_corr, _ = spearmanr(training_loss, validation_loss)
print(f"Spearman's correlation coefficient: {spearman_corr:.4f}")

"""#### CNOT Prediction"""

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

amp1_refined = np.arange(best_combination[0]-0.05,best_combination[0]+0.05,0.01)
amp2_refined = np.arange(best_combination[1]-0.05,best_combination[1]+0.05,0.01)
amp3_refined = np.arange(best_combination[2]-0.05,best_combination[2]+0.05,0.01)

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

highest_finer_fid = max(finer_fid)
finer_amp_with_highest_fid = finer_amp[finer_fid.index(highest_finer_fid)]
print(finer_amp_with_highest_fid)

"""## Check Result

Use the refined ampitude obtained above to check that it is a valid and optimized result.
"""

def check_hadamard_circuit(backend, amplitude):
    numOfQubits = 1
    realistic_circ = QuantumCircuit(numOfQubits)
    idealistic_circ = QuantumCircuit(numOfQubits)

    phase = np.pi
    amp = amplitude
    with pulse.build(backend, name='Hadamard') as h_q0:
        pulse.shift_phase(phase, pulse.drive_channel(0))
        pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=-0.35835396095069005, angle=0.008783280252964184), pulse.drive_channel(0))
        pulse.shift_phase(phase, pulse.drive_channel(0))

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

check_hadamard_circuit(realistic_backend, finer_amp_with_highest_fid)

check_h_circuit = check_hadamard_circuit(realistic_backend, finer_amp_with_highest_fid)
check_h_circ_transpile = transpile(check_h_circuit[0], realistic_backend)
check_h_circ_sched = schedule(check_h_circ_transpile, realistic_backend)
check_h_circ_sched.draw()

had_job = realistic_backend.run(check_h_circ_sched, shots=1024)
h_counts = had_job.result().get_counts()
plot_histogram(h_counts)

"""As can be seen the above histogram, there is an approximate 50/50 distribution between states, which is what is expected for the optimum distribution for the Hadamard gate.

# Pauli-X Gate

## Simulation

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

"""## Regression with Deep Neural Network

### Preprocess Data

Load and prepare Pauli-X gate data.
"""

os.chdir("/home/lunet/phzf/Downloads")
x_df = pd.read_csv('x_gate_data.csv')
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

"""As can be seen the above histogram, there is an approximate 0% and 100% distribution between states, which is what is expected for the optimum distribution for the Pauli-X gate.

# CNOT Gate

## Simultion

A break down of the original CNOT gate, such that we can use the same parameters as Qiskit other than the DRAG amplitudes.
"""

original_cnot = QuantumCircuit(2)
original_cnot.cx(0,1)
original_cnot_transpile = transpile(original_cnot, realistic_backend)
original_cnot_sched = schedule(original_cnot_transpile, realistic_backend)
original_cnot_sched

"""Define the custom CNOT gate."""

def custom_CNOT_circuit(backend):
    numOfQubits = 2
    realistic_circ = QuantumCircuit(numOfQubits)

    amp1 = random.uniform(0.000001, 0.999999)
    amp2 = random.uniform(0.000001, 0.999999)
    amp3 = random.uniform(0.000001, 0.999999)
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

custom_CNOT_circuit(realistic_backend)

"""Transpile and schedule the custom CNOT gate."""

cnot_circ = custom_CNOT_circuit(realistic_backend)
CNOT_circ_transpile = transpile(cnot_circ[0], realistic_backend)
CNOT_circ_sched = schedule(CNOT_circ_transpile, realistic_backend)
CNOT_circ_sched.draw()

"""Run the custom Pauli-X gate and time it for analysing the how long it takes to run of different devices. Then show the count of states in a histogram."""

CNOT_job = realistic_backend.run(CNOT_circ_sched, shots=1024)
start_time = time.time()
CNOT_counts = CNOT_job.result().get_counts()
end_time = time.time()
CNOT_time = end_time - start_time
print(CNOT_time)

plot_histogram(CNOT_counts)

CNOT_counts

def CNOT_count(counts_dict):
    state00 = 0
    state01 = 0
    state10 = 0
    state11 = 0
    if '00' in counts_dict:
        state00 = counts_dict['00']
    if '01' in counts_dict:
        state01 = counts_dict['01']
    if '10' in counts_dict:
        state10 = counts_dict['10']
    if '11' in counts_dict:
        state11 = counts_dict['11']
    return [state00, state01, state10, state11]

"""Put counts of states in an array, then convert it to a percentage."""

realistic_CNOT_array = CNOT_count(CNOT_counts)
print(realistic_CNOT_array)

def percentage_contribution(arr):
    total_sum = sum(arr)
    percentages = [elem / total_sum for elem in arr]
    return percentages

real_percent_cnot = percentage_contribution(realistic_CNOT_array)
print(real_percent_cnot)

ideal_cnot_counts = [1024, 0, 0, 0]
ideal_percent_cnot = percentage_contribution(ideal_cnot_counts)
print(ideal_percent_cnot)

"""Calculate fidelity between the custom CNOT gate count and the ideal gate count."""

cnot_fidelity_value = fidelity(ideal_percent_cnot, real_percent_cnot)
print("CNOT fidelity:", cnot_fidelity_value)

"""The whole process in one function that can be used to gather data."""

def custom_cnot_process():
    customCNOT = custom_CNOT_circuit(realistic_backend)
    realistic_circ = customCNOT[0]
    realistic_data = customCNOT[1]
    cnot_circ_transpile = transpile(realistic_circ, realistic_backend)
    cnot_circ_sched = schedule(cnot_circ_transpile, realistic_backend)
    job = realistic_backend.run(cnot_circ_sched, shots=1024)
    start_time = time.time()
    counts = job.result().get_counts()
    end_time = time.time()
    cnot_time = end_time - start_time
    realistic_cnot_array = CNOT_count(counts)
    real_percent_cnot = percentage_contribution(realistic_cnot_array)
    ideal_cnot_counts = [1024, 0, 0, 0]
    ideal_percent_cnot = percentage_contribution(ideal_cnot_counts)
    cnot_fidelity_value = fidelity(ideal_percent_cnot, real_percent_cnot)
    process_data = [realistic_data, [cnot_time], realistic_cnot_array, real_percent_cnot, [cnot_fidelity_value]]
    return process_data

custom_cnot_process()

"""## Gather Data"""

num_runs = 1000000
folder = '/home/lunet/phzf/Documents'
filename = 'cnot_data.csv'
file_path = os.path.join(folder, filename)
with open(file_path, 'a', newline='') as f:
    write = csv.writer(f)
    for i in range(num_runs):
        array = custom_cnot_process()
        flattened_array = [str(i) for sublist in array for i in sublist]
        write.writerow(flattened_array)

"""## Regression with Deep Neural Network

### Preprocess Data
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

"""Check results"""

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

"""As you can see these are not the best results since for a CNOT gate we want it to be 100% in state 00. The reason this is not such a good result is (a) because there is not enough intitial data for a multi-qubit gate (judging by the loss curves above) and (b) because there is probably some sort of quantum control theory in play here that needs to be investigated further.

# Random Custom Circuits

Custom Hadamard gate with qubit channel defined for circuit.
"""

def custom_hadamard(qubit_channel, backend):
    phase = random.uniform(0.000001, 6.283) #Phase in radians
    amp = random.uniform(0.000001, 0.999999)
    with pulse.build(backend, name='Hadamard') as h_q0:
        pulse.shift_phase(phase, pulse.drive_channel(qubit_channel))
        pulse.play(Drag(duration=160, amp=amp, sigma=40, beta=-0.35835396095069005, angle=0.008783280252964184), pulse.drive_channel(qubit_channel))#amp=0.09619222815230141
        pulse.shift_phase(phase, pulse.drive_channel(qubit_channel))
    data = [phase, amp]
    return h_q0, data

custom_hadamard(0, realistic_backend)[0].draw()

"""Custom CNOT gate with qubit channels defined for circuit."""

def custom_cnot(control_channel, target_channel, backend):
    phase1 =random.uniform(0.000001, 6.283)
    amp1 = random.uniform(0.000001, 0.999999)
    amp2 = random.uniform(0.000001, 0.999999)
    amp3 = random.uniform(0.000001, 0.999999)
    amp4 = random.uniform(0.000001, 0.999999)
    amp5 = random.uniform(0.000001, 0.999999)
    amp6 = random.uniform(0.000001, 0.999999)
    amp7 = random.uniform(0.000001, 0.999999)
    with pulse.build(backend, name='CNOT') as cnot_gate:
        pulse.shift_phase(phase1, pulse.drive_channel(control_channel))#1.5707963267948966
        pulse.play(Drag(duration=160, sigma=40, beta=-0.25388969010654494, amp=amp1, angle=-1.5707963267948968), pulse.drive_channel(control_channel))#0.19290084722113582
        pulse.play(Drag(duration=160, sigma=40, beta=-0.5196057292826135, amp=amp2, angle=0.007898523847627245), pulse.drive_channel(target_channel))#0.0746414463895804
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=amp3, angle=0.023765720478682046), pulse.drive_channel(target_channel))#0.06740617502322209
        pulse.delay(duration=160, channel=ControlChannel(control_channel))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=amp4, angle=0.4451938954606395), ControlChannel(control_channel))#0.37716020465737493
        pulse.delay(duration=432, channel=pulse.drive_channel(control_channel))
        pulse.play(Drag(duration=160, sigma=40, beta=-0.25388969010654494, amp=amp5, angle=0.0), pulse.drive_channel(control_channel))#0.19290084722113582
        pulse.delay(duration=160, channel=pulse.drive_channel(target_channel))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=amp6, angle=-3.1178269331111115), pulse.drive_channel(target_channel))#0.06740617502322209
        pulse.delay(duration=160, channel=ControlChannel(control_channel))
        pulse.play(GaussianSquare(duration=432, sigma=64, width=176, amp=amp7, angle=-2.696398758129154), ControlChannel(control_channel))#0.3771602046573749
    data = [phase1, amp1, amp2, amp3, amp4, amp5, amp6, amp7]
    return cnot_gate, data

custom_cnot(0, 1, realistic_backend)[0].draw()

"""Random two-qubit circuit function, that will have Hadamard and CNOT gates randomly distributed along the two-qubit channels."""

def random_pulse_circuit():
    numOfQubits = 2
    realistic_circ = QuantumCircuit(numOfQubits)
    idealistic_circ = QuantumCircuit(numOfQubits)
    data = []
    for _ in range(random.randrange(1, 4)):
        qubit = random.randrange(numOfQubits)
        if random.random() < 0.5:  # Randomly choose between Hadamard and CNOT
            custom_h_gate = Gate('Hadamard', 1, [])
            realistic_circ.append(custom_h_gate, [qubit])
            rand_had_gate = custom_hadamard(qubit, realistic_backend)
            realistic_circ.add_calibration(custom_h_gate, [qubit], rand_had_gate[0])
            data.append(rand_had_gate[1])
            idealistic_circ.h(qubit)
        else:
            control_qubit = qubit
            target_qubit = random.choice([q for q in range(numOfQubits) if q != control_qubit])
            custom_c_gate = Gate('CNOT', 2, [])
            realistic_circ.append(custom_c_gate, [control_qubit, target_qubit])
            rand_cnot_gate = custom_cnot(control_qubit, target_qubit, realistic_backend)
            realistic_circ.add_calibration(custom_c_gate, [control_qubit, target_qubit], rand_cnot_gate[0])
            data.append(rand_cnot_gate[1])
            idealistic_circ.cx(control_qubit, target_qubit)

    with pulse.build(realistic_backend, name='Measure') as measure:
        pulse.acquire(120, pulse.acquire_channel(0), MemorySlot(0))
        pulse.acquire(120, pulse.acquire_channel(1), MemorySlot(1))

    measuring_gate = Gate('Measurement', 2, [])
    realistic_circ.append(measuring_gate, [0, 1])
    realistic_circ.add_calibration(measuring_gate, [0, 1], measure)
    idealistic_circ.measure_all()
    return realistic_circ, idealistic_circ, data, numOfQubits

"""Show the circuit with random custom Hadamard and CNOT gates."""

circ_test = random_pulse_circuit()
circ_test[0].draw()

"""Show the circuit with random generic Hadamard and CNOT gates."""

circ_test[1].draw()

"""Transpile and schedule the custom random circuit."""

random_circ_transpile = transpile(circ_test[0], realistic_backend)
random_circ_sched = schedule(random_circ_transpile, realistic_backend)
random_circ_sched.draw()

"""Run the custom random circuit and time it for analysing the how long it takes to run of different devices. Then show the count of states in a histogram."""

job = realistic_backend.run(random_circ_sched, shots=1024)
start_time = time.time()
counts = job.result().get_counts()
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

plot_histogram(counts)

print(counts)

def realistic_counts(counts_dict):
    state00 = 0
    state01 = 0
    state10 = 0
    state11 = 0
    if '00' in counts_dict:
        state00 = counts_dict['00']
    if '01' in counts_dict:
        state01 = counts_dict['01']
    if '10' in counts_dict:
        state10 = counts_dict['10']
    if '11' in counts_dict:
        state11 = counts_dict['11']
    return [state00, state01, state10, state11]

def idealistic_counts(counts_dict):
    state00 = 0
    state01 = 0
    state10 = 0
    state11 = 0
    if '00' in counts_dict:
        state00 = counts_dict['00']
    if '01' in counts_dict:
        state01 = counts_dict['01']
    if '10' in counts_dict:
        state10 = counts_dict['10']
    if '11' in counts_dict:
        state11 = counts_dict['11']
    # the returned array has state "10" and "01" switched cause of the
    # little-endia convention accociated with qiskit
    return [state00, state10, state01, state11]

"""Put counts of states in an array, then convert it to a percentage."""

realistic_result_array = realistic_counts(counts)
print(realistic_result_array)

realistic_percent_array = percentage_contribution(realistic_result_array)
print(realistic_percent_array)

"""Transpile and run the generic version of the random circuit on the ideal backend, in order to get the ideal state distribution for that circuit."""

simulated_circ = transpile(circ_test[1], idealistic_backend)

sim_job = idealistic_backend.run(simulated_circ)
ideal_counts = sim_job.result().get_counts(circ_test[1])
plot_histogram(ideal_counts)

print(ideal_counts)

ideal_result_array = idealistic_counts(ideal_counts)
print(ideal_result_array)

ideal_percent_array = percentage_contribution(ideal_result_array)
print(ideal_percent_array)

"""Calculate the fidelity between the random custom pulse circuit and the ideal version."""

def fidelity(ideal_probs, realistic_probs):
    psi_ideal = np.sqrt(ideal_probs)
    psi_realistic = np.sqrt(realistic_probs)
    fidelity_value = abs(np.dot(psi_ideal, psi_realistic))**2
    return fidelity_value

fidelity_value = fidelity(realistic_percent_array, ideal_percent_array)
print("Fidelity:", fidelity_value)

"""Create arrays of the gate sequences where the Hadamard and CNOT are encoded as 1 and 2, respectively. This allows us to represent the circuit."""

def encode_pulse_circuit(circuit):
    random_circ, idealistic_circ, data, num_qubits = circuit
    gate_encoding = {
        'Hadamard': 1,
        'CNOT': 2,
        'Measurement' : 3
    }
    gate_sequence = [[] for _ in range(num_qubits)]
    for gate, qargs, _ in random_circ.data:
        if gate.name in gate_encoding:
            gate_code = gate_encoding[gate.name]
            for qubit in qargs:
                qubit_index = qubit.index
                gate_sequence[qubit_index].append(gate_code)
    circ = random_circ.draw()
    return circ, gate_sequence

encode_pulse_circuit(random_pulse_circuit())

"""Code the adjacency matrix for this circuit in order to show the connection between different qubit channels through multi-qubit (CNOT) gates."""

def pulse_adjacency_matrix(circuit):
    num_qubits = circuit.num_qubits
    adjacency = np.zeros((num_qubits, num_qubits), dtype=int)
    for gate in circuit.data:
        if gate[0].name == 'CNOT':
            control_qubit = gate[1][0].index#find_bit
            target_qubit = gate[1][1].index#find_bit
            adjacency[control_qubit][target_qubit] = 1
            adjacency[target_qubit][control_qubit] = 1
    return adjacency

def show_pulse_circ_and_adj():
    random_circ = random_pulse_circuit()[0]
    circ = random_circ.draw()
    adj_matrix = pulse_adjacency_matrix(random_circ)
    return circ, adj_matrix

show_pulse_circ_and_adj()





from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Define your DNN architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,), name='ReLU'),
    Dense(128, activation='tanh', name='tanh_1'),
    Dense(256, activation='tanh', name='tanh_2'),
    Dense(128, activation='tanh', name='tanh_3'),
    Dense(64, activation='relu', name='ReLU_1'),
    Dense(1, activation='sigmoid', name='Sigmoid_Output')
])

# Plot the model architecture and save it to a file
plot_model(model, to_file='dnn_architecture.png', show_shapes=True, show_layer_names=True)

from IPython.display import Image

Image('dnn_architecture.png')

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.utils import plot_model

# Define input layer
input_layer = Input(shape=(1,), name='Input')

# Define the first group of hidden layers
hidden_layer_1 = Dense(128, activation='tanh', name='tanh_1')(input_layer)
hidden_layer_2 = Dense(256, activation='tanh', name='tanh_2')(hidden_layer_1)

# Define the second group of hidden layers
hidden_layer_3 = Dense(128, activation='tanh', name='tanh_3')(input_layer)
hidden_layer_4 = Dense(64, activation='relu', name='ReLU_1')(hidden_layer_3)

# Concatenate the outputs of the two groups of hidden layers
concatenated_output = Concatenate(name='Concatenate')([hidden_layer_2, hidden_layer_4])

# Define output layer
output_layer = Dense(1, activation='sigmoid', name='Output')(concatenated_output)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Plot the model architecture with horizontal layout
plot_model(model, to_file='dnn_architecture.png', show_shapes=True, show_layer_names=True, rankdir='LR')

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/hadamard_data.csv')
Amplitude = df.iloc[200:310, 0]
# filtered_Amplitude = Amplitude[:237] + Amplitude[237 + 1:]
Fidelity = df.iloc[200:310, 1]
# filtered_Fidelity = Amplitude[:237] + Amplitude[237 + 1:]
hadamard_dataset = pd.DataFrame({'Amplitude': Amplitude, 'Fidelity': Fidelity})
hadamard_dataset.head()

# @title Amplitude vs Fidelity

from matplotlib import pyplot as plt
hadamard_dataset.plot(kind='scatter', x='Amplitude', y='Fidelity', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/hadamard_data.csv')
Amplitude = df.iloc[200:310, 0]
# filtered_Amplitude = Amplitude[:237] + Amplitude[237 + 1:]
Fidelity = df.iloc[200:310, 1]
# filtered_Fidelity = Amplitude[:237] + Amplitude[237 + 1:]
hadamard_dataset = pd.DataFrame({'Amplitude': Amplitude, 'Fidelity': Fidelity})
hadamard_dataset.head()

df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/x_gate_data.csv')
x_Amplitude = df2.iloc[0:60, 0]
# filtered_Amplitude = Amplitude[:237] + Amplitude[237 + 1:]
x_Fidelity = df2.iloc[0:60, 1]
# filtered_Fidelity = Amplitude[:237] + Amplitude[237 + 1:]
x_dataset = pd.DataFrame({'Amplitude': x_Amplitude, 'Fidelity': x_Fidelity})
x_dataset.head()

# @title Amplitude vs Fidelity

from matplotlib import pyplot as plt
x_dataset.plot(kind='scatter', x='Amplitude', y='Fidelity', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

