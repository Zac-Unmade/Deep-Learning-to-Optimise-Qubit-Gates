"""
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
