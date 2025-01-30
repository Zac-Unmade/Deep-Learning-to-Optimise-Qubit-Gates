from utils import fidelity, percentage_difference, percentage_contribution, single_gate_counts, multi_gate_count

"""
A break down of the original CNOT gate, such that we can use the same parameters as Qiskit other than the DRAG amplitudes.
"""
original_cnot = QuantumCircuit(2)
original_cnot.cx(0,1)
original_cnot_transpile = transpile(original_cnot, realistic_backend)
original_cnot_sched = schedule(original_cnot_transpile, realistic_backend)
original_cnot_sched


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

"""Put counts of states in an array, then convert it to a percentage."""

realistic_CNOT_array = multi_gate_count(CNOT_counts)
print(realistic_CNOT_array)


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
    realistic_cnot_array = multi_gate_count(counts)
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
