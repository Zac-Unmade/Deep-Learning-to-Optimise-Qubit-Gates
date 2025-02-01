# Deep-Learning-to-Optimise-Qubit-Gates
This repository contains the code for a novel approach to optimising pulse waveform parameters for high fidelity qubit gates, as presented in our paper. The method leverages Deep Neural Networks (DNNs) to model the relationship between pulse amplitudes and gate fidelities, improving the implementation of quantum gates such as Hadamard, Pauli-X, and CNOT.

The approach is built around training DNNs on a dataset of amplitudes and corresponding fidelities, generated through quantum simulations in Qiskit. A two-stage process is used to fine-tune the amplitudes, achieving high fidelities for single-qubit gates (0.999976 for Hadamard, 0.999923 for Pauli-X) and reasonable fidelity for the two-qubit CNOT gate (0.695313). This repository provides the code for generating the dataset, training the DNNs, and using them for pulse scheduling optimisation.

Key features include:
- Simulation of quantum gate fidelities using Qiskit.
- Deep Neural Network-based optimisation for high-fidelity qubit gate implementations.
- Two-stage optimisation to maximize gate fidelities.
- A focus on single-qubit gates (Hadamard, Pauli-X) and two-qubit gates (CNOT).

This work is part of the ongoing effort to improve the scalability and precision of quantum computing systems. 

## Paper

This code is associated with the paper titled *Optimisation of Pulse Waveforms for Qubit Gates using Deep Learning* (available on [arXiv:2408.02376](https://arxiv.org/abs/2408.02376)).

## Citation

If you use this code in your work, please cite our paper:

**Fillingham, Zachary & Nevisi, Hossein & Dora, Shirin**, *Optimisation of Pulse Waveforms for Qubit Gates using Deep Learning*, arXiv:2408.02376, 2024. [Available here](https://arxiv.org/abs/2408.02376).
