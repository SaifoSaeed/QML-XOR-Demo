# QML-XOR-Demo

A technical demonstration of Quantum Machine Learning (QML) classifying the non-linearly separable XOR dataset using Qiskit and Support Vector Machines (SVM).

## Overview
Classical linear classifiers fundamentally fail to separate XOR logic in a 2D plane. This repository demonstrates how to bypass this limitation using **Quantum Kernel Methods**. By encoding classical data into quantum states via a `ZZFeatureMap`, the dataset is mapped into a higher-dimensional Hilbert space where a linear hyperplane can successfully separate the classes.

## Features
* **Head-to-Head Benchmarking:** Direct comparison between a Classical Linear SVM and a Quantum Kernel SVM.
* **Decision Boundary Visualization:** Generates a high-resolution 2D meshgrid contour plot to visualize the non-linear decision surface derived from the quantum kernel.
* **Circuit Inspection:** Decomposes and plots the quantum feature map circuit to illustrate the physical gate structure, including local feature encoding (Phase gates) and non-linear joint encoding (CNOT entanglement).
* **High-Performance Simulation:** Utilizes exact statevector linear algebra (`FidelityStatevectorKernel`) and `qiskit-aer` V2 primitives to bypass stochastic shot-noise and V1 ISA transpilation overhead.

## Installation
Ensure you have a working Python environment, then install the required dependencies:

```bash
pip install qiskit qiskit-machine-learning qiskit-aer scikit-learn matplotlib numpy
```

## Usage
Execute the main script to train the models, compute the exact analytic state overlaps, and generate the visualizations:

```bash
python test.py
```

## Results
* **Classical Linear SVM:** Fails to capture XOR logic, achieving ~75% accuracy (only after breaking symmetry to avoid degenerate optimization).
* **Quantum Kernel SVM:** Achieves 100% accuracy. The $Z \otimes Z$ interaction dynamically warps the feature space, allowing perfect classification.

## Visual Outputs

* `H2H.png`: The side-by-side decision boundary plot.
* `feature_map.png`: The decomposed quantum circuit diagram.
