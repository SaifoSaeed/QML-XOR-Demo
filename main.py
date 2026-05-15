import warnings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
warnings.filterwarnings("ignore", category=DeprecationWarning)

X = np.array([[0, 0], [0, 1], [1, 0], [1.01, 1.01]])    #Avoid direct symmetry to avoid plotting errors.
y = np.array([0, 1, 1, 0])

#Plot Head-to-Head comparison of classifiers.
def plot_decision_boundary(clf, ax, title):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu, s=150)
    ax.set_title(title)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

#Classical linear SVM implementation.
clf_linear = SVC(C=10, kernel="linear")
clf_linear.fit(X, y)
y_pred_linear = clf_linear.predict(X)
print(f"Classical Linear SVM Accuracy: {accuracy_score(y, y_pred_linear):.2f}")

#Quantum Kernel Classification.
feature_map = ZZFeatureMap(feature_dimension=2, reps=1, entanglement='linear')
qkernel = FidelityStatevectorKernel(feature_map=feature_map)
clf_quantum = SVC(kernel=qkernel.evaluate)
clf_quantum.fit(X, y)
y_pred_quantum = clf_quantum.predict(X)
print(f"Quantum Kernel SVM Accuracy: {accuracy_score(y, y_pred_quantum):.2f}")


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

plot_decision_boundary(clf_linear, axs[0], "Classical Linear SVM")
plot_decision_boundary(clf_quantum, axs[1], "Quantum Kernel SVM")

plt.tight_layout()
plt.savefig("H2H.png")
plt.show()

#Inspecting the circuit...
circuit_fig = feature_map.decompose().draw(output='mpl')
circuit_fig.savefig("feature_map.png")
# circuit_fig.show()