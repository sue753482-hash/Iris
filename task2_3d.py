"""
Task 2: 3D Decision Boundary
Features: sepal_length, sepal_width, petal_length
Classes: Setosa vs Versicolor
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

print("Task 2: 3D Decision Boundary")
print("============================")

# Load first two classes
iris = load_iris()
X = iris.data[:100, :3]
y = iris.target[:100]

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print("Classes: Setosa (0), Versicolor (1)")

# Train SVM
clf = SVC(kernel='linear')
clf.fit(X, y)

# Create 3D plot
fig = plt.figure(figsize=(12, 10))

# Main 3D view
ax = fig.add_subplot(111, projection='3d')

# Plot points
colors = ['red', 'blue']
for i in range(2):
    mask = (y == i)
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
              color=colors[i], label=f'Class {i}',
              s=50, alpha=0.8, edgecolor='black')

# Decision plane
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                     np.linspace(y_min, y_max, 10))

w = clf.coef_[0]
b = clf.intercept_[0]
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

ax.set_xlabel('sepal_length (cm)')
ax.set_ylabel('sepal_width (cm)')
ax.set_zlabel('petal_length (cm)')
ax.set_title('3D Decision Boundary')
ax.legend()

plt.tight_layout()
plt.savefig('task2_result.png', dpi=200, bbox_inches='tight')
print("Image saved: task2_result.png")
plt.show()

print(f"\nSVM accuracy: {clf.score(X, y):.3f}")
print("Task 2 completed.")