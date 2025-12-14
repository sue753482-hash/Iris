"""
Task 1: 2D Classification Visualization
Features: Petal Length, Petal Width
Classes: Setosa, Versicolor, Virginica
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

print("Task 1: 2D Classification Visualization")
print("======================================")

# Load data
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
print("Features: petal_length, petal_width")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Create grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Decision boundary
ax = axes[0, 0]
ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab20c')
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20c', 
                     edgecolor='black', s=50)
ax.set_xlabel('petal_length (cm)')
ax.set_ylabel('petal_width (cm)')
ax.set_title('Decision Boundaries')
plt.colorbar(scatter, ax=ax, ticks=[0, 1, 2])

# 2-4. Probability maps
class_names = ['Setosa', 'Versicolor', 'Virginica']
for i in range(3):
    row, col = divmod(i + 1, 2)
    ax = axes[row, col]
    
    prob_grid = probs[:, i].reshape(xx.shape)
    contour = ax.contourf(xx, yy, prob_grid, alpha=0.8, 
                         cmap='YlOrRd', levels=20)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20c', 
              edgecolor='black', s=30, alpha=0.5)
    
    ax.set_xlabel('petal_length (cm)')
    ax.set_ylabel('petal_width (cm)')
    ax.set_title(f'P({class_names[i]})')
    plt.colorbar(contour, ax=ax)

plt.tight_layout()
plt.savefig('task1_result.png', dpi=200, bbox_inches='tight')
print("Image saved: task1_result.png")
plt.show()

# Results
accuracy = model.score(X, y)
print(f"\nModel accuracy: {accuracy:.3f}")
print("Task 1 completed.")