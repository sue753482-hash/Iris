"""
Task 4: Creative Combination
Combining 3D boundary with probability analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

print("Task 4: Creative Combination")
print("============================")

# Load data
iris = load_iris()
X = iris.data[:100, :3]
y = iris.target[:100]

print(f"Data: {X.shape[0]} samples, 3 features")
print("Combining SVM boundary with logistic regression probability")

# Train both models
svm_clf = SVC(kernel='linear')
svm_clf.fit(X, y)

lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X, y)

# Combined visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. SVM decision boundary
ax = axes[0, 0]
for i in range(2):
    mask = (y == i)
    ax.scatter(X[mask, 0], X[mask, 1],
              color=['red', 'blue'][i],
              label=['Setosa', 'Versicolor'][i],
              s=40, alpha=0.7)

# Decision line
w = svm_clf.coef_[0]
b = svm_clf.intercept_[0]
x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_line = (-w[0] * x_line - b) / w[1]
ax.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_title('SVM: 2D Decision Boundary')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Probability heatmap
ax = axes[0, 1]
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
xx, yy = np.meshgrid(x_range, y_range)

grid_2d = np.c_[xx.ravel(), yy.ravel(),
                np.full(xx.ravel().shape, X[:, 2].mean())]
prob_grid = lr_clf.predict_proba(grid_2d)[:, 1].reshape(xx.shape)

im = ax.imshow(prob_grid, extent=[x_range.min(), x_range.max(),
                                 y_range.min(), y_range.max()],
              origin='lower', cmap='coolwarm', alpha=0.8)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20',
          edgecolor='black', s=30)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_title('Probability Heatmap')
plt.colorbar(im, ax=ax, label='P(Versicolor)')

# 3. 3D combination
ax = axes[0, 2]
ax = fig.add_subplot(2, 3, 3, projection='3d')

# Points colored by class
for i in range(2):
    mask = (y == i)
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
              color=['red', 'blue'][i],
              s=40, alpha=0.7)

# Decision plane
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                     np.linspace(y_min, y_max, 10))
zz = (-w[0] * xx - w[1] * yy - b) / w[2]

ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')
ax.set_title('3D Decision Plane')

# 4. Model comparison
ax = axes[1, 0]
svm_acc = svm_clf.score(X, y)
lr_acc = lr_clf.score(X, y)

models = ['SVM', 'Logistic\nRegression']
accuracies = [svm_acc, lr_acc]
colors = ['orange', 'purple']

bars = ax.bar(models, accuracies, color=colors)
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
ax.set_ylim(0, 1.1)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{acc:.3f}', ha='center', va='bottom')

# 5. Feature importance
ax = axes[1, 1]
features = ['sepal_length', 'sepal_width', 'petal_length']
importance = np.abs(lr_clf.coef_[0])
importance = importance / importance.sum()

bars = ax.bar(features, importance, color='skyblue')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance')
ax.set_ylim(0, 1)

for bar, imp in zip(bars, importance):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{imp:.2f}', ha='center', va='bottom')

# 6. Confusion matrix (simplified)
ax = axes[1, 2]
y_pred_svm = svm_clf.predict(X)
y_pred_lr = lr_clf.predict(X)

# Calculate confusion for SVM
tp = np.sum((y_pred_svm == 1) & (y == 1))
fp = np.sum((y_pred_svm == 1) & (y == 0))
tn = np.sum((y_pred_svm == 0) & (y == 0))
fn = np.sum((y_pred_svm == 0) & (y == 1))

conf_matrix = np.array([[tp, fp], [fn, tn]])

im = ax.imshow(conf_matrix, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred 1', 'Pred 0'])
ax.set_yticklabels(['True 1', 'True 0'])
ax.set_title('Confusion Matrix (SVM)')

# Add text
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, conf_matrix[i, j],
                      ha="center", va="center",
                      color="white" if conf_matrix[i, j] > tp/2 else "black")

plt.suptitle('Task 4: Combined Analysis of Classification Methods', fontsize=14)
plt.tight_layout()
plt.savefig('task4_result.png', dpi=200, bbox_inches='tight')
print("Image saved: task4_result.png")
plt.show()

print(f"\nSVM Accuracy: {svm_acc:.3f}")
print(f"Logistic Regression Accuracy: {lr_acc:.3f}")
print("Task 4 completed.")