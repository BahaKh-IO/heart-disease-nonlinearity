import numpy as np
import matplotlib.pyplot as plt
from svm import SVM

def plot_decision_boundary(model, X_2d, y, feature_names, title, save_path):
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")

    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                color="blue", label="No disease", edgecolors="k")
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                color="red", label="Disease", edgecolors="k")

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

    train_acc = np.mean(model.predict(X_2d) == y)
    print(f"\n[{title}] 2D accuracy: {train_acc:.3f}")
def plot_3d_boundary(model, X, y, feature_indices, feature_names, title, save_path):
    """
    Plots data points in 3D space with the SVM decision plane.
    Works for any model with a .predict() method.
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Extract 3 features
    X_3d = X[:, feature_indices]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual data points
    ax.scatter(
        X_3d[y == 0, 0], X_3d[y == 0, 1], X_3d[y == 0, 2],
        color="blue", label="No disease", edgecolors="k", s=50
    )
    ax.scatter(
        X_3d[y == 1, 0], X_3d[y == 1, 1], X_3d[y == 1, 2],
        color="red", label="Disease", edgecolors="k", s=50
    )

    # Create a grid on X-Y plane to draw the decision surface
    x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
    y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 30),
        np.linspace(y_min, y_max, 30)
    )

    # For each (x,y) grid point, find the z value where
    # the model's boundary lies — we'll use the mean of
    # the third feature as a fixed slice
    z_mean = X_3d[:, 2].mean()
    zz = np.full(xx.shape, z_mean)

    # Stack into (N, 3) array and get predictions
    grid_3d = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # We need a 3-feature model for this
    svm_3d = SVM(learning_rate=0.001, n_iterations=1000, C=1.0)
    svm_3d.fit(X_3d, y)

    predictions = svm_3d.predict(grid_3d).reshape(xx.shape)

    # Plot the decision surface as a transparent plane
    ax.plot_surface(
        xx, yy, zz,
        facecolors=plt.cm.coolwarm(predictions.astype(float)),
        alpha=0.3
    )

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    ax.set_title(title)
    ax.legend()

    plt.savefig(save_path)
    plt.show()

    print(f"\n[{title}] 3D plot saved.")