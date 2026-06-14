import numpy as np
from data_loader import load_data, prepare_data
from logistic_regression import LogisticRegression
from visualize import plot_decision_boundary
from visualize import plot_3d_boundary
from knn import KNN
from svm import SVM
import xgboost as xgb
import matplotlib.pyplot as plt

df = load_data()
X_train, X_test, y_train, y_test = prepare_data(df)

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

train_acc = accuracy(y_train, train_predictions)
test_acc = accuracy(y_test, test_predictions)

print("\n--- Logistic Regression Results ---")
print(f"Training accuracy: {train_acc:.3f}")
print(f"Testing accuracy:  {test_acc:.3f}")

print("\nCost over training (every 100 iterations):")
print(model.cost_history)

print("\nFinal weights:")
print(model.weights)
print("Final bias:", model.bias)

feature_names_all = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                      "restecg", "thalach", "exang", "oldpeak",
                      "slope", "ca", "thal"]

X_2d_train = X_train[:, [0, 7]]  # age, thalach

lr_2d = LogisticRegression(learning_rate=0.1, n_iterations=1000)
lr_2d.fit(X_2d_train, y_train)

plot_decision_boundary(
    lr_2d, X_2d_train, y_train,
    feature_names=["age (scaled)", "thalach (scaled)"],
    title="Logistic Regression: Age vs Max Heart Rate",
    save_path="results/lr_age_thalach.png"
)

knn_model = KNN(k=10)
knn_model.fit(X_train, y_train)

knn_train_predictions = knn_model.predict(X_train)
knn_test_predictions = knn_model.predict(X_test)

knn_train_acc = accuracy(y_train, knn_train_predictions)
knn_test_acc = accuracy(y_test, knn_test_predictions)

print("\n--- k-NN Results (k=10) ---")
print(f"Training accuracy: {knn_train_acc:.3f}")
print(f"Testing accuracy:  {knn_test_acc:.3f}")

knn_2d = KNN(k=10)
knn_2d.fit(X_2d_train, y_train)  # reuses X_2d_train from above

plot_decision_boundary(
    knn_2d, X_2d_train, y_train,
    feature_names=["age (scaled)", "thalach (scaled)"],
    title="k-NN (k=10): Age vs Max Heart Rate",
    save_path="results/knn_age_thalach.png"
)

svm_model = SVM(learning_rate=0.001, n_iterations=1000, C=1.0)
svm_model.fit(X_train, y_train)

svm_train_predictions = svm_model.predict(X_train)
svm_test_predictions = svm_model.predict(X_test)

svm_train_acc = accuracy(y_train, svm_train_predictions)
svm_test_acc = accuracy(y_test, svm_test_predictions)

print("\n--- SVM Results ---")
print(f"Training accuracy: {svm_train_acc:.3f}")
print(f"Testing accuracy:  {svm_test_acc:.3f}")

X_2d_train = X_train[:, [0, 7]]

svm_2d = SVM(learning_rate=0.001, n_iterations=1000, C=1.0)
svm_2d.fit(X_2d_train, y_train)

plot_decision_boundary(
    svm_2d, X_2d_train, y_train,
    feature_names=["age (scaled)", "thalach (scaled)"],
    title="SVM (C=1.0): Age vs Max Heart Rate",
    save_path="results/svm_age_thalach.png"
)

plot_3d_boundary(
    svm_model, X_train, y_train,
    feature_indices=[0, 7, 4],
    feature_names=["age", "thalach", "chol"],
    title="SVM Decision Boundary in 3D",
    save_path="results/svm_3d.png"
)

xgb_model = xgb.XGBClassifier(
    n_estimators=91,       # optimal number found by early stopping
    learning_rate=0.05,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.05,
    min_child_weight=3,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)



xgb_train_predictions = xgb_model.predict(X_train)
xgb_test_predictions  = xgb_model.predict(X_test)

xgb_train_acc = accuracy(y_train, xgb_train_predictions)
xgb_test_acc  = accuracy(y_test,  xgb_test_predictions)

print("\n--- XGBoost Results ---")
print(f"Training accuracy: {xgb_train_acc:.3f}")
print(f"Testing accuracy:  {xgb_test_acc:.3f}")


# ── Plot 1: Feature Importance ─────────────────────────
feature_names = ["age","sex","cp","trestbps","chol","fbs",
                 "restecg","thalach","exang","oldpeak",
                 "slope","ca","thal"]

importances = xgb_model.feature_importances_
indices     = np.argsort(importances)          # sort low → high
sorted_features    = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.barh(
    sorted_features,
    sorted_importances,
    color="mediumseagreen",
    edgecolor="black"
)

# Add value label on each bar
for bar, val in zip(bars, sorted_importances):
    ax.text(
        val + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center", fontsize=9
    )

ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title(
    "XGBoost — Feature Importances\n"
    "(how much each feature contributed to predictions)",
    fontsize=13, fontweight="bold"
)
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/xgb_feature_importance.png", dpi=150)
plt.show()
print("\nFeature importance chart saved.")


# ── Plot 2: Learning Curve (loss over rounds) ──────────
results = xgb_model.evals_result()
train_loss = results["validation_0"]["logloss"]
test_loss  = results["validation_1"]["logloss"]
rounds     = range(1, len(train_loss) + 1)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(rounds, train_loss,
        label="Train log-loss", color="steelblue",
        linewidth=2)
ax.plot(rounds, test_loss,
        label="Test log-loss",  color="coral",
        linewidth=2, linestyle="--")

ax.set_xlabel("Boosting Round (number of trees)", fontsize=12)
ax.set_ylabel("Log Loss", fontsize=12)
ax.set_title(
    "XGBoost — Learning Curve\n"
    "(loss decreasing with each boosting round)",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11)
ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("results/xgb_learning_curve.png", dpi=150)
plt.show()
print("Learning curve saved.")


# ── Plot 3: Decision Boundary ──────────────────────────
class XGBWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

xgb_2d = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.1,
    min_child_weight=5,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb_2d.fit(X_2d_train, y_train)
xgb_wrapper = XGBWrapper(xgb_2d)

plot_decision_boundary(
    xgb_wrapper, X_2d_train, y_train,
    feature_names=["age (scaled)", "thalach (scaled)"],
    title="XGBoost: Age vs Max Heart Rate",
    save_path="results/xgboost_age_thalach.png"
)
