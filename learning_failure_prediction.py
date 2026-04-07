import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
# ======================================
# LOAD HARD LEARNING RESULTS
# ======================================
gemma = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_gemma_results.csv")
llama = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_llama_results.csv")
qwen = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_qwen_results.csv")
mistral = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_mistral_results.csv")

# ======================================
# ADD MODEL LABEL
# ======================================
gemma["model"] = "gemma"
llama["model"] = "llama"
qwen["model"] = "qwen"
mistral["model"] = "mistral"

# ======================================
# MERGE
# ======================================
df = pd.concat([gemma, llama, qwen, mistral], ignore_index=True)

print("Combined shape:", df.shape)
print(df.head())

# ======================================
# SELECT FEATURES + TARGET
# ======================================
features = [
    "task_family",
    "noise_family",
    "target_position",
    "conflict_strength",
    "model"
]

X = df[features]
y = df["correct"]

# ======================================
# PREPROCESSING
# ======================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ]
)

# ======================================
# TRAIN / TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 1. LOGISTIC REGRESSION
# ======================================
log_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ======================================
# 2. RANDOM FOREST
# ======================================
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ======================================
# 3. SVM (RBF)
# ======================================
svm_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", SVC(kernel="rbf", C=1.0))
])

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\n=== SVM (RBF) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ======================================
# 4. LINEAR SVM
# ======================================
linear_svm_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearSVC(max_iter=5000))
])

linear_svm_model.fit(X_train, y_train)
y_pred_linear_svm = linear_svm_model.predict(X_test)

print("\n=== LINEAR SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear_svm))
print(classification_report(y_test, y_pred_linear_svm))

# ======================================
# FEATURE IMPORTANCE (RF)
# ======================================
ohe = rf_model.named_steps["preprocess"].named_transformers_["cat"]
feature_names = ohe.get_feature_names_out(features)

importances = rf_model.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== TOP 20 IMPORTANT FEATURES ===")
print(feat_imp.head(20))

ftop_features = feat_imp.head(10)

top_features = feat_imp.head(10)

plt.figure()
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top Predictors of LLM Failure")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()