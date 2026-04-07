import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ================================
# LOAD DATA
# ================================
llama = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_llama_results.csv")
mistral = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_mistral_results.csv")
qwen = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_qwen_results.csv")
gemma = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_gemma_results.csv")

# ================================
# ADD MODEL LABEL
# ================================
llama["model"] = "llama"
mistral["model"] = "mistral"
qwen["model"] = "qwen"
gemma["model"] = "gemma"

# ================================
# MERGE
# ================================
df = pd.concat([llama, mistral, qwen, gemma], ignore_index=True)

print("Combined shape:", df.shape)

# ================================
# FEATURES
# ================================
features = [
    "attention_module",
    "noise_family",
    "target_position",
    "conflict_strength",
    "dominant_signal_type",
    "context_length_bucket",
    "distance_to_target",
    "model"
]

X = df[features]
y = df["correct"]

# ================================
# PREPROCESSING
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ]
)

# ================================
# SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 1. LOGISTIC REGRESSION
# ================================
log_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n=== LOGISTIC REGRESSION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ================================
# 2. RANDOM FOREST
# ================================
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ================================
# 3. SVM (RBF KERNEL)
# ================================
svm_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", SVC(kernel="rbf", C=1.0))
])

svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\n=== SVM (RBF) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ================================
# 4. LINEAR SVM
# ================================
linear_svm_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LinearSVC(max_iter=5000))
])

linear_svm_model.fit(X_train, y_train)
y_pred_linear_svm = linear_svm_model.predict(X_test)

print("\n=== LINEAR SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear_svm))
print(classification_report(y_test, y_pred_linear_svm))

# ================================
# FEATURE IMPORTANCE (RF ONLY)
# ================================
ohe = rf_model.named_steps["preprocess"].named_transformers_["cat"]
feature_names = ohe.get_feature_names_out(features)

importances = rf_model.named_steps["model"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== TOP 20 IMPORTANT FEATURES ===")
print(feat_imp.head(20))