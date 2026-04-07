import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ======================================
# LOAD ATTENTION (ALL MODELS)
# ======================================
att_gemma = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_gemma_results.csv")
att_llama = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_llama_results.csv")
att_qwen = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_qwen_results.csv")
att_mistral = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_mistral_results.csv")

att_gemma["model"] = "gemma"
att_llama["model"] = "llama"
att_qwen["model"] = "qwen"
att_mistral["model"] = "mistral"

attention = pd.concat([att_gemma, att_llama, att_qwen, att_mistral], ignore_index=True)

# ======================================
# LOAD LEARNING (ALL MODELS)
# ======================================
learn_gemma = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_gemma_results.csv")
learn_llama = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_llama_results.csv")
learn_qwen = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_qwen_results.csv")
learn_mistral = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/learning_hard_mistral_results.csv")

learn_gemma["model"] = "gemma"
learn_llama["model"] = "llama"
learn_qwen["model"] = "qwen"
learn_mistral["model"] = "mistral"

learning = pd.concat([learn_gemma, learn_llama, learn_qwen, learn_mistral], ignore_index=True)

# ======================================
# ADD TASK TYPE
# ======================================
attention["task_type"] = "attention"
learning["task_type"] = "learning"

# ======================================
# COMBINE DATASETS
# ======================================
df = pd.concat([attention, learning], ignore_index=True)

print("Combined shape:", df.shape)
print("\nModel distribution:\n", df["model"].value_counts())

# ======================================
# SELECT FEATURES
# ======================================
features = [
    "task_type",
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
# MODEL
# ======================================
model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ======================================
# TRAIN / TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nUnified Model Accuracy:", accuracy)

# ======================================
# FEATURE IMPORTANCE
# ======================================
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
feature_names = ohe.get_feature_names_out(features)

importances = model.named_steps["rf"].feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Features:\n")
print(feat_imp.head(20))

# ======================================
# CLEAN PLOT (FINAL)
# ======================================
top = feat_imp.head(12)

plt.figure()
plt.barh(top["feature"], top["importance"])
plt.title("Unified Predictors of LLM Failure (Attention + Learning)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()