import pandas as pd
import matplotlib.pyplot as plt

# ======================================
# LOAD MODEL RESULTS
# ======================================
llama_df = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_llama_results.csv")
mistral_df = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_mistral_results.csv")
qwen_df = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_qwen_results.csv")
gemma_df = pd.read_csv("/Users/chanduesukula/Downloads/Kaggle/attention_benchmark_v8_gemma_results.csv")

# ======================================
# ADD MODEL NAME
# ======================================
llama_df["model"] = "Llama 3 8B"
mistral_df["model"] = "Mistral"
qwen_df["model"] = "Qwen 2.5 7B"
gemma_df["model"] = "Gemma 2 9B"

# ======================================
# KEEP REQUIRED COLUMNS
# ======================================
keep_cols = [
    "question_id",
    "attention_module",
    "noise_family",
    "target_position",
    "correct",
    "model"
]

llama_df = llama_df[keep_cols]
mistral_df = mistral_df[keep_cols]
qwen_df = qwen_df[keep_cols]
gemma_df = gemma_df[keep_cols]

# ======================================
# COMBINE DATA
# ======================================
df = pd.concat([llama_df, mistral_df, qwen_df, gemma_df], ignore_index=True)

print("Loaded:", df.shape)

# ======================================
# 1. POSITION ANALYSIS
# ======================================
position_order = ["none", "beginning", "middle", "end"]

acc_position = df.groupby("target_position")["correct"].mean().reindex(position_order)

print("\nAccuracy by Position:\n", acc_position)

acc_position.plot(kind="bar", figsize=(8, 5))
plt.title("LLM Accuracy vs Target Position")
plt.ylabel("Accuracy")
plt.xlabel("Target Position")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

# ======================================
# 2. POSITION (BY MODEL)
# ======================================
acc_position_model = df.groupby(["target_position", "model"])["correct"].mean().unstack().reindex(position_order)

print("\nAccuracy by Position (Model-wise):\n", acc_position_model)

acc_position_model.plot(kind="bar", figsize=(10, 6))
plt.title("Accuracy vs Target Position (by Model)")
plt.ylabel("Accuracy")
plt.xlabel("Target Position")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.show()

# ======================================
# 3. NOISE ANALYSIS
# ======================================
noise_order = [
    "none",
    "random_irrelevant",
    "emotional_salient",
    "misleading_context",
    "instructional_interference",
    "stale_context"
]

acc_noise = df.groupby("noise_family")["correct"].mean().reindex(noise_order)

print("\nAccuracy by Noise Type:\n", acc_noise)

acc_noise.plot(kind="bar", figsize=(10, 5))
plt.title("LLM Accuracy vs Noise Type")
plt.ylabel("Accuracy")
plt.xlabel("Noise Type")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.show()

# ======================================
# 4. NOISE (BY MODEL)
# ======================================
acc_noise_model = df.groupby(["noise_family", "model"])["correct"].mean().unstack().reindex(noise_order)

print("\nNoise vs Model:\n", acc_noise_model)

acc_noise_model.plot(kind="bar", figsize=(12, 6))
plt.title("Accuracy vs Noise Type (by Model)")
plt.ylabel("Accuracy")
plt.xlabel("Noise Type")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(title="Model")
plt.show()

# ======================================
# 5. MODEL COMPARISON
# ======================================
model_order = ["Gemma 2 9B", "Qwen 2.5 7B", "Llama 3 8B", "Mistral"]

acc_model = df.groupby("model")["correct"].mean().reindex(model_order)

print("\nOverall Model Accuracy:\n", acc_model)

acc_model.plot(kind="bar", figsize=(8, 5))
plt.title("Overall Accuracy by Model")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()