import ollama
import pandas as pd
import time

# ================================
# CONFIG
# ================================
INPUT_PATH = "/Users/chanduesukula/Downloads/Kaggle/learning_hard_benchmark_v1.csv"

MODELS = {
    "gemma2:9b": "learning_hard_gemma_results.csv",
    "qwen2.5:7b": "learning_hard_qwen_results.csv",
    "llama3:8b": "learning_hard_llama_results.csv",
    "mistral": "learning_hard_mistral_results.csv",
}

# ================================
# LOAD DATA
# ================================
df_base = pd.read_csv(INPUT_PATH)

# ================================
# STRICT EVALUATION
# ================================
def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(".", "").replace(",", "").replace(":", "").replace(";", "")
    return " ".join(text.split())

def is_correct(row) -> bool:
    gold = normalize_answer(row["gold_answer"])
    pred = normalize_answer(row["model_output_clean"])

    # exact match
    if pred == gold:
        return True

    # last token match fallback
    pred_tokens = pred.split()
    if pred_tokens and pred_tokens[-1] == gold:
        return True

    return False

# ================================
# MODEL CALL
# ================================
def run_model(prompt: str, model_name: str) -> str:
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        print("Error:", e)
        return ""

# ================================
# LOOP THROUGH MODELS
# ================================
for model_name, output_file in MODELS.items():
    print(f"\n=== Running {model_name} ===")

    df = df_base.copy()
    outputs = []

    for i, row in df.iterrows():
        print(f"{model_name} → {i+1}/{len(df)}")
        out = run_model(row["noisy_prompt"], model_name)
        outputs.append(out)
        time.sleep(0.2)

    df["model_output"] = outputs
    df["model_output_clean"] = df["model_output"].astype(str).apply(normalize_answer)
    df["correct"] = df.apply(is_correct, axis=1)

    output_path = f"/Users/chanduesukula/Downloads/Kaggle/{output_file}"
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Accuracy:", df["correct"].mean())

    print("\nBy task family:")
    print(df.groupby("task_family")["correct"].mean())

    print("\nBy noise:")
    print(df.groupby("noise_family")["correct"].mean())

    print("\nBy position:")
    print(df.groupby("target_position")["correct"].mean())