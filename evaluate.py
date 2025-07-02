from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

# Load model and tokenizer
model_path = "./acaqas_lora_final"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

# Load evaluation data (instruction + ground truth format)
with open("data/eval.json") as f:
    data = json.load(f)

# Prepare for metric calculation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

gold_answers = []
predicted_answers = []

for item in tqdm(data):
    prompt = item["instruction"]
    gt_answer = item["response"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    pred_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    gold_answers.append(gt_answer.strip().lower())
    predicted_answers.append(pred_answer.strip().lower())

# Simple string match evaluation
def binary_match(gold, pred):
    return int(gold.strip() == pred.strip())

labels = [binary_match(g, p) for g, p in zip(gold_answers, predicted_answers)]

# Binary classification metrics (match vs mismatch)
accuracy = sum(labels) / len(labels)
precision = precision_score(labels, labels, average='binary')
recall = recall_score(labels, labels, average='binary')
f1 = f1_score(labels, labels, average='binary')

print("Evaluation Results:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
