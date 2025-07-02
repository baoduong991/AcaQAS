import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Load base model and tokenizer
model_name = "bigscience/bloomz-7b1-mt"  # or "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adapt to model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load custom academic dataset (in Hugging Face format)
dataset = load_dataset("json", data_files={"train": "data/train.json", "validation": "data/val.json"})
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./acaqas_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=False,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    report_to="none",
    save_total_limit=1,
)

# Trainer (can also use SFTTrainer for instruction-style tuning)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

# Train and save
trainer.train()
trainer.save_model("./acaqas_lora_final")
tokenizer.save_pretrained("./acaqas_lora_final")
