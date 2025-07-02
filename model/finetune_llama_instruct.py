import os
import argparse
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils.other import prepare_model_for_int8_training
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="decapoda-research/llama-7b-hf")
    parser.add_argument("--data_path", type=str, default="instruct.json")
    parser.add_argument("--output_dir", type=str, default="./llama-instruct")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--val_set_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_on_inputs", type=bool, default=False)
    parser.add_argument("--add_eos_token", type=bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_in_8bit", type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=args.data_path)
    def format_instruction(example):
        prompt = example["instruction"]
        response = example["output"]
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        return tokenizer(text, truncation=True, max_length=args.cutoff_len, padding="max_length")

    tokenized_data = data["train"].map(format_instruction)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data.select(range(args.val_set_size)),
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    )

    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
