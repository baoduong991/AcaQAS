# **AcaQAS: An Academic Question Answering System Based on Finetuning Large Language Models**
AcaQAS is an open academic QA system based on LoRA fine-tuning of open-source LLMs (Bloomz and LLaMA-7B). This repository includes the dataset, training scripts, evaluation benchmarks, and results for adapting LLMs to domain-specific academic question answering under resource constraints.

**Base Models**  
- [Bloomz-mt-7B](https://huggingface.co/bigscience/bloomz)
- [LLaMA-7B (HF-compatible)](https://huggingface.co/decapoda-research/llama-7b-hf)


## Proposed architectures

### The Transformer architecture
![Biểu đồ kết quả](architecture/transformer_architecture.png)

### Bloom architecture
![Biểu đồ kết quả](architecture/bloom_architecture.png)

## Datasets:
Pile dataset 
We used a subset of [The Pile dataset](https://pile.eleuther.ai/) containing academic sources (arXiv, PubMed, PhilPapers, etc.).
Website to the [original dataset](https://github.com/EleutherAI/the-pile)

### 🔧 Training Configuration
| Parameter                    | Value                          |
|-----------------------------|---------------------------------|
| **Model**                   | Bloomz-mt-7B / LLaMA-7B         |
| **GPU Environment**         | Google Colab Pro (Tesla T4, 16GB VRAM) |
| **LoRA Rank**               | 8                               |
| **LoRA Alpha**              | 16                              |
| **LoRA Dropout**            | 0.05                            |
| **Number of Epochs**        | 3                               |
| **Batch Size**              | 4 (gradient accumulation = 8)   |
| **Learning Rate**           | 2e-4                            |
| **Optimizer**               | AdamW                           |
| **Warmup Ratio**            | 0.03                            |
| **Sequence Length**         | 512 tokens                      |
| **Precision**               | 8-bit (`bitsandbytes`)          |
| **Frameworks**              | PyTorch, Hugging Face Transformers |
## Result

| Prompt |	Response |
| ------------- | -------------|
|Solve for x: 2x+5=15.	| x = 5. |
| In calculus, what is the derivative of the function f(x) = e2x? |	The derivative of f ′(x) = 2e2x? |
| What is the derivative of f(x)=x3+2x with respect to x?	| The derivative is f ′(x)=3x2+2
| Define the term "Big O notation" in algorithm analysis.	| Big O notation describes the upper bound of an algorithm's time or space complexity, representing its worst-case growth rate relative to input size.
| Define the concept of a prime number.	 | A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

### Metric-Based Evaluation Results of Fine-Tuned Models.
| Model	Accuracy (%)	| Precision (%)	| Recall (%)	| F1 Score (%)
| ------------- | ------------- |------------- |------------- |
| LLaMA-7B (Pre-Fine-Tuning) |	68.5 |	70.2 |	66.8 |	68.4 |
| LLaMA-7B (Post-Fine-Tuning) |	78.9 |	80.7 |	77.5 |	79.1 |
| Bloomz (Pre-Fine-Tuning)	| 69.3	| 71.0 |	67.2 |	69.0 |
| Bloomz (Post-Fine-Tuning) |	81.2 |	83.5 |	79.8 |	81.6 |

### LLM-Based Evaluation on MT-Bench Reasoning Set
| Model	Average Relevance Score |	Average Helpfulness Score |	Average Accuracy Score |	Average Level of Detail Score |	Overall Average Score (1-10) |
| ------------- | ------------- |------------- |------------- |------------- |
| Bloomz (Pre-Fine-Tuning) |	6.3 |	6.0 |	5.8 |	6.1 |	6.05 |
| Bloomz (Post-Fine-Tuning) |	8.0	| 8.2 |	8.1 |	8.0 |	8.08 |
| LLaMA-7B (Pre-Fine-Tuning) |	6.1 |	5.9 |	5.7 |	5.8	| 5.88 |
| LLaMA-7B (Post-Fine-Tuning) |	7.7 |	7.8 |	7.5 |	7.6 |	7.65 |

## 📊 Comparative Performance on Academic QA Benchmark

| Model           | Params | Method                          | Accuracy (%) | F1 Score (%) |
|----------------|--------|----------------------------------|--------------|--------------|
| Alpaca-7B       | 7B     | Instruction Tuning (Stanford-52K) | 67.3         | 66.9         |
| AcaQAS (Ours)   | 7B     | LoRA + Academic Alignment        | 78.9         | 79.1         |

## ❗ Sample Failure Cases in Academic QA

| Prompt                                                            | Generated Response                                                            | Issue Type           | Notes                                                                                       |
|------------------------------------------------------------------|--------------------------------------------------------------------------------|----------------------|---------------------------------------------------------------------------------------------|
| Compare supervised and unsupervised learning in machine learning. | “Supervised learning is better than unsupervised learning because it uses more data.” | Oversimplification    | Lacks contrastive analysis and misses key concepts like labeled vs. unlabeled data.         |
| What does the AUC metric evaluate?                                | “AUC measures the recall of a classification model.”                           | Factual Error         | Incorrect definition; AUC measures the trade-off between TPR and FPR, not recall.           |
| Define the concept of epistemology in philosophy.                | “Epistemology is the study of how science works in society.”                   | Hallucination         | Confuses epistemology with sociology of science; not grounded in standard definitions.      |
| Explain the P=NP problem in theoretical computer science.         | “P=NP means that all math problems can be solved easily by a computer.”        | Inaccuracy            | Misleading simplification of a core complexity theory question.                             |
| Describe the function of mitochondria in eukaryotic cells.       | “Mitochondria help plants photosynthesize energy.”                             | Domain Confusion      | Confuses plant chloroplasts with mitochondria.                                               |
| What is the derivative of f(x) = sin(x)?                         | “The derivative of sin(x) is -sin(x).”                                         | Factual Error         | Incorrect derivative; should be cos(x).                                                     |

### To finetune the model using LoRA and 8-bit quantization, run:
### Fine-tuning Command (Bloomz-7B1-mt on Academic QA Dataset)

Run the following command to fine-tune `bigscience/bloomz-7b1-mt` on `val.json` using LoRA and 8-bit quantization:
<pre><code>
python finetune_bloomz_instruct.py \
    --base_model 'bigscience/bloomz-7b1-mt' \
    --data_path 'val.json' \
    --output_dir './bloomz-acaqas-val' \
    --batch_size 2 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --cutoff_len 512 \
    --val_set_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --train_on_inputs False \
    --add_eos_token True \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 25 \
    --save_total_limit 2 \
    --load_in_8bit True
</code></pre>
### Fine-tuning Command (LLaMA-7B on Academic QA Dataset)
Run the following command to fine-tune `decapoda-research/llama-7b-hf` using LoRA with 8-bit quantization:
<pre><code>
python finetune_llama_instruct.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'val.json' \
    --output_dir './llama-acaqas-val' \
    --batch_size 2 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --cutoff_len 512 \
    --val_set_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --train_on_inputs False \
    --add_eos_token True \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 25 \
    --save_total_limit 2 \
    --load_in_8bit True
</code></pre>
