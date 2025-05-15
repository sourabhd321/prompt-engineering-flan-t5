import os
import torch
import pandas as pd
from datasets import load_dataset
from config import MODEL_NAME, DATASET_NAME, DATASET_CONFIG, SPLIT, NUM_EXAMPLES
from prompts import generate_prompts
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

# GPU check
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not detected by PyTorch")

# Device setup
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda" if device == 0 else "cpu")
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# ROUGE scorer setup
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def get_response(prompt):
    try:
        output = summarizer(prompt, max_new_tokens=100, do_sample=False)
        return output[0]['generated_text'].strip()
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def compute_rouge(reference, generated):
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

def truncate_text(text, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def main():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT).shuffle(seed=42)

    # Few-shot examples
    seed_examples = dataset.select(range(2))
    seed_summaries = []
    for ex in seed_examples:
        article_text = truncate_text(ex["article"])
        summary = get_response(f"Summarize the following article in one line: {article_text}")
        seed_summaries.append({"text": article_text, "summary": summary})

    test_examples = dataset.select(range(2, 2 + NUM_EXAMPLES))
    results = []

    for example in test_examples:
        article_text = truncate_text(example["article"])
        reference_summary = example["highlights"]
        prompts = generate_prompts(article_text, few_shot_examples=seed_summaries)

        row = {
            "original_text": article_text,
            "reference_summary": reference_summary
        }

        for prompt_type, prompt in prompts.items():
            generated = get_response(prompt)
            row[prompt_type] = generated

            rouge = compute_rouge(reference_summary, generated)
            row[f"{prompt_type}_rouge1"] = rouge["rouge1"]
            row[f"{prompt_type}_rouge2"] = rouge["rouge2"]
            row[f"{prompt_type}_rougeL"] = rouge["rougeL"]

        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/outputs.csv", index=False)
    print("Results saved to results/outputs.csv")

if __name__ == "__main__":
    main()
