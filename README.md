
# Prompt Engineering with Flan-T5 for News Summarization

This project evaluates different prompt engineering strategies using the Flan-T5 language model to summarize news articles from the CNN/DailyMail dataset. It measures output quality using ROUGE metrics and identifies which prompt types work best.


## Installation
git clone https://github.com/sourabhd321/prompt-engineering-flan-t5.git

cd prompt-engineering-flan-t5

pip install -r requirements.txt

## Usage
python runner.py

### What Happens When You Run `runner.py`:

1. Loads the Flan-T5 model
2. Generates summaries using different prompt styles
3. Evaluates outputs using ROUGE metrics
4. Saves results to `results/outputs.csv`

## Prompting Techniques
| Prompt Type         | Description                        |
| ------------------- | ---------------------------------- |
| `zero_shot`         | No examples, direct input          |
| `one_shot`          | One example before the prompt      |
| `few_shot`          | Two examples shown                 |
| `instruction_tuned` | Natural instruction form           |
| `role_based`        | Model plays a "news editor"        |
| `chain_of_thought`  | Step-by-step reasoning             |
| `generative_config` | Stylized summary with tone control |

## Evaluation
Summaries are evaluated using:
- ROUGE-1
- ROUGE-2
- ROUGE-L
## Results Summary

| Prompt Type         | ROUGE-L Score |
|---------------------|---------------|
| `chain_of_thought`  | **0.2817**    |
| `role_based`        | 0.2793        |
| `zero_shot`         | 0.2793        |
| `few_shot`          | 0.2627        |
| `one_shot`          | 0.2525        |

`chain_of_thought` prompting achieved the highest ROUGE-L score, indicating it generated the most accurate summaries based on longest common subsequence matching.

---

## ROUGE Score Interpretation

The ROUGE-L scores range from **0.25 to 0.28**, which is normal and expected for a **prompt-only approach** using a general-purpose model like `google/flan-t5-base`.

- These results are **without any fine-tuning**, using just different prompting styles.
- In comparison, fine-tuned models like **BART** or **PEGASUS** can score **~0.40–0.45**, but they require labeled training data and supervised learning.
- The key insight here is not the absolute score, but the **relative performance between prompts**.

This validates that **`chain_of_thought`** prompting is more effective for this task than few-shot or one-shot approaches — even without tuning the model.

## Project Structure
.
├── config.py           # Model & dataset settings
├── prompts.py          # Prompt templates
├── runner.py           # Main execution file
├── requirements.txt    # Dependencies
├── README.md           # This file
└── results/            # Outputs (ignored in .gitignore)

## Technologies Used
- Flan-T5 (from Hugging Face)
- transformers
- datasets
- torch
- rouge-score
- Python 3.12+

## Acknowledgements

 - [Hugging Face](https://huggingface.co)
 - [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
 - [ROUGE Score](https://huggingface.co/evaluate-metric)


## Authors

- [@sourabhd321](https://github.com/sourabhd321)



## Badges

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Flan--T5-blueviolet)

