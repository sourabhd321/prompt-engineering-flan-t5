import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
SPLIT = "test"
NUM_EXAMPLES = 50


# Generative configurations
GEN_CONFIGS = {
    "default": {"temperature": 0.7, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    "creative": {"temperature": 1.0, "top_p": 0.95, "frequency_penalty": 0.2, "presence_penalty": 0.2},
    "precise": {"temperature": 0.3, "top_p": 0.7, "frequency_penalty": 0.5, "presence_penalty": 0.3}
}