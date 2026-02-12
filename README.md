# 🏦 RBI SLM Fine-Tuning with LoRA

This project demonstrates fine-tuning a Small Language Model (SLM) using
**LoRA (Low-Rank Adaptation)** on an RBI (Reserve Bank of India) focused
dataset.

The objective is to build a lightweight, domain-adapted language model
capable of answering RBI-related queries efficiently while minimizing
training cost and GPU memory usage.

------------------------------------------------------------------------

## 🚀 Project Overview

-   **Base Model:** Small Language Model (SLM)
-   **Fine-Tuning Method:** LoRA (Parameter-Efficient Fine-Tuning -
    PEFT)
-   **Training Type:** Supervised Fine-Tuning (SFT)
-   **Dataset:** RBI domain-specific Q&A dataset
-   **Optimization:** Quantization with BitsAndBytes

------------------------------------------------------------------------

## 🧰 Frameworks & Libraries Used

-   transformers\
-   peft\
-   trl\
-   accelerate\
-   datasets\
-   bitsandbytes\
-   torch\
-   huggingface_hub

------------------------------------------------------------------------

## 📂 Project Structure

    kaggleslm/
    │
    ├── myslm.ipynb                         # Fine-tuning notebook
    ├── rbi_sft_dataset_3000_corrected.json # RBI domain dataset
    ├── requirements.txt                    # Project dependencies
    └── README.md                           # Project documentation

------------------------------------------------------------------------

## 🛠 Installation

### 1️⃣ Create Virtual Environment

``` bash
python -m venv venv
```

Activate environment:

**Windows:**

``` bash
venv\Scripts\activate
```

------------------------------------------------------------------------

### 2️⃣ Install Dependencies

Using requirements file:

``` bash
pip install -r requirements.txt
```

Or manually:

``` bash
pip install -U bitsandbytes transformers accelerate peft trl datasets torch huggingface_hub
```

------------------------------------------------------------------------

## 🧠 Fine-Tuning Approach

-   Implemented **LoRA (Low-Rank Adaptation)** for efficient
    fine-tuning\
-   Reduced number of trainable parameters significantly\
-   Applied **Supervised Fine-Tuning (SFT)**\
-   Used **BitsAndBytes quantization (4-bit / 8-bit)** for optimized
    memory usage\
-   Designed for efficient GPU utilization

------------------------------------------------------------------------

## 📊 Dataset

The dataset contains RBI-related domain-specific question-answer pairs.

### Format

``` json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

This structure enables instruction-based supervised fine-tuning.

------------------------------------------------------------------------

## 🔐 Security Note

Hugging Face access tokens are **NOT hardcoded** in the notebook.

Use environment variables:

``` python
from huggingface_hub import login
import os

login(os.getenv("HF_TOKEN"))
```

Set token in terminal:

**Windows:**

``` bash
set HF_TOKEN=your_token_here
```

------------------------------------------------------------------------

## 🎯 Key Learning Outcomes

-   Practical implementation of LoRA on SLM\
-   Domain-specific model adaptation\
-   Efficient GPU memory optimization\
-   Secure token management\
-   Clean Git repository workflow

------------------------------------------------------------------------

## 👩‍💻 Author

**Shruti Johnson**\
GitHub: https://github.com/shrutijohnson1306-blip

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Add evaluation metrics\
-   Compare base vs fine-tuned model performance\
-   Build inference API\
-   Add quantized inference pipeline
