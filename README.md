# 🧠 Educational Repository for Fine-Tuning Large Language Models (LLMs) — PyTorch Only

Welcome to the **LLM Fine-Tuning Educational Repository**, built entirely with **pure PyTorch** — no HuggingFace, no external frameworks. This project is designed to provide a hands-on, from-scratch learning experience for understanding and fine-tuning large language models. Perfect for learners who want to build a solid foundation by implementing everything step by step.

---

## 📚 What You'll Learn

- Core architecture and internals of LLMs and SLMs
- Dataset handling, preprocessing and multi-gpu training with PyTorch
- Building tokenizers and vocabularies from scratch
- Implementing training loops and loss functions
- Fine-tuning techniques without relying on external libraries
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- DPO (Direct Preference Optimization) for aligning models using preference data

---

## 🛠️ Repository Structure

```
.
├── demo/                 # Markdowns for demos
├── scripts/              # Utility scripts
├── slm_full_sft.py       # Supervised Full Fine-Tuning Smollm2
├── slm_lora_sft.py       # Supervised Fine-Tuning Smollm2 via LoRA
├── llm_lora_sft.py       # Supervised Fine-Tuning Gemma3 via LoRA
└── README.md             # This file
```

## 🧪 Demos
| Markdown                           | Description                             |
| ---------------------------------- | --------------------------------------- |
| `YugiohGPT.md`                     | Yugioh Card generation via LLM          |


## 💡 Why No HuggingFace?
This repo is intended for educational purposes. By not using external libraries, you’ll:

- Learn how everything works under the hood
- Gain deep insight into training dynamics and model architecture
- Build skills that translate to research and custom implementations