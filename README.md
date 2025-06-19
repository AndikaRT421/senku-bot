# SenkuBot

SenkuBot is an **open-source AI assistant** that recognizes and classifies mineral specimens from images, then enriches its answers with web search and multimodal language-model reasoning for deeper insights. It is designed to **synchronize effortlessly with existing mineral-management systems** for fast, automated data entry and verification.

## ✨ Key Features
| Feature | What it does | Tech behind it |
|---------|--------------|----------------|
| Mineral identification | Detects minerals such as Biotite, Bornite, Chrysocolla, Malachite, Pyrite, Quartz, and Muscovite from a single photo | **DenseNet-121** CNN classifier |
| Multimodal Q&A | Lets users ask follow-up questions about the sample (e.g., chemical makeup, common uses) | **Gemma-3** vision-language model |
| Dynamic fusion | Blends search results with model output for up-to-date facts | Retrieval-augmented generation pipeline |
| Seamless sync | Hooks into existing databases / dashboards | REST & gRPC adapters |

## 📂 Dataset
We fine-tune on the [**Minerals Identification Dataset**](https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset/data) from Kaggle (2,800+ labelled images).

## 👥 Authors
- Andika Rahman Teja
- Malvin Leonardo Hartanto
- Rafli Raihan Pramudya
