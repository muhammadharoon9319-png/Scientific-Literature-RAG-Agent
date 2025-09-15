# PubMed Agentic RAG for Scientific & Medical Queries

This repository provides an **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed to answer scientific and medical questions by retrieving relevant **PubMed abstracts** and combining them with large language model reasoning. It leverages **DeepSeek R1 distilled models** for accurate, context-aware responses.

The application includes a **Streamlit-based interactive interface** where users can query PubMed literature in real time, explore retrieved abstracts, and receive synthesized insights.

---

## 🚀 Features

- 🔎 **Agentic RAG pipeline** – retrieves PubMed abstracts and integrates them into model reasoning  
- 🧠 **Supports multiple DeepSeek R1 models** – choose between different model sizes for speed vs. accuracy  
- 📑 **Cites PubMed abstracts** – see which studies were used to generate answers  
- 🎛️ **Interactive Streamlit UI** – simple interface for entering queries, exploring results, and switching models  
- ⚡ **Automatic model downloads** – first-time use will fetch selected models automatically  

---
## Setup

### Step 1: Create a Conda Environment

Create a new conda environment:

```bash
conda create -n agentic_rag python==3.12
conda activate agentic_rag
```

### Step 2: Install Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

Start the Streamlit application with:

```bash
streamlit run agentic_rag_streamlit_app.py
```

> **Note:** On the first run, the selected models will be automatically downloaded.

## Available Models

Currently, the application supports the following DeepSeek R1 Distilled models:

- `DeepSeek-R1-Distill-Llama-8B`
- `DeepSeek-R1-Distill-QWen-7B`
- `DeepSeek-R1-Distill-QWen-14B`

Select your desired model directly through the Streamlit UI.
