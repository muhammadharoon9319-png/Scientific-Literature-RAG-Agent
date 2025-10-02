# MedAgent RAG: Scientific Literature Query Assistant

This project provides an advanced, AI-powered Retrieval-Augmented Generation (RAG) workflow dedicated to answering complex biomedical and scientific questions. It functions by intelligently retrieving relevant PubMed abstracts and grounding the responses with powerful reasoning from DeepSeek R1 distilled models to ensure reliability and context-awareness.
The application features an intuitive Streamlit dashboard where users can interactively query the medical literature, review the retrieved sources, and instantly receive synthesized, evidence-backed insights.

---

## 🌟 Key Features
-**Intelligent Literature Retrieval:** A Smart RAG pipeline fetches necessary PubMed abstracts and integrates them into the model's reasoning process.

- **Flexible Reasoning Models:** Supports multiple DeepSeek R1 distilled variants, allowing users to choose between models optimized for speed or precision.

- **Evidence-Backed Answers:** Every synthesized response is linked and cites the PubMed abstracts used for verification and context.

- **Interactive Streamlit Dashboard:** Provides a simple interface for entering queries, exploring retrieved documents, and selecting the preferred model.

-  **Seamless Model Setup:** Required models are downloaded automatically upon the application's first launch.

-----

--
## ⚙️ Installation & Usage
- Step 1: Set Up the Environment
- Begin by creating a new Python environment using Conda:

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
streamlit run MedAgent_rag_app.py
```

> **Note:** On the first run, the selected models will be automatically downloaded.

## Available Models

Currently, the application supports the following DeepSeek R1 Distilled models:

- `DeepSeek-R1-Distill-Llama-8B`
- `DeepSeek-R1-Distill-QWen-7B`
- `DeepSeek-R1-Distill-QWen-14B`

Select your desired model directly through the Streamlit UI.
