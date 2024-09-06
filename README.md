# Medical Chatbot Research Project

This repository contains code and resources for a research project focused on developing a medical chatbot. The project leverages various open-source frameworks, libraries, and tools from the Python ecosystem to fine-tune Large Language Models (LLMs) for medical-related tasks, and includes a Retrieval-Augmented Generation (RAG) system.

## Project Overview

This research falls under the category of studies that do not involve human participants. Despite its comparative nature, the study relies solely on standardized numerical performance metrics for analysis.

Key components of the project include:
- Use of GPUs for accelerated computational tasks.
- A well-configured system should ideally possess at least **15 GB of RAM** and **15-20 GB of GPU shared memory** to ensure optimal performance.
  
## Datasets

Two distinct datasets were used for fine-tuning the LLMs in this project:

- **Meadow-MedQA**: A publicly available dataset on Hugging Face with **10,178 training instances**. Each instance includes an input, instruction, and output.
- **MedMCQA**: A large-scale, multiple-choice question-answering dataset from Kaggle, consisting of **194,000 questions** across 21 medical subjects. The dataset is divided into training, testing, and validation sets in CSV format.

## Vector Database Architecture

The vector database was constructed from a diverse corpus of medical texts, including books, notes, and journals. Key components include:

- **Text and PDF Parsing**: Using NLP libraries like `LangChain` and `PyPDFLoader`, the documents were processed and split into smaller chunks of 500 characters with an overlap of 50 characters.
- **Embeddings**: The `gtr-t5-large` model was used to generate vector representations of the chunks. These embeddings were stored using the `FAISS` library.
- The vector database was stored in two files:
  - `index.faiss`: Contains index data.
  - `index.pkl`: Contains metadata and configuration details.

| **Hyperparameters/Characteristics** | **Value** |
|--------------------------------------|-----------|
| Chunk Size                           | 500       |
| Overlap Size                         | 50        |
| Embedding Model                      | `gtr-t5-large` |
| Computation Time                     | 7,560 Seconds |
| `index.faiss` File Size              | 753,412 KB |
| `index.pkl` File Size                | 122,143 KB |

## Fine-Tuning LLMs

This project focused on fine-tuning three types of LLMs to develop the chatbot:

1. **Flan-T5-Large**: An encoder-decoder model fine-tuned using PyTorch, Transformers, and PEFT libraries for sequence-to-sequence tasks. The model was loaded with 8-bit quantization and adapted for training with LoRA (Low-Rank Adaptation).
   
2. **LLaMA-2-7B**: A decoder-only architecture, designed for chat-based interactions, loaded with 4-bit quantization and LoRA adaptations for memory-efficient fine-tuning.

3. **Mistral-7B**: A GPTQ-quantized version used for optimal performance. Similar training methods were applied as for the LLaMA model.

## Chatbot System Design

### 1. Base Model with RAG

The base model used the `gtr-t5-large` embedding model and the `FAISS` vector database for Retrieval-Augmented Generation (RAG). Upon receiving a user query, the system performed the following steps:

- Embed the query and search for similar content in the vector database.
- Retrieve the top documents and use them as context for generating a response.

| **Parameters**            | **Value** |
|---------------------------|-----------|
| `do_sample`               | True      |
| `top_k`                   | 1         |
| `temperature`             | 0.1       |
| `max_new_tokens`          | 150       |

### 2. Finetuned Model without RAG

This model relied entirely on the fine-tuned LLM, without retrieving any context from a vector database. The system was simplified to load the model, tokenize user inputs, and generate responses based solely on the trained model's internal knowledge.

### 3. Finetuned Model with RAG

This design incorporated both the fine-tuned model and the RAG system, combining the advantages of both approaches. It allowed the chatbot to retrieve relevant medical information from the vector database before generating responses.

## Web-Based Chatbot Design

A user-friendly, web-based chatbot was developed using **Flask** for the backend, allowing for real-time interactions between the user and the model via a web interface. The frontend was built using HTML, CSS, and JavaScript, with functions like `sendMessage` and `displayMessage` to manage the input/output cycle.

---

### Citation and References
- **Meadow-MedQA** dataset available on Hugging Face.
- **MedMCQA** dataset available on Kaggle.

For more details, refer to the specific libraries and tools used:
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Flask](https://flask.palletsprojects.com/)
