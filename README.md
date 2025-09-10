# SageRAG
An AI research assistant for scientific papers using RAG

# AI Research Assistant

A conversational AI research assistant built with RAG (Retrieval-Augmented Generation) for scientific papers from arXiv.

## Features

- Query analysis and improvement suggestions
- Multi-query retrieval from scientific papers
- Conversational memory for follow-up questions
- Document relevance assessment
- Semantic search with re-ranking

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your environment variables in a `.env` file

## Reproducibility

Notebooks (/notebooks) contain:

Creating Vector DB

Query evaluation by LLM

ResearchPro1 -- the main program

Statistic Significance Test (Friedman, Wilcoxon, Holm correction)

Database (/Database) includes:

arXiv_papers/ → sample PDFs

Vector_DB/ → prebuilt Chroma vector database

## Usage

```python
from src.research_assistant import ResearchAssistant
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize components
llm = OllamaLLM(model="llama3.2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="../Database/Vector_DB", embedding_function=embeddings)

# Create research assistant
assistant = ResearchAssistant(llm=llm, vector_db=vector_db)

# Ask a question
answer = assistant.process_query("What are the latest developments in transformer models?")
print(answer)
```

## 📄 Reference

This repository accompanies the paper describing SageRAG titled: SageRAG: Query Rewriting for Retrieval Enhancement and Retrieval-Augmented Generation for Grounded Responses in AI Research Assistance.  
Once published, please cite the official version of the article.  
