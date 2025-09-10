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

## Usage

```python
from src.research_assistant import ResearchAssistant
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Initialize components
llm = OllamaLLM(model="llama3.2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./arxiv_vector1_db", embedding_function=embeddings)

# Create research assistant
assistant = ResearchAssistant(llm=llm, vector_db=vector_db)

# Ask a question
answer = assistant.process_query("What are the latest developments in transformer models?")
print(answer)```

## Usage

Notebooks (/notebooks) contain:

Query evaluation

Answer quality scoring

Statistical analysis (Friedman, Wilcoxon, Holm correction)

Database (/Database) includes:

arxiv_data/ → sample PDFs

arxiv_vector1_db/ → prebuilt Chroma vector database
