import pandas as pd
import re
import os # Import os for file checking
from tqdm import tqdm
from sagerag import ResearchAssistant, OllamaLLM, Chroma, HuggingFaceEmbeddings # Assuming your class is in sagerag.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.vectorstores import VectorStore
from langchain.chains import LLMChain
# from langchain.llms.base import BaseLLM
from langchain.schema.runnable import Runnable
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import requests
import os
import uuid
import time

print("---Starting Automated Baseline Data Generation (Batched) ---")

# --- 1. SETUP ---
print("1. Initializing models and database...")
DB_DIR_NEW = "../Database/Vector-DB-New"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
vector_db = Chroma(persist_directory=DB_DIR_NEW, embedding_function=embedding_model)
llm = OllamaLLM(model="llama3.2", temperature=0.2) # Make sure Ollama is set up on Colab or use a Colab-friendly LLM
assistant = ResearchAssistant(llm=llm, vector_db=vector_db)
print("Setup Complete.")

# --- 2. HELPER FUNCTIONS ---
# (Include generate_hypothetical_answer, run_rag_A_and_B_pipeline, run_rag_C_pipeline,
#  get_answer_A_original, get_answer_B_zero_shot, get_answer_C_single_pass_best_from_csv
#  exactly as defined in our previous correct version)
def generate_hypothetical_answer(assistant, query):
    hyde_prompt = PromptTemplate.from_template(
        "Generate a detailed, high-quality paragraph that is a hypothetical "
        "but plausible answer to the following research question. Do not say it is hypothetical. "
        "Just generate the answer directly.\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = hyde_prompt | assistant.llm | StrOutputParser()
    return chain.invoke({"question": query})

def run_baseline_rag_pipeline(assistant, query_for_retrieval, original_question):
    """A non-interactive function to run the retrieve-and-generate pipeline."""
    try:
        retrieved_docs, _ = assistant.retrieve_documents(queries=[query_for_retrieval], k=5)
        if not retrieved_docs:
            return "Error: Could not retrieve any relevant documents."
        final_answer = assistant.generate_standard_answer(
            question=original_question,
            docs=retrieved_docs,
            max_docs=5
        )
        return final_answer
    except Exception as e:
        return f"Error in RAG pipeline: {e}"
def run_rag_C_pipeline(assistant, query_for_retrieval, original_question):
    """The MID-TIER RAG pipeline for System C. Uses the CITED prompt."""
    try:
        # Retrieve documents using the query directly
        retrieved_docs, _ = assistant.retrieve_documents(queries=[query_for_retrieval], k=5)
        if not retrieved_docs:
            return "Error: Could not retrieve any relevant documents."
        
        # Call the mid-tier generator with citations
        return assistant.generate_cited_answer(original_question, retrieved_docs)
    except Exception as e:
        return f"Error in RAG pipeline: {e}"

def get_answer_A_original(assistant, original_query):
    """System A: Just use the original query."""
    return run_baseline_rag_pipeline(assistant, original_query, original_query)

def get_answer_B_zero_shot(assistant, original_query):
    """System B: Use a simple zero-shot rewrite."""
    zero_shot_prompt = PromptTemplate.from_template(
        "You are an AI assistant. Rewrite the following query to be more effective for a "
        "semantic search over academic papers. Provide only the single improved query. "
        "Original Query: {question}\nRewritten Query:"
    )
    chain = zero_shot_prompt | assistant.llm | StrOutputParser()
    rewritten_query = chain.invoke({"question": original_query}).strip().strip('"')
    return run_baseline_rag_pipeline(assistant, rewritten_query, original_query)

def get_answer_C_single_pass_best(assistant, original_query, df_rewrites):
    """System C: Generate 5 rewrites and auto-select the one with the highest similarity score."""
    
    all_candidates = df_rewrites[df_rewrites['Original Query'] == original_query]['Refined Query'].tolist()
    candidates = [c for c in all_candidates if c != original_query]
    if not candidates:
        return "Error: Could not find pre-generated rewrites for this query."

    scored_candidates = assistant.score_queries(original_query, candidates)
    best_query = scored_candidates[0][0] # The list is sorted, so the first one is the best

    print(f"Auto-Selected Best Rewrite (Score: {scored_candidates[0][1]:.2f}): {best_query}")
    return run_rag_C_pipeline(assistant, best_query, original_query)

# --- 3. EXECUTION ---
print("\n3. Starting the main execution loop...")
# --- Configuration ---
INPUT_CSV_PATH = 'QueryStatistics.csv'
OUTPUT_CSV_PATH = 'baselines_A_B_C.csv' # Output file
TOTAL_SAMPLES_NEEDED = 200
SAMPLES_ALREADY_DONE = 10 # Set to 0 if starting fresh
BATCH_SIZE = 40 # Process 40 queries per batch

# --- Load Data and Determine Queries to Process ---
df_full = pd.read_csv(INPUT_CSV_PATH)
df_full.columns = df_full.columns.str.strip()
unique_queries = df_full['Original Query'].unique().tolist()

# Calculate the slice of queries to process in this run
start_index = SAMPLES_ALREADY_DONE
end_index = TOTAL_SAMPLES_NEEDED
queries_to_process = unique_queries[start_index:end_index]

print(f"Loaded {len(unique_queries)} unique queries.")
print(f"Targeting {TOTAL_SAMPLES_NEEDED} total samples.")
print(f"Already completed: {SAMPLES_ALREADY_DONE}.")
print(f"Will process {len(queries_to_process)} queries in this run (indices {start_index} to {end_index-1}).")

# Prepare the lookup DataFrame for System C
df_rewrites = df_full[['Original Query', 'Refined Query']].copy()

# --- Batch Processing Loop ---
all_results_for_run = []
for i in tqdm(range(0, len(queries_to_process), BATCH_SIZE), desc="Processing Batches"):
    batch_queries = queries_to_process[i:i + BATCH_SIZE]
    batch_results = []
    
    print(f"\n--- Starting Batch {i // BATCH_SIZE + 1} ({len(batch_queries)} queries) ---")
    
    for query in tqdm(batch_queries, desc=f"Batch {i // BATCH_SIZE + 1}", leave=False):
        assistant.clear_memory()
        
        # Check if the query exists in df_rewrites to avoid errors
        if query not in df_rewrites['Original Query'].values:
            print(f"Warning: Skipping query '{query}' as it's not found in the rewrite lookup table.")
            continue

        ans_A = get_answer_A_original(assistant, query)
        ans_B = get_answer_B_zero_shot(assistant, query)
        ans_C = get_answer_C_single_pass_best(assistant, query, df_rewrites)
        
        batch_results.append({
            'Query': query,
            'System A Ans': ans_A,
            'System B Ans': ans_B,
            'System C Ans': ans_C
        })
        all_results_for_run.extend(batch_results) # Keep track of all results from this run

    # --- Append results of the current batch to the CSV ---
    df_batch = pd.DataFrame(batch_results)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(OUTPUT_CSV_PATH)
    
    df_batch.to_csv(
        OUTPUT_CSV_PATH, 
        mode='a',        # 'a' for append
        header=not file_exists, # Write header only if file doesn't exist
        index=False, 
        encoding='utf-8'
    )
    print(f"--- Batch {i // BATCH_SIZE + 1} completed. Results appended to '{OUTPUT_CSV_PATH}'. ---")

print(f"\n---Automated Baselines Generation Complete! ---")
print(f"Processed {len(all_results_for_run)} queries in this run.")
print(f"All results saved to: '{OUTPUT_CSV_PATH}'")