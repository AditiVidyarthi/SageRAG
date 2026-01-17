# File: 3_generate_systemD_and_merge.py
import pandas as pd
import os
import re
from tqdm import tqdm
from sagerag import ResearchAssistant, OllamaLLM, Chroma, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("--- 🚀 Starting System D Generation and Final Merge (Resumable) ---")

# --- 1. SETUP ---
print("1. Initializing models and database...")

# --- Configuration ---
DB_PATH = "Database/Vector-DB-New"
QUERY_STATS_INPUT_PATH = 'QueryStatistics.csv'
CHOICES_INPUT_PATH = 'system_D_choices.csv'
BASELINES_INPUT_PATH = 'baselines_A_B_C.csv' # <-- Corrected from your first script's output
SYSTEM_D_OUTPUT_PATH = 'system_D_answers.csv' # <-- Intermediate, resumable output
FINAL_OUTPUT_PATH = 'human_evaluation_data_final.csv'

# --- KEY CHANGE: Set device to 'cuda' for the server ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
llm = OllamaLLM(model="llama3.2", temperature=0.2)
assistant = ResearchAssistant(llm=llm, vector_db=vector_db)
print("✅ Setup Complete.")

# --- 2. HELPER FUNCTIONS (Preserved from your original script) ---
def generate_hypothetical_answer(assistant, query):
    """Generates a hypothetical answer to be used for retrieval (HyDE)."""
    hyde_prompt = PromptTemplate.from_template(
        "Generate a detailed, high-quality paragraph that is a hypothetical "
        "but plausible answer to the following research question. Do not say it is hypothetical. "
        "Just generate the answer directly.\n\nQuestion: {question}\n\nAnswer:"
    )
    chain = hyde_prompt | assistant.llm | StrOutputParser()
    return chain.invoke({"question": query})

def run_system_D_pipeline(assistant, original_query, selected_indices_str, df_rewrites):
    """Runs the full System D pipeline using a user's pre-selected query choices."""
    # Part 1: Reconstruct the query options to find what text was chosen
    all_candidates = df_rewrites[df_rewrites['Original Query'] == original_query]['Refined Query'].tolist()
    candidates = [c for c in all_candidates if c != original_query]
    
    query_options = [original_query] + candidates # Simplified list of actual query strings
    
    try:
        selected_indices = [int(idx.strip()) for idx in selected_indices_str.split(",")]
    except (ValueError, AttributeError):
        selected_indices = [0] # Default to original if choices are invalid

    selected_queries = [query_options[idx] for idx in selected_indices if 0 <= idx < len(query_options)]
    if not selected_queries:
        selected_queries = [original_query]

    # Part 2: Execute the RAG pipeline with HyDE
    hyde_document = generate_hypothetical_answer(assistant, selected_queries[0])
    
    if len(selected_queries) > 1:
        # Case 1: MULTIPLE queries selected -> Use combined answer generation
        _, query_docs_map = assistant.retrieve_documents(queries=selected_queries, k=3)
        return assistant.generate_combined_answer(original_query, query_docs_map)
    else:
        # Case 2: SINGLE query selected -> Use premium answer generation
        retrieved_docs, _ = assistant.retrieve_documents(queries=[hyde_document], k=5)
        return assistant.generate_premium_answer(original_query, retrieved_docs)

# --- 3. EXECUTION: GENERATE SYSTEM D ANSWERS ---

print(f"\n3. Loading choices and data...")
df_choices = pd.read_csv(CHOICES_INPUT_PATH)
df_full = pd.read_csv(QUERY_STATS_INPUT_PATH)
df_full.columns = df_full.columns.str.strip()
df_rewrites = df_full[['Original Query', 'Refined Query']].copy()

# --- KEY CHANGE: Start from the 11th query (index 10) ---
df_choices_to_process = df_choices.iloc[10:].copy()

# --- Resumability Logic ---
completed_queries = set()
if os.path.exists(SYSTEM_D_OUTPUT_PATH):
    try:
        df_existing_d = pd.read_csv(SYSTEM_D_OUTPUT_PATH)
        completed_queries = set(df_existing_d['Query'].tolist())
        print(f"Found {len(completed_queries)} previously generated System D answers. Resuming...")
        # Filter out already completed queries from the list to process
        df_choices_to_process = df_choices_to_process[~df_choices_to_process['Query'].isin(completed_queries)]
    except pd.errors.EmptyDataError:
        print(f"'{SYSTEM_D_OUTPUT_PATH}' is empty. Processing all queries in range.")

if df_choices_to_process.empty:
    print("\n--- ✅ All System D answers are already generated. Proceeding to merge. ---")
else:
    print(f"\n--- Generating {len(df_choices_to_process)} remaining System D answers ---")
    
    for index, row in tqdm(df_choices_to_process.iterrows(), total=df_choices_to_process.shape[0], desc="Generating System D Answers"):
        query = row['Query']
        choices_str = str(row['Selected Indices'])
        assistant.clear_memory()
        
        ans_D = run_system_D_pipeline(assistant, query, choices_str, df_rewrites)
        
        # --- KEY CHANGE: Append result immediately for safety ---
        new_result_df = pd.DataFrame([{'Query': query, 'System D Ans': ans_D}])
        file_exists = os.path.isfile(SYSTEM_D_OUTPUT_PATH)
        new_result_df.to_csv(
            SYSTEM_D_OUTPUT_PATH,
            mode='a',
            header=not file_exists or os.path.getsize(SYSTEM_D_OUTPUT_PATH) == 0,
            index=False,
            encoding='utf-8'
        )
    print("\n--- ✅ System D Generation Complete! ---")

# --- 4. MERGE RESULTS ---
print(f"\n4. Merging all results...")
try:
    df_baselines = pd.read_csv(BASELINES_INPUT_PATH)
    df_system_d = pd.read_csv(SYSTEM_D_OUTPUT_PATH)
    
    df_final = pd.merge(df_baselines, df_system_d, on='Query', how='left')
    
    # Save the final combined file
    df_final.to_csv(FINAL_OUTPUT_PATH, index=False, encoding='utf-8')
    
    print(f"\n--- ✅ All Done! ---")
    print(f"Final merged data for human evaluation saved to: '{FINAL_OUTPUT_PATH}'")

except FileNotFoundError as e:
    print(f"\n--- ❌ Error during merge: Could not find file {e.filename}. ---")
    print("Please ensure both 'baselines_A_B_C.csv' and 'system_D_answers.csv' are present before merging.")