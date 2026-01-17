from langsmith import traceable
from langchain.callbacks.manager import tracing_v2_enabled
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
from langchain.schema import Document
import requests
import os
import re
import uuid
import time
from typing import List, Literal
import re
import json
from typing import Dict, List, Tuple, Optional, Any
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

# Suppress library-specific logging
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('LangChainDeprecationWarning').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Then set up your own logging as before
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SageRAG")

# Rich console for prettier output
console = Console()

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse output that contains a numbered list and return as a list of strings."""
    
    def parse(self, text: str) -> List[str]:
        """Parse text into a list of strings."""
        # Updated regex to be more robust with various numbering styles
        lines = re.findall(r"^\s*\d+\.?\s+(.*?)$", text, re.MULTILINE)
        return [line.strip() for line in lines if line.strip()]
    
    @property
    def _type(self) -> str:
        return "line_list"

class QueryRating(BaseModel):
    """Schema for query rating output."""
    rating: int = Field(description="Rating from 1-5")
    explanation: str = Field(description="Explanation for the rating")

@dataclass
class RetrievedDocument:
    """Dataclass for tracking retrieved document info."""
    document: Document
    score: float
    query: str = ""
    rank: int = 0


class ResearchAssistant:
    """AI Research Assistant using RAG for scientific paper queries."""
    
    def __init__(
        self, 
        llm: OllamaLLM, 
        vector_db: VectorStore, 
        embeddings_model: Optional[SentenceTransformer] = None,
        relevance_threshold: float = 0.2,
        max_docs_per_query: int = 4
    ):
        """Initialize the Research Assistant with LLM and vector database.
        
        Args:
            llm: The language model to use for generation
            vector_db: Vector database for document retrieval
            embeddings_model: Optional SentenceTransformer model for semantic similarity
            relevance_threshold: Minimum relevance score for documents
            max_docs_per_query: Maximum documents to use per query
        """
        self.llm = llm
        self.vector_db = vector_db
        self.relevance_threshold = relevance_threshold
        self.max_docs_per_query = max_docs_per_query
        
        # Initialize sentence transformer model for semantic similarity if not provided
        if embeddings_model is None:
            logger.info("Initializing default embedding model")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.semantic_model = embeddings_model
            
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create a retriever for direct use
        self.retriever = vector_db.as_retriever()
        
        # Define parsers
        self.json_parser = JsonOutputParser(pydantic_object=QueryRating)
        self.list_parser = LineListOutputParser()
        
        logger.info("Research Assistant initialized successfully")

    @traceable(name="rate_query")
    def rate_query(self, query: str) -> Dict[str, Any]:
        """Rates the query and gives an explanation.
        
        Args:
            query: The user's query to be evaluated
            
        Returns:
            Dict containing rating and explanation
        """
        prompt = PromptTemplate.from_template("""
        You are an intelligent assistant trained to evaluate search queries for a scientific research database.
        Given the following query: "{query}"
        
        1. Rate the query on a scale of 1 (very poor) to 5 (excellent) based on:
           - Clarity: Is the query clear and unambiguous?
           - Specificity: Does it contain specific technical terms or concepts?
           - Relevance: Is it focused on retrieving scientific content?
           - Retrievability: Will it work well with vector search?
        
        2. Provide a short explanation for the rating (what makes it effective or ineffective).
        
        Respond in JSON format:
        {{
          "rating": <number between 1-5>,
          "explanation": "<your explanation>"
        }}
        """)
        
        chain: Runnable = prompt | self.llm | self.json_parser
        
        try:
            return chain.invoke({"query": query})
        except Exception as e:
            logger.error(f"Error rating query: {e}")
            # Fallback response if parsing fails
            return {
                "rating": 3, 
                "explanation": "Unable to rate query properly. Consider adding more specific terms."
            }

    @traceable(name="suggest_rewrites")
    def suggest_rewrites(self, query: str, chat_history: Optional[List] = None) -> List[str]:
        """Returns rephrased versions of the query optimized for retrieval.
        
        Args:
            query: Original query to rewrite
            chat_history: Optional conversation history for context
            
        Returns:
            List of rewritten queries
        """
        history_context = ""
        # console.print(f"this is working yesss")
        if chat_history:
            history_context = "Consider this conversation context when rewriting the query:\n"
            for message in chat_history[-3:]:  # Use only last 3 messages for brevity
                if hasattr(message, "content"):
                    role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                    history_context += f"{role}: {message.content}\n"
        
        prompt = PromptTemplate(
            input_variables=["question", "history_context"],
            template="""You are an AI assistant specializing in scientific research queries.
        {history_context}
        TASK: Create 5 alternative versions of the user's question that will help improve retrieval from a scientific paper database.
        Original question: {question}

        For each alternative version:
        Write a COMPLETE, EXECUTABLE query (not just a description)
        Make each version meaningfully different while preserving the core information need
        Include the ACTUAL REWRITTEN QUESTION text

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS EXAMPLE:

        Original question: batch size affect what?
        1. Rewrite 1: [Short description of the rewrite strategy]: "[Rewritten query text]"

        Here is one of the high-quality examples provided in the prompt to guide the model:
        1. **Adding technical terminology for specificity: "What are the effects of batch size on model convergence and optimization in deep learning neural networks?"
        2. **Decomposition into multiple sub-questions**: "How does the batch size parameter influence the performance of stochastic gradient descent algorithms? What impact does it have on the training time and accuracy of machine learning models?" 
        3. **Expanding the query with related concepts**: "Examine the effects of batch size, mini-batch size, and data partitioning on the generalization capabilities and robustness of deep neural networks."
        4. **Focusing on a specific aspect or trade-off**: "In what ways do changes in batch size affect the trade-off between model training speed and accuracy in deep learning architectures?"
        5. **Specifying key entities and relationships**: "Analyze the causal relationship between batch size, neural network architecture, and optimization algorithms (e.g., Adam, SGD) on the convergence of training objectives and model performance metrics."

        Now, for the question "{question}", provide 5 complete, executable rewritten queries following the exact format above. Each rewrite must include the full rewritten question text, not just a description of how to rewrite it.

        """
        )
        
        try:
            result = (prompt | self.llm | self.list_parser).invoke({
                "question": query,
                "history_context": history_context
            })
            return result
        except Exception as e:
            logger.error(f"Error suggesting rewrites: {e}")
            # Return a minimal set of rewrites if parsing fails
            return [
                query,  # Original query
                f"research about {query}",
                f"papers discussing {query}"
            ]
    def compute_confidence(self, original: str, rewritten: str) -> float:
        """Computes semantic similarity between original and rewritten queries.
        
        Args:
            original: Original query
            rewritten: Rewritten version
            
        Returns:
            Similarity score between 0-1
        """
        try:
            vec_orig = self.semantic_model.encode(original, convert_to_tensor=True)
            vec_rewrite = self.semantic_model.encode(rewritten, convert_to_tensor=True)
            return float(util.pytorch_cos_sim(vec_orig, vec_rewrite).item())
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.6  # Default reasonable value
    
    def score_queries(self, original: str, queries: List[str]) -> List[Tuple[str, float]]:
        """Scores and sorts rephrased queries by confidence.
        
        Args:
            original: Original query
            queries: List of rewritten queries
            
        Returns:
            List of (query, score) tuples sorted by score
        """
        scored = [(q, self.compute_confidence(original, q)) for q in queries]
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def present_query_options(self, original: str, queries: List[str]) -> List[Tuple[str, float]]:
        """Presents the original and rewritten queries with confidence scores.
        
        Args:
            original: Original query
            queries: List of rewritten queries
            
        Returns:
            List of (query, score) tuples including original query
        """
        # Add original query at the top
        scored_queries = [("Original: " + original, 1.0)]
        
        # Add scored rewritten queries
        rewritten_scores = self.score_queries(original, queries)
        for i, (query, score) in enumerate(rewritten_scores, 1):
            if "Query:" in query:
                parts = query.split("Query:")
                # Format with Query on a new line and bold
                formatted_query = f"{parts[0]}\n[bold]Query:[/bold]{parts[1]}"
                scored_queries.append((f"Rewrite {i}: {formatted_query}", score))
            else:
                scored_queries.append((f"Rewrite {i}: {query}", score))
            
        # Display options in a nice format
        console.print("\n[bold cyan]Available search queries:[/bold cyan]")
        for i, (query, score) in enumerate(scored_queries):
            confidence_color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
            console.print(f"[bold]{i}.[/bold] {query}")
            console.print(f"   [bold {confidence_color}]Confidence: {score:.2f}[/bold {confidence_color}]")
            
        return scored_queries
    
    @traceable(name="retrieve_documents")
    def retrieve_documents(
        self, 
        queries: List[str], 
        k: int = 5
    ) -> Tuple[List[RetrievedDocument], Dict[str, List[RetrievedDocument]]]:
        """Retrieves documents using multiple queries, preserving query information.
        
        Args:
            queries: List of queries to retrieve documents for
            k: Number of documents to retrieve per query
            
        Returns:
            Tuple of (all unique documents, query->documents mapping)
        """
        all_docs: List[RetrievedDocument] = []
        unique_content = set()
        query_docs_map: Dict[str, List[RetrievedDocument]] = {}
        
        # Retrieve docs for each query
        for query in queries:
            console.print(f"\n[bold blue]Query:[/bold blue] '{query}'")
            
            try:
                results = self.vector_db.similarity_search_with_relevance_scores(query, k=k)
                print("LOG123: this is results length",len(results))
                docs_for_query: List[RetrievedDocument] = []
                
                for i, (doc, score) in enumerate(results, 1):
                    # Create retrieved document object
                    retrieved_doc = RetrievedDocument(
                        document=doc,
                        score=score,
                        query=query,
                        rank=i
                    )
                    
                    # Display result info
                    score_color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
                    console.print(f"[bold]--- Result {i} ---[/bold]")
                    console.print(f"[bold {score_color}]Score: {score:.4f}[/bold {score_color}]")
                    
                    # Show document preview
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    console.print(Panel(preview, title="Content Preview", width=100))
                    
                    # Show metadata if available
                    if hasattr(doc, 'metadata') and doc.metadata:
                        source = doc.metadata.get('source', 'Unknown')
                        console.print(f"[dim]Source: {source}[/dim]")
                    
                    # Add to results for this query
                    docs_for_query.append(retrieved_doc)
                    
                    # Only add unique documents to overall collection
                    if doc.page_content not in unique_content:
                        unique_content.add(doc.page_content)
                        all_docs.append(retrieved_doc)
                query_docs_map[query] = docs_for_query
                
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
                console.print(f"[bold red]Error retrieving documents for query: {query}[/bold red]")
        
        console.print(f"\n[bold green]Total unique documents:[/bold green] {len(all_docs)}")
        return all_docs, query_docs_map

    def rerank_docs(self, query: str, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Reranks documents based on semantic similarity to the query.
        
        Args:
            query: Query to compare documents against
            docs: List of retrieved documents
            
        Returns:
            Reranked list of documents with updated scores
        """
        try:
            query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
            reranked = []
            
            for doc in docs:
                doc_embedding = self.semantic_model.encode(doc.document.page_content, convert_to_tensor=True)
                new_score = float(util.pytorch_cos_sim(query_embedding, doc_embedding).item())
                
                # Create new RetrievedDocument with updated score
                reranked_doc = RetrievedDocument(
                    document=doc.document,
                    score=new_score,
                    query=doc.query,
                    rank=0  # Will be updated after sorting
                )
                reranked.append(reranked_doc)
            
            # Sort by score and update ranks
            reranked.sort(key=lambda x: x.score, reverse=True)
            for i, doc in enumerate(reranked, 1):
                doc.rank = i
                
            return reranked
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return docs  # Return original documents if reranking fails
    
    def can_answer_without_retrieval(self, question: str) -> Tuple[bool, Optional[str]]:
        """Determines if a question can be answered directly from memory without retrieval.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (can_answer, answer_if_available)
        """
        # Get chat history from memory if available
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        if not chat_history:
            return False, None
            
        # Create the prompt to check if we can answer directly from the conversation
        prompt = PromptTemplate.from_template("""
        You are an AI assistant helping with a scientific research conversation.
        Given the following conversation history and a new question, determine if the question:
        1. Can be answered directly based ONLY on the conversation history (like "what was my previous question?")
        2. Does NOT require retrieving new information from scientific papers
        
        If both conditions are true, provide the answer. Otherwise, respond with "NEEDS_RETRIEVAL".
        
        Conversation History:
        {chat_history}
        
        New Question: {question}
        
        Your assessment (answer directly or respond with "NEEDS_RETRIEVAL"):
        """)
        
        # Format the chat history for context
        history_str = ""
        for message in chat_history[-5:]:  # Use only last 5 messages for brevity
            if hasattr(message, "content"):
                role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                history_str += f"{role}: {message.content}\n"
        
        try:
            # Ask the LLM if this can be answered without retrieval
            response = self.llm.invoke(
                prompt.format(chat_history=history_str, question=question)
            )
            
            # If the response is "NEEDS_RETRIEVAL", we need to use retrieval
            if "NEEDS_RETRIEVAL" in response:
                return False, None
            else:
                logger.info("Question can be answered from conversation history")
                return True, response
        except Exception as e:
            logger.error(f"Error determining if retrieval needed: {e}")
            return False, None  

    @traceable(name="generate_standard_answer")
    def generate_standard_answer(self, question: str, docs: List[RetrievedDocument], max_docs: int = 5) -> str:
        """
        Generates a standard RAG answer. This version ALWAYS uses the retrieved context
        and does NOT have an LLM fallback, for fair experimental comparison.
        """
        if not docs:
            return "Error: No documents were retrieved, so a grounded answer cannot be generated."
    
        # Chat history is removed as it's cleared in the evaluation loop
        chat_context = ""
        
        # Format document content
        context_pieces = [f"Document {i+1}:\n{doc.document.page_content}" for i, doc in enumerate(docs[:max_docs])]
        context = "\n\n---\n\n".join(context_pieces)
        
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are a helpful assistant. Use ONLY the following pieces of context to answer the question at the end. 
    If the information is not in the context, explicitly state that the provided documents do not contain the answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
        )
        
        try:
            chain = prompt | self.llm
            answer = chain.invoke({"question": question, "context": context})
            return answer
        except Exception as e:
            logger.error(f"Error generating standard answer: {e}")
            return f"Error during final answer generation: {e}"

    def generate_cited_answer(self, question: str, docs: List[RetrievedDocument], max_docs: int = 5) -> str:
        """
        Generates a RAG answer with enforced citations (for System C).
        """
        if not docs:
            return "Error: No documents were retrieved for this query."
    
        # Use a clear context format
        context_pieces = []
        for i, doc in enumerate(docs[:max_docs]):
            source_link = doc.document.metadata.get('source', 'Unknown Source')
            context_pieces.append(f"Document {i+1} ({source_link}):\n{doc.document.page_content}")
        context = "\n\n---\n\n".join(context_pieces)
        
        # This is the new mid-tier prompt
        cited_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are a helpful scientific research assistant. Use the following retrieved context to answer the user's question.
    
            Guidelines:
            - Synthesize information from the documents to provide a comprehensive answer.
            - For every piece of information you use, you MUST cite the document it came from (e.g., "(Document 1)") and at the end mention the source url Document 1: url..
            - If the information is not in the context, state that clearly.
    
            Context:
            {context}
    
            Question: {question}
    
            Answer:
            
            """
        )
        
        try:
            chain = cited_prompt | self.llm
            return chain.invoke({"question": question, "context": context})
        except Exception as e:
            logger.error(f"Error generating cited answer: {e}")
            return f"Error during generation: {e}"

    @traceable(name="generate_premium_answer")
    def generate_premium_answer(
        self, 
        question: str, 
        docs: List[RetrievedDocument], 
        max_docs: int = 5
    ) -> str:
        """
        Generates a high-quality, structured answer with enforced citations,
        broader context, and a formatted source list. This version is optimized
        for the controlled evaluation experiment.
        """
        if not docs:
            return "Error: No documents were provided to the generation step."
    
        # --- 1. CORRECT CONTEXT FORMATTING ---
        # Create clearly labeled context chunks with the source link in the header.
        context_pieces = []
        for i, doc in enumerate(docs[:5]): # Use a hardcoded limit of 5 docs
            source_link = doc.document.metadata.get('source', 'Unknown Source')
            context_pieces.append(
                f"--- START: Document {i+1} ({source_link}) ---\n"
                f"{doc.document.page_content}\n"
                f"--- END: Document {i+1} ---"
            )
        context = "\n\n".join(context_pieces)
        
        # --- 2. DEFINE THE PREMIUM PROMPT ---
        premium_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert scientific research assistant. Your task is to answer the user's question with accuracy and clarity, using the provided research paper excerpts.
    
            **CONTEXT FROM RESEARCH PAPERS:**
            {context}
    
            **USER'S QUESTION:** {question}
    
            **INSTRUCTIONS:**
            1.  **Direct Answer:** Write a comprehensive answer based ONLY on the provided "CONTEXT". Synthesize information across multiple documents. If the context contains conflicting information, highlight the disagreement. If the information is not in the context, state that clearly.
            2.  **In-Text Citations:** For every claim you make, you MUST cite the source using its identifier (e.g., "(Source: Document 1)").
            3.  **Broader Context:** After the direct answer, add a section titled "Broader Context". Here, provide additional perspective based on your general knowledge.
            4.  **Sources List:** Finally, create a section titled "Sources Used". List all the documents you cited by their full source link.
    
            **RESPONSE FORMAT:**
    
            **Direct Answer (from sources):**
            [Your synthesized answer with in-text citations.]
    
            **Broader Context:**
            [Your supplementary explanation.]
    
            **Sources Used:**
            - Document 1: [Full source link from the context]
            - Document 2: [Full source link from the context]
            """
        )
        
        # --- 3. GENERATE THE ANSWER ---
        try:
            chain = premium_prompt | self.llm
            # --- START: DEBUGGING ---
            # Add this print statement to see what's going in
            print(f"DEBUG: Context length: {len(context)}, Question: {question}")
            # --- END: DEBUGGING ---

            answer = chain.invoke({"question": question, "context": context})
            
            # --- START: ROBUST GUARDRAIL (CRITICAL FIX) ---
            # This catches the silent 'None' failure
            if not answer or not isinstance(answer, str) or answer.strip() == "":
                # Manually raise an exception so your 'except' block can catch it
                raise Exception("LLM returned an empty or None response. This is likely due to a context window overflow.")
            # --- END: ROBUST GUARDRAIL ---

            # Save to memory
            self.memory.chat_memory.add_messages([
                HumanMessage(content=question),
                AIMessage(content=answer) 
            ])
            
            # Return the valid string
            return answer
        except Exception as e:
            logger.error(f"Error generating premium answer: {e}")
            return f"Error during final answer generation: {e}"

    @traceable(name="generate_combined_answer")
    def generate_combined_answer(self, question: str, query_results: Dict[str, List[RetrievedDocument]]) -> str:
        """
        Generates a premium, structured answer by synthesizing documents from multiple queries.
        This version is optimized for the controlled evaluation experiment.
        """
        if not query_results:
            return "Error: No documents were retrieved, so a combined answer cannot be generated."
    
        # --- 1. COMBINE AND DEDUPLICATE DOCUMENTS ---
        # Use a dictionary to store unique documents based on their content to avoid redundancy.
        unique_docs = {}
        for docs in query_results.values():
            for doc in docs:
                if doc.document.page_content not in unique_docs:
                    unique_docs[doc.document.page_content] = doc
        
        final_docs = list(unique_docs.values())
        
        if not final_docs:
            return "Error: All retrieval queries resulted in empty document sets."
    
        # --- 2. CORRECT CONTEXT FORMATTING ---
        # Use the same premium formatting as generate_premium_answer.
        context_pieces = []
        for i, doc in enumerate(final_docs[:5]): # Use a hardcoded limit of 5 total unique docs
            source_link = doc.document.metadata.get('source', 'Unknown Source')
            context_pieces.append(
                f"--- START: Document {i+1} ({source_link}) ---\n"
                f"{doc.document.page_content}\n"
                f"--- END: Document {i+1} ---"
            )
        context = "\n\n".join(context_pieces)
        
        # --- 3. DEFINE THE PREMIUM PROMPT ---
        # This is the exact same prompt used in generate_premium_answer for consistency.
        premium_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert scientific research assistant. Your task is to answer the user's question with accuracy and clarity, using the provided research paper excerpts.
    
            **CONTEXT FROM RESEARCH PAPERS:**
            {context}
    
            **USER'S QUESTION:** {question}
    
            **INSTRUCTIONS:**
            1.  **Direct Answer:** Write a comprehensive answer based ONLY on the provided "CONTEXT". Synthesize information across multiple documents. If the context contains conflicting information, highlight the disagreement. If the information is not in the context, state that clearly.
            2.  **In-Text Citations:** For every claim you make, you MUST cite the source using its identifier (e.g., "(Source: Document 1)").
            3.  **Broader Context:** After the direct answer, add a section titled "Broader Context". Here, provide additional perspective based on your general knowledge.
            4.  **Sources List:** Finally, create a section titled "Sources Used". List all the documents you cited by their full source link.
    
            **RESPONSE FORMAT:**
    
            **Direct Answer (from sources):**
            [Your synthesized answer with in-text citations.]
    
            **Broader Context:**
            [Your supplementary explanation.]
    
            **Sources Used:**
            - Document 1: [Full source link from the context]
            - Document 2: [Full source link from the context]
            """
        )
        
        # --- 4. GENERATE THE ANSWER ---
        try:
            chain = premium_prompt | self.llm
            answer = chain.invoke({"question": question, "context": context})
            
            # Save to memory (good practice, though cleared in the loop)
            self.memory.chat_memory.add_messages([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            return answer
        except Exception as e:
            logger.error(f"Error generating combined answer: {e}")
            return f"Error during final combined answer generation: {e}"

    @traceable(name="doc_relevance")
    def evaluate_document_relevance(self, question: str, docs: List[RetrievedDocument]) -> Tuple[bool, float]:
        """Evaluates whether documents are sufficient to answer the question.
        
        Args:
            question: User's question
            docs: List of retrieved documents
            
        Returns:
            Tuple of (is_sufficient, confidence_score)
        """
        if not docs:
            return False, 0.0
            
        # If all docs have very low scores, they're likely not relevant
        avg_score = sum(doc.score for doc in docs) / len(docs)
        
        # Create a prompt to evaluate document relevance
        prompt = PromptTemplate.from_template("""
        You are evaluating whether a set of retrieved documents contains information that can help 
        to answer a user's question.
        
        User Question: {question}
        
        Retrieved Information:
        {doc_summaries}
        
        Please analyze whether the retrieved documents is somewhere related to the question and can in a way contribute or provide some sources and information.
        Respond with a JSON object:
        {{
          "is_relevant": true/false,
          "confidence": <number between 0-1>,
          "explanation": "<brief explanation>"
        }}
        
        Only respond with the JSON object.
        """)
        
        # Create short summaries of each document
        doc_summaries = []
        for i, doc in enumerate(docs[:3]):  # Use top 3 docs for evaluation
            summary = doc.document.page_content[:200] + "..." if len(doc.document.page_content) > 200 else doc.document.page_content
            doc_summaries.append(f"Document {i+1} [Score: {doc.score:.2f}]: {summary}")
        
        doc_summaries_text = "\n\n".join(doc_summaries)
        
        try:
            # Parse the LLM response as JSON
            response = self.llm.invoke(
                prompt.format(question=question, doc_summaries=doc_summaries_text)
            )
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', response.replace('\n', ' '), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                is_sufficient = result.get("is_relevant", False)
                confidence = result.get("confidence", 0.0)
                
                logger.info(f"Document relevance: Sufficient={is_sufficient}, Confidence={confidence}")
                logger.info(f"Explanation: {result.get('explanation', 'No explanation provided')}")
                
                return is_sufficient, confidence
            else:
                # If JSON parsing fails, use heuristics based on average score
                return avg_score > self.relevance_threshold, avg_score
                
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {e}")
            # Fall back to using avg score
            return avg_score > self.relevance_threshold, avg_score

    def generate_llm_answer(self, question: str) -> str:
        """Generates an answer directly from the LLM when documents aren't sufficient.
        
        Args:
            question: User's question
            
        Returns:
            Generated answer
        """
        # Get chat history from memory if available
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        # Create chat history string for context if available
        chat_context = ""
        if chat_history:
            chat_context = "Previous conversation:\n"
            for message in chat_history[-3:]:  # Use last 3 messages for context
                if hasattr(message, "content"):
                    role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                    chat_context += f"{role}: {message.content}\n"
        
        prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""You are a helpful scientific research assistant. 
            
            The user has asked a question, but we couldn't find relevant documents in our scientific database.
            Since this appears to be a question you can answer from your general knowledge, please provide
            a helpful response.
            
            Guidelines:
            - Be clear that you're answering from general knowledge rather than specific papers
            - Provide accurate, factual information
            - If the question is about a very specialized scientific topic that requires references to papers,
              indicate that you lack specific citations but can provide general information
            - If it's a basic concept, provide a thorough explanation
            
            {chat_history}
            
            Question: {question}
            
            Answer:"""
        )
        
        try:
            # Generate answer
            answer = (prompt | self.llm).invoke({
                "question": question,
                "chat_history": chat_context
            })
            
            # Save the QA pair to memory
            self.memory.chat_memory.add_messages([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            return answer
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return "I'm sorry, I couldn't find relevant information in my scientific database and encountered an error trying to provide a general answer. Please try rephrasing your question."
    
    def ask_with_memory(self, question: str, num_results: int = 5) -> str:
        """Uses conversation memory to process follow-up questions.
        
        Args:
            question: User's question
            num_results: Number of results to retrieve
            
        Returns:
            Generated answer
        """
        try:
            # First, check if this is a question we can answer directly from memory
            can_direct_answer, direct_answer = self.can_answer_without_retrieval(question)
            if can_direct_answer:
                console.print("[bold green]Question can be answered directly from conversation history...[/bold green]")
                # Save the QA pair to memory manually
                self.memory.chat_memory.add_messages([
                    HumanMessage(content=question),
                    AIMessage(content=direct_answer)
                ])
                return direct_answer
# Otherwise, use the full query improvement pipeline
            
            # Step 1: Rate the query
            console.print("\n[bold cyan]Evaluating your query (with memory context)...[/bold cyan]")
            rating_result = self.rate_query(question)
            
            rating_color = "green" if rating_result['rating'] >= 4 else "yellow" if rating_result['rating'] >= 3 else "red"
            console.print(f"[bold {rating_color}]Query Rating: {rating_result['rating']}/5[/bold {rating_color}]")
            console.print(f"Explanation: {rating_result['explanation']}")
            console.print("\n" + "-"*50 + "\n")
            
            # Step 2: Suggest rewrites if the rating is less than perfect
            rewritten_queries = []
            if rating_result["rating"] < 5:
                console.print("[bold cyan]Generating improved query variations...[/bold cyan]")
                # Get chat history for context in rewrites
                chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
                rewritten_queries = self.suggest_rewrites(question, chat_history)
                console.print("\n" + "-"*50 + "\n")
            
            # Step 3: Present options to the user
            query_options = self.present_query_options(question, rewritten_queries)
            console.print("\n" + "-"*50 + "\n")
            
            # Step 4: Get user selection
            console.print("[bold cyan]Select queries to use:[/bold cyan]")
            console.print("Enter the numbers of the queries you want to use (comma-separated, e.g., '0,2,3')")
            selected_indices_input = input("> ")
            
            try:
                selected_indices = [int(idx.strip()) for idx in selected_indices_input.split(",")]
            except ValueError:
                console.print("[bold red]Invalid input. Using original query only.[/bold red]")
                selected_indices = [0]  # Default to original query
            
            # Get the selected queries
            selected_queries = []
            for idx in selected_indices:
                if idx == 0:  # Original query
                    selected_queries.append(question)
                elif 0 < idx <= len(rewritten_queries):  # Rewritten query
                    query_text = rewritten_queries[idx-1]
                    selected_queries.append(query_text)
            
            if not selected_queries:
                console.print("[bold red]No valid queries selected. Using original query.[/bold red]")
                selected_queries = [question]
                
            console.print(f"\n[bold green]Selected {len(selected_queries)} queries for retrieval.[/bold green]")
            console.print("\n" + "-"*50 + "\n")
            
            # Step 5: Retrieve documents
            console.print("[bold cyan]Retrieving relevant documents...[/bold cyan]")
            
            if len(selected_queries) > 1:
                # If multiple queries were selected, use the multi-query approach
                docs, query_docs_map = self.retrieve_documents(selected_queries, k=num_results)
                
                # Generate a combined answer
                console.print("\n[bold cyan]Generating combined answer with conversation+docs context...[/bold cyan]")
                answer = self.generate_combined_answer(question, query_docs_map)
            else:
                # If only one query was selected, use the standard approach
                try:
                    retrieved_docs = []
                    results = self.vector_db.similarity_search_with_relevance_scores(selected_queries[0], k=num_results)
                    
                    for i, (doc, score) in enumerate(results, 1):
                        retrieved_docs.append(RetrievedDocument(
                            document=doc,
                            score=score,
                            query=selected_queries[0],
                            rank=i
                        ))
                    
                    console.print(f"[bold green]Retrieved {len(retrieved_docs)} documents[/bold green]")
                    
                    # Show previews of the retrieved documents
                    console.print("\n[bold cyan]Top retrieved documents:[/bold cyan]")
                    for i, doc in enumerate(retrieved_docs[:5], 1):
                        score_color = "green" if doc.score > 0.8 else "yellow" if doc.score > 0.6 else "red"
                        console.print(f"{i}. [bold {score_color}]Score: {doc.score:.4f}[/bold {score_color}]")
                        
                        preview = doc.document.page_content[:150] + "..." if len(doc.document.page_content) > 150 else doc.document.page_content
                        console.print(f"   Preview: {preview}")
                        
                        if hasattr(doc.document, 'metadata') and doc.document.metadata:
                            console.print(f"   Source: {doc.document.metadata.get('source', 'Unknown')}")
                    
                    # Generate answer with memory context
                    console.print("\n[bold cyan]Generating answer with conversation + doc context...[/bold cyan]")
                    answer = self.generate_premium_answer(question, retrieved_docs)
                    
                except Exception as e:
                    logger.error(f"Error in retrieval: {e}")
                    return f"I'm sorry, I encountered an error while retrieving documents: {str(e)}"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in ask_with_memory: {e}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"

    def search_with_query_feedback(self, query: str, num_results: int = 5) -> str:
        """Main pipeline that processes a query and returns an answer."""
        # Step 1: Rate the query
        console.print("\n[bold cyan]Evaluating your query...[/bold cyan]")
        rating_result = self.rate_query(query)
            
        rating_color = "green" if rating_result['rating'] >= 4 else "yellow" if rating_result['rating'] >= 3 else "red"
        console.print(f"[bold {rating_color}]Query Rating: {rating_result['rating']}/5[/bold {rating_color}]")
        console.print(f"Explanation: {rating_result['explanation']}")
        console.print("\n" + "-"*50 + "\n")
            
        # Step 2: Suggest rewrites if the rating is less than perfect
        rewritten_queries = []
        if rating_result["rating"] < 5:
            console.print("[bold cyan]Generating improved query variations...[/bold cyan]")
            rewritten_queries = self.suggest_rewrites(query)
            console.print("\n" + "-"*50 + "\n")


        with tracing_v2_enabled(False):
            # Step 3: Present options to the user
            query_options = self.present_query_options(query, rewritten_queries)
            console.print("\n" + "-"*50 + "\n")
    
    
            
            # Step 4: Get user selection
            console.print("[bold cyan]Select queries to use:[/bold cyan]")
            console.print("Enter the numbers of the queries you want to use (comma-separated, e.g., '0,2,3')")
            selected_indices_input = input("> ")
            
            try:
                selected_indices = [int(idx.strip()) for idx in selected_indices_input.split(",")]
            except ValueError:
                console.print("[bold red]Invalid input. Using original query only.[/bold red]")
                selected_indices = [0]  # Default to original query
            
            # Get the selected queries
            selected_queries = []
            for idx in selected_indices:
                if idx == 0:  # Original query
                    selected_queries.append(query)
                elif 0 < idx <= len(rewritten_queries):  # Rewritten query
                    query_text = rewritten_queries[idx-1]
                    selected_queries.append(query_text)
                    
            if not selected_queries:
                console.print("[bold red]No valid queries selected. Using original query.[/bold red]")
                selected_queries = [query]
                
            console.print(f"\n[bold green]Selected {len(selected_queries)} queries for retrieval.[/bold green]")
            console.print("\n" + "-"*50 + "\n")
        
        # Step 5: Retrieve documents
        console.print("[bold cyan]Retrieving relevant documents...[/bold cyan]")
        
        # If multiple queries were selected, use the multi-query approach
        print("LOG123: multiple queries this is working")
        docs, query_docs_map = self.retrieve_documents(selected_queries, k=num_results)
        # Generate a combined answer
        console.print("\n[bold cyan]Generating combined answer with documents context...[/bold cyan]")
        answer = self.generate_combined_answer(query, query_docs_map)
        return answer
    
    def process_query(self, query: str, use_memory: bool = False, num_results: int = 5) -> str:
        """Main entry point that decides whether to use memory or the full pipeline.
        
        Args:
            query: User's question
            use_memory: Whether to explicitly use memory mode
            num_results: Number of results to retrieve
            
        Returns:
            Generated answer
        """
        # Check if this might be a follow-up question that should use memory
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        should_use_memory = use_memory and (chat_history and self._is_likely_followup(query))
        print("should_use_memory",should_use_memory)
        if should_use_memory:
            console.print("[bold cyan]Using conversation memory to process this query...[/bold cyan]")
            return self.ask_with_memory(query, num_results)
        else:
            console.print("[bold cyan]Using full query improvement pipeline...[/bold cyan]")
            return self.search_with_query_feedback(query, num_results)
    
    def _is_likely_followup(self, query: str) -> bool:
        """Heuristically determines if a query is likely a follow-up question.
        
        Args:
            query: User's question
            
        Returns:
            Boolean indicating if this is likely a follow-up
        """
        # Look for pronouns, references, and questions that seem to refer to previous context
        followup_indicators = [
            # Pronouns
            "it", "this", "that", "they", "them", "these", "those",
            # Reference terms
            "previous", "earlier", "above", "mentioned", "last", "former",
            # Follow-up phrases
            "what about", "how about", "tell me more", "elaborate", "clarify", "explain further",
            "can you expand", "additionally", "furthermore", "also", "related to that",
            # Short questions that likely need context
            "why", "how does", "can you explain", "what does", "what is", "who is"
        ]
        
        query_lower = query.lower()
        
        # Check for any of the indicators
        if any(indicator in query_lower for indicator in followup_indicators):
            return True
            
        # # Check for very short queries (likely follow-ups)
        # if len(query.split()) < 4:
        #     return True
            
        return False
    
    def clear_memory(self) -> None:
        """Clears the conversation memory."""
        self.memory.clear()
        console.print("[bold green]Conversation memory cleared.[/bold green]")
        
    def get_chat_history(self) -> str:
        """Returns the formatted chat history.
        
        Returns:
            String representation of chat history
        """
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])
        
        if not chat_history:
            return "No conversation history."
            
        history_str = ""
        for i, message in enumerate(chat_history):
            if hasattr(message, "content"):
                role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                history_str += f"{role}: {message.content}\n\n"
                
        return history_str

    def Load_query(self, query):
        """
        Load research papers from arXiv based on the given query and store them in the temporary database.
        Returns the top 3 papers information.
        """
        console.print(f"[bold cyan]Searching arXiv for papers related to:[/bold cyan] {query}")
        
        try:
            # Use the working ArxivLoader from langchain_community
            loader = ArxivLoader(
                query=query,
                load_max_docs=3,  # Get top 3 papers
                load_all_available_meta=True
            )
            
            documents = loader.load()
            console.print(f"[green]Found {len(documents)} relevant papers[/green]")
            
            # Download PDFs for more complete content
            enhanced_docs = self.download_pdf_content(documents)
            
            # Store documents in temporary database
            paper_ids = self.StoreInDB(enhanced_docs)
            
            # Display basic information about the loaded papers
            self.display_paper_summaries(enhanced_docs)
            
            return paper_ids
        
        except Exception as e:
            console.print(f"[bold red]Error loading papers: {str(e)}[/bold red]")
            # # Fall back to simulated papers if needed
            # if self.allow_fallback:
            #     console.print("[yellow]Using fallback data for demonstration purposes.[/yellow]")
            #     return self.create_fallback_papers(query)
            return []
    
    def download_pdf_content(self, documents):
        """
        Enhance documents by downloading the full PDF content when available.
        """
        enhanced_docs = []
        
        for doc in documents:
            try:
                # Get metadata
                metadata = doc.metadata
                title = metadata.get("Title", "Untitled")
                console.print(f"[cyan]Processing paper: {title}[/cyan]")
                
                # Try to find arXiv ID from page content
                arxiv_id = None
                match = re.search(r'arXiv:(\d{4}\.\d{5})', doc.page_content)
                
                if match:
                    arxiv_id = match.group(1)
                else:
                    # Try to extract from Entry ID
                    entry_id = metadata.get("Entry ID", "")
                    id_match = re.search(r'abs/(\d{4}\.\d{5})', entry_id)
                    if id_match:
                        arxiv_id = id_match.group(1)
                
                if arxiv_id:
                    # Generate PDF URL
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    console.print(f"[cyan]Found PDF URL: {pdf_url}[/cyan]")
                    
                    # Create temp directory if it doesn't exist
                    temp_dir = "./temp_pdfs"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Download PDF
                    pdf_filename = f"{temp_dir}/{arxiv_id}.pdf"
                    
                    # Check if file already exists
                    if not os.path.exists(pdf_filename):
                        console.print(f"[cyan]Downloading PDF...[/cyan]")
                        response = requests.get(pdf_url, timeout=30)
                        with open(pdf_filename, 'wb') as f:
                            f.write(response.content)
                    
                    # Load PDF content
                    console.print(f"[cyan]Extracting content from PDF...[/cyan]")
                    pdf_loader = PyPDFLoader(pdf_filename)
                    pdf_docs = pdf_loader.load()
                    
                    # Create enhanced document with PDF content
                    full_content = doc.page_content + "\n\n" + "\n\n".join([pdf_doc.page_content for pdf_doc in pdf_docs])
                    
                    # Update metadata with PDF info
                    updated_metadata = metadata.copy()
                    updated_metadata["pdf_url"] = pdf_url
                    updated_metadata["pdf_path"] = pdf_filename
                    updated_metadata["pdf_pages"] = len(pdf_docs)
                    
                    enhanced_doc = Document(page_content=full_content, metadata=updated_metadata)
                    enhanced_docs.append(enhanced_doc)
                    
                else:
                    # If PDF can't be found, use original document
                    console.print(f"[yellow]Could not extract arXiv ID for {title}. Using abstract only.[/yellow]")
                    enhanced_docs.append(doc)
            
            except Exception as e:
                console.print(f"[yellow]Error enhancing document: {str(e)}. Using abstract only.[/yellow]")
                enhanced_docs.append(doc)
        
        return enhanced_docs
    
    def StoreInDB(self, documents):
        """
        Store loaded documents in the temporary vector database.
        Returns list of document IDs.
        """
        paper_ids = []
        
        console.print("[bold cyan]Storing papers in temporary database...[/bold cyan]")
        
        try:
            # Split documents into chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            
            for doc in documents:
                # Create document chunks
                chunks = text_splitter.split_documents([doc])
                
                # Create unique ID for the paper
                paper_id = f"paper_{uuid.uuid4().hex[:8]}"
                paper_ids.append(paper_id)
                
                # Add metadata to each chunk and sanitize it
                for i, chunk in enumerate(chunks):
                    # Create clean metadata dictionary
                    clean_metadata = {
                        "paper_id": paper_id,
                        "chunk_id": i,
                        "title": doc.metadata.get("Title", "Untitled"),
                        "authors": doc.metadata.get("Authors", "Unknown"),
                        "published": doc.metadata.get("Published", "Unknown")
                    }
                    
                    # Ensure all values are valid types (str, int, float, bool)
                    for key, value in clean_metadata.items():
                        if value is None:
                            clean_metadata[key] = "None"  # Convert None to string
                        elif not isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = str(value)  # Convert complex types to string
                    
                    # Replace existing metadata with clean version
                    chunk.metadata = clean_metadata
                
                # Add to vector database
                self.vector_db.add_documents(chunks)
            
            # Persist the temporary database
            self.vector_db.persist()
            console.print(f"[green]Successfully stored {len(documents)} papers in database[/green]")
            # Fixed typo from 'onsole' to 'console' and removed debug line
            return paper_ids
        
        except Exception as e:
            console.print(f"[bold red]Error storing documents: {str(e)}[/bold red]")
            return []
    
    def display_paper_summaries(self, documents):
        """Display a summary of each loaded paper."""
        console.print("\n[bold cyan]Loaded Papers:[/bold cyan]")
        
        for i, doc in enumerate(documents):
            title = doc.metadata.get("Title", "Untitled")
            authors = doc.metadata.get("Authors", "Unknown")
            published = doc.metadata.get("Published", "Unknown")
            
            console.print(f"\n[bold]{i+1}. {title}[/bold]")
            console.print(f"   Authors: {authors}")
            console.print(f"   Published: {published}")
            
            # Generate a brief summary of the paper
            summary = self.generate_paper_summary(doc)
            console.print(f"   Summary: {summary}")
    
    def generate_paper_summary(self, document):
        """Generate a concise summary of a paper using the LLM."""
        try:
            prompt = f"""
            Generate a concise 5-sentence summary of the following research paper:
            
            Title: {document.metadata.get('Title', 'Untitled')}
            Abstract: {document.page_content[:500]}...
            
            Summary:
            """
            
            response = self.llm.predict(prompt)
            return response.strip()
        except:
            return "Summary not available."
    
    def query_temp_database(self, query, paper_ids=None, k=5):
        """
        Query the temporary database with the given query.
        
        Args:
            query: The query string
            paper_ids: Optional list of paper IDs to restrict the search to
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        console.print(f"[bold cyan]Searching temporary database for:[/bold cyan] {query}")
        
        try:
            # Create filter for specific papers if provided
            filter_dict = None
            if paper_ids:
                filter_dict = {"paper_id": {"$in": paper_ids}}
            
            # Query the vector database
            search_results = self.vector_db.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            console.print(f"[green]Found {len(search_results)} relevant sections[/green]")
            return search_results
        
        except Exception as e:
            console.print(f"[bold red]Error querying database: {str(e)}[/bold red]")
            return []
    
    def answer_research_question(self, query, context_docs=None):
        """
        Answer a research question using context from the temporary database.
        
        Args:
            query: The research question
            context_docs: Optional pre-retrieved context documents
            
        Returns:
            Detailed answer to the research question
        """
        # If context not provided, retrieve it
        if not context_docs:
            context_docs = self.query_temp_database(query, k=5)
        
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(context_docs)])
        
        # Prepare sources
        sources = []
        for doc in context_docs:
            paper_title = doc.metadata.get("title", "Unknown Title")
            authors = doc.metadata.get("authors", "Unknown Authors")
            source = f"- {paper_title} by {authors}"
            if source not in sources:
                sources.append(source)
        
        # Generate the answer
        prompt = f"""
        You are a helpful research assistant. Answer the following question based on the provided research paper extracts.
        
        Question: {query}
        
        Here are relevant extracts from research papers:
        {context_text}
        
        Provide a comprehensive answer, citing specific information from the papers.
        Make sure to organize your response clearly and highlight key insights.
        """
        
        try:
            answer = self.llm.predict(prompt)
            
            # Add sources at the end
            final_answer = f"{answer}\n\nSources:\n" + "\n".join(sources)
            return final_answer
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"




# def generate_premium_answer(
#     self, 
#     question: str, 
#     retrieved_input: Union[List[RetrievedDocument], Dict[str, List[RetrievedDocument]]]
# ) -> str:
#     """
#     Generates a single, high-quality, structured answer that can handle input from
#     either a single query (list of docs) or multiple queries (dict of docs).
#     """
    
#     final_docs = []
#     # --- 1. UNIFY THE INPUT ---
#     # Check if the input is a dictionary (from multiple queries)
#     if isinstance(retrieved_input, dict):
#         # Flatten the dictionary and deduplicate documents
#         unique_docs = {}
#         for docs in retrieved_input.values():
#             for doc in docs:
#                 if doc.document.page_content not in unique_docs:
#                     unique_docs[doc.document.page_content] = doc
#         final_docs = list(unique_docs.values())
#     # Check if the input is a list (from a single query)
#     elif isinstance(retrieved_input, list):
#         final_docs = retrieved_input
    
#     if not final_docs:
#         return "Error: No valid documents were provided to the generation step."

#     # --- 2. FORMAT CONTEXT (Same as before) ---
#     context_pieces = []
#     for i, doc in enumerate(final_docs[:5]): # Use top 5 unique docs
#         source_link = doc.document.metadata.get('source', 'Unknown Source')
#         context_pieces.append(
#             f"--- START: Document {i+1} ({source_link}) ---\n"
#             f"{doc.document.page_content}\n"
#             f"--- END: Document {i+1} ---"
#         )
#     context = "\n\n".join(context_pieces)
    
#     # --- 3. DEFINE PREMIUM PROMPT (Same as before) ---
#     premium_prompt = PromptTemplate(
#         input_variables=["question", "context"],
#         template="""You are an expert scientific research assistant... (the full premium prompt)"""
#     )
    
#     # --- 4. GENERATE THE ANSWER ---
#     try:
#         chain = premium_prompt | self.llm
#         answer = chain.invoke({"question": question, "context": context})
        
#         # Save to memory
#         self.memory.chat_memory.add_messages([
#             HumanMessage(content=question),
#             AIMessage(content=answer)
#         ])
#         return answer
#     except Exception as e:
#         logger.error(f"Error generating premium answer: {e}")
#         return f"Error during final answer generation: {e}"