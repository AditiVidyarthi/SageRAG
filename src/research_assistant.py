"""
Research Assistant implementation using RAG.
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
from sentence_transformers import SentenceTransformer, util

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, BaseOutputParser
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.vectorstores import VectorStore
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.schema.runnable import Runnable
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

from .utils import console, logger

# Custom output parsers
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
        llm: BaseLLM, 
        vector_db: VectorStore, 
        embeddings_model: Optional[SentenceTransformer] = None,
        relevance_threshold: float = 0.6,
        max_docs_per_query: int = 3
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
    
    def suggest_rewrites(self, query: str, chat_history: Optional[List] = None) -> List[str]:
        """Returns rephrased versions of the query optimized for retrieval.
        
        Args:
            query: Original query to rewrite
            chat_history: Optional conversation history for context
            
        Returns:
            List of rewritten queries
        """
        history_context = ""
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
            
            Rephrase the question in 5 different ways to improve retrieval from a scientific paper database. 
            Focus on:
            1. Using technical terminology for better vector matching
            2. Breaking down complex queries into clearer formulations
            3. Adding relevant synonyms or related concepts
            4. Varying syntax while preserving semantic meaning
            5. Including key entities and relationships from the original query
            
            Number each version starting with 1.
            
            Question: {question}"""
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
            return 0.7  # Default reasonable value
    
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
            scored_queries.append((f"Rewrite {i}: {query}", score))
            
        # Display options in a nice format
        console.print("\n[bold cyan]Available search queries:[/bold cyan]")
        for i, (query, score) in enumerate(scored_queries):
            confidence_color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
            console.print(f"[bold]{i}.[/bold] {query}")
            console.print(f"   [bold {confidence_color}]Confidence: {score:.2f}[/bold {confidence_color}]")
            
        return scored_queries
    
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
                    console.print(f"Content Preview: {preview}")
                    
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
            return False, None  # Default to retrieval if there's an error
    
    def generate_final_answer(
        self, 
        question: str, 
        docs: List[RetrievedDocument], 
        max_docs: int = 5
    ) -> str:
        """Generates the final answer from retrieved documents with LLM fallback.
        
        Args:
            question: User's question
            docs: List of retrieved documents
            max_docs: Maximum number of documents to include
            
        Returns:
            Generated answer
        """
        # First, evaluate if the documents are sufficient
        is_sufficient, confidence = self.evaluate_document_relevance(question, docs)
        
        # If documents are insufficient, use LLM fallback
        if not is_sufficient or confidence < 0.4:  # Adjust threshold as needed
            console.print("[yellow]Retrieved documents insufficient. Using LLM fallback...[/yellow]")
            return self.generate_llm_answer(question)
        
        # Otherwise, continue with RAG as before
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
            input_variables=["question", "context", "chat_history"],
            template="""You are a helpful scientific research assistant analyzing scientific papers. 
            Use the following retrieved context chunks to answer the user's question thoroughly.
            
            Guidelines:
            - Focus on providing accurate information from the papers
            - Synthesize information across documents when appropriate
            - Cite the sources of information in your answer (e.g., "According to Document 1...")
            - If the information isn't in the context, indicate what's missing rather than making up information
            - If the context contains conflicting information, highlight the disagreement and possible reasons
            - Maintain scientific accuracy above all else
            
            {chat_history}
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )
        
        # Format document content with metadata
        context_pieces = []
        for i, doc in enumerate(docs[:max_docs]):
            chunk = f"Document {i+1} [Relevance: {doc.score:.2f}]:\n{doc.document.page_content}"
            
            # Add metadata if available
            if hasattr(doc.document, 'metadata') and doc.document.metadata:
                source = doc.document.metadata.get('source', 'Unknown')
                chunk += f"\nSource: {source}"
                
            context_pieces.append(chunk)
            
        context = "\n\n---\n\n".join(context_pieces)
        
        try:
            # Generate answer
            answer = (prompt | self.llm).invoke({
                "question": question, 
                "context": context,
                "chat_history": chat_context
            })
            
            # Save the QA pair to memory
            self.memory.chat_memory.add_messages([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I encountered an error while generating your answer. Please try rephrasing your question."
    
    def generate_combined_answer(
        self, 
        question: str, 
        query_results: Dict[str, List[RetrievedDocument]]
    ) -> str:
        """Generates a combined answer from multiple query results with LLM fallback.
        
        Args:
            question: User's question
            query_results: Dictionary mapping queries to retrieved documents
            
        Returns:
            Generated answer
        """
        # Combine all relevant documents from different queries
        all_docs = []
        for query, docs in query_results.items():
            all_docs.extend(docs[:self.max_docs_per_query])
        
        # Evaluate if the documents are sufficient
        is_sufficient, confidence = self.evaluate_document_relevance(question, all_docs)
        
        # If documents are insufficient, use LLM fallback
        if not is_sufficient or confidence < 0.4:  # Adjust threshold as needed
            console.print("[yellow]Retrieved documents insufficient. Using LLM fallback...[/yellow]")
            return self.generate_llm_answer(question)
        
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
        
        # Combine all context sections
        all_context_sections = []
        
        for query, docs in query_results.items():
            # Use only the top N documents for each query
            docs_for_query = docs[:self.max_docs_per_query]
            if docs_for_query:
                all_context_sections.append(f"Results for query: '{query}'")
                for i, doc in enumerate(docs_for_query, 1):
                    section = f"Document {i} [Relevance: {doc.score:.2f}]:\n{doc.document.page_content}"
                    
                    # Add metadata if available
                    if hasattr(doc.document, 'metadata') and doc.document.metadata:
                        source = doc.document.metadata.get('source', 'Unknown')
                        section += f"\nSource: {source}"
                        
                    all_context_sections.append(section)
        
        # Join all context sections
        combined_context = "\n\n---\n\n".join(all_context_sections)
        
        # Create a prompt that emphasizes synthesizing information across different query results
        prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""You are a helpful scientific research assistant analyzing scientific papers. 
            The user's question has been reformulated in several ways, and each formulation returned different documents.
            
            Guidelines:
            - Synthesize information across ALL retrieved documents to provide a comprehensive answer
            - Compare and contrast findings from different sources
            - Highlight the most relevant information from each source
            - Cite the specific documents you're referencing (e.g., "According to the paper in Document 3...")
            - If information conflicts across sources, explain the different perspectives
            - If the information isn't in the context, indicate what's missing
            - Maintain scientific accuracy above all else
            
            {chat_history}
            
            Context from multiple query formulations:
            {context}
            
            Original Question: {question}
            
            Answer:"""
        )
        
        try:
            # Generate answer
            answer = (prompt | self.llm).invoke({
                "question": question, 
                "context": combined_context,
                "chat_history": chat_context
            })
            
            # Save the QA pair to memory
            self.memory.chat_memory.add_messages([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            return answer
        except Exception as e:
            logger.error(f"Error generating combined answer: {e}")
            return "I'm sorry, I encountered an error while generating your answer. Please try rephrasing your question."
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
        You are evaluating whether a set of retrieved documents contains information 
        to answer a user's question.
        
        User Question: {question}
        
        Retrieved Information:
        {doc_summaries}
        
        Please analyze whether the retrieved documents contain enough information to answer the question.
        Respond with a JSON object:
        {{
          "is_sufficient": true/false,
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
                is_sufficient = result.get("is_sufficient", False)
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
                console.print("\n[bold cyan]Generating combined answer with conversation context...[/bold cyan]")
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
                    console.print("\n[bold cyan]Generating answer with conversation context...[/bold cyan]")
                    answer = self.generate_final_answer(question, retrieved_docs)
                    
                except Exception as e:
                    logger.error(f"Error in retrieval: {e}")
                    return f"I'm sorry, I encountered an error while retrieving documents: {str(e)}"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in ask_with_memory: {e}")
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
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
        
        should_use_memory = use_memory or (chat_history and self._is_likely_followup(query))
        
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
            
        # Check for very short queries (likely follow-ups)
        if len(query.split()) < 4:
            return True
            
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
