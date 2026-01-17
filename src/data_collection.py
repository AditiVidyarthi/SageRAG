"""
Functions for collecting and processing research papers from arXiv.
"""

import os
import sys
import logging
import arxiv
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from .utils import ensure_dir, token_length_function, get_tokenizer, console

# Set up logging
logger = logging.getLogger(__name__)

DEFAULT_TOPICS = {
    "artificial_intelligence": "artificial intelligence OR AI OR intelligent systems OR autonomous agents",
    "machine_learning": "machine learning OR supervised learning OR unsupervised learning OR reinforcement learning",
    "natural_language_processing": "natural language processing OR NLP OR computational linguistics OR language models",
    "computer_vision": "computer vision OR image recognition OR object detection OR pattern recognition",
}

class ArxivCollector:
    """Class to download and process papers from arXiv."""
    
    def __init__(
        self, 
        base_dir: str = "./arxiv_data", 
        db_dir: str = "./arxiv_vector1_db",
        tokenizer_model: str = "NousResearch/Llama-2-7b-hf"
    ):
        """
        Initialize the ArxivCollector.
        
        Args:
            base_dir: Directory to store downloaded PDFs
            db_dir: Directory to store the vector database
            tokenizer_model: Model name for the tokenizer
        """
        self.base_dir = ensure_dir(base_dir)
        self.db_dir = db_dir
        self.tokenizer = get_tokenizer(tokenizer_model)
        
    def download_arxiv_papers(self, query: str, max_results: int = 20, save_dir: str = None) -> List[str]:
        """
        Download papers from arXiv based on a query.
        
        Args:
            query: Search query for arXiv
            max_results: Maximum number of papers to download
            save_dir: Directory to save the papers (defaults to a subdirectory of base_dir)
            
        Returns:
            List of paths to the downloaded PDF files
        """
        if save_dir is None:
            # Create a directory name from the query
            dir_name = query.replace(" ", "_").replace(":", "").replace("/", "_")[:30]
            save_dir = os.path.join(self.base_dir, dir_name)
            
        ensure_dir(save_dir)
        console.print(f"Searching arXiv for: {query}")
        
        # Set up the search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        pdf_paths = []
        for result in search.results():
            paper_id = result.get_short_id()
            title = result.title.replace(" ", "_").replace("/", "_")[:50]
            filename = f"{paper_id}_{title}.pdf"
            filepath = os.path.join(save_dir, filename)

            if not os.path.exists(filepath):
                try:
                    console.print(f"Downloading: {title}")
                    result.download_pdf(dirpath=save_dir, filename=filename)
                except Exception as e:
                    logger.error(f"Failed to download {title}: {e}")
                    continue

            pdf_paths.append(filepath)
            
        console.print(f"Downloaded {len(pdf_paths)} papers to {save_dir}")
        return pdf_paths
        
    def process_topic(self, topic_query: str, topic_name: str, max_results: int = 20) -> List[Document]:
        """
        Download and process papers for a specific topic.
        
        Args:
            topic_query: Query string for the topic
            topic_name: Name of the topic (used for directory naming)
            max_results: Maximum number of papers to download
            
        Returns:
            List of processed documents
        """
        console.print(f"Processing Topic: {topic_name}")
        topic_dir = os.path.join(self.base_dir, topic_name)
        pdf_paths = self.download_arxiv_papers(topic_query, max_results, topic_dir)

        all_docs = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["sub_domain"] = topic_name
                    # Add arxiv ID to metadata
                    filename = os.path.basename(path)
                    arxiv_id = filename.split("_")[0]
                    doc.metadata["source"] = f"https://arxiv.org/pdf/{arxiv_id}"
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                
        return all_docs
        
    def split_documents(self, docs: List[Document], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Document]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            docs: List of documents to split
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of split document chunks
        """
        console.print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: token_length_function(text, self.tokenizer),
        )
        return splitter.split_documents(docs)
        
    def load_or_create_vector_db(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> tuple:
        """
        Load an existing vector database or create a new one.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            
        Returns:
            Tuple of (vector_db, embedding_model)
        """
        console.print("Loading or creating vector DB...")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            console.print("Existing DB found. Loading...")
            vector_db = Chroma(persist_directory=self.db_dir, embedding_function=embedding_model)
        else:
            console.print("No DB found. Will create a new one.")
            vector_db = None
            
        return vector_db, embedding_model
        
    def add_chunks_to_db(self, chunks: List[Document], vector_db, embedding_model) -> Chroma:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of document chunks to add
            vector_db: Existing vector database or None
            embedding_model: Embedding model to use
            
        Returns:
            Updated vector database
        """
        if not chunks:
            console.print("No chunks to add to DB. Skipping.")
            return vector_db

        if vector_db is None:
            console.print("Creating new DB from chunks...")
            vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embedding_model, 
                persist_directory=self.db_dir
            )
        else:
            console.print(f"Adding {len(chunks)} new chunks to existing DB...")
            vector_db.add_documents(chunks)
            
        vector_db.persist()
        return vector_db
        
    def process_multiple_topics(self, topics: Dict[str, str] = None, max_results_per_topic: int = 20) -> Chroma:
        """
        Process multiple topics and add them to the vector database.
        
        Args:
            topics: Dictionary mapping topic names to queries
            max_results_per_topic: Maximum papers to download per topic
            
        Returns:
            Vector database with all documents
        """
        if topics is None:
            topics = DEFAULT_TOPICS
            
        vector_db, embedding_model = self.load_or_create_vector_db()
        
        for topic_name, topic_query in topics.items():
            try:
                console.print(f"\nProcessing topic: {topic_name}")
                docs = self.process_topic(topic_query, topic_name, max_results_per_topic)
                chunks = self.split_documents(docs)
                vector_db = self.add_chunks_to_db(chunks, vector_db, embedding_model)
            except Exception as e:
                logger.error(f"Error processing topic {topic_name}: {e}")
                
        return vector_db
        
    def load_and_process_pdfs(
        self,
        main_folder: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> Chroma:
        """
        Load all PDFs from a folder and its subdirectories and process them.
        
        Args:
            main_folder: Folder containing PDFs (defaults to self.base_dir)
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model_name: Name of the embedding model to use
            
        Returns:
            Vector database with all documents
        """
        if main_folder is None:
            main_folder = self.base_dir
            
        console.print(f"Loading all PDFs from {main_folder} and subdirectories...")
        all_docs = []

        for subdir, dirs, files in os.walk(main_folder):
            for file in files:
                if file.endswith(".pdf"):
                    path = os.path.join(subdir, file)
                    try:
                        loader = PyPDFLoader(path)
                        docs = loader.load()
                        
                        filename = os.path.basename(path)
                        arxiv_id = filename.replace(".pdf", "").split("_")[0]
                        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
                        subdomain = os.path.basename(subdir)

                        for doc in docs:
                            doc.metadata["source"] = arxiv_url
                            doc.metadata["subdomain"] = subdomain

                        all_docs.extend(docs)
                    except Exception as e:
                        logger.error(f"Failed to load {path}: {e}")

        console.print(f"Found {len(all_docs)} document pages")
        
        # Split documents
        chunks = self.split_documents(all_docs, chunk_size, chunk_overlap)
        console.print(f"Split into {len(chunks)} chunks")
        
        # Create or update vector DB
        vector_db, embedding_model = self.load_or_create_vector_db(embedding_model_name)
        vector_db = self.add_chunks_to_db(chunks, vector_db, embedding_model)
        
        return vector_db