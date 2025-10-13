#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Text Processor for Greenpeace Dataset
==========================================

This script processes text files from the dataset directory, chunks them into 300-character pieces,
creates embeddings using a local SentenceTransformer model, and indexes them in ChromaDB using 
metadata from greenpeace.csv.

Features:
- Uses local embeddings (no API calls required)
- 300-character text chunking
- ChromaDB vector storage
- Metadata integration from CSV
- Validation of file existence

Usage:
    python rag_text_processor.py

Requirements:
    - Conda environment 'llm' activated
    - greenpeace/greenpeace.csv metadata file
    - dataset/ directory with .txt files
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
from dotenv import load_dotenv

# LangChain and ChromaDB imports
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGTextProcessor:
    """Processes text files for RAG indexing with ChromaDB."""
    
    def __init__(self, dataset_dir: str = "dataset", 
                 metadata_csv: str = "greenpeace/greenpeace.csv",
                 chroma_db_dir: str = "chroma_db_rag",
                 chunk_size: int = 300):
        """
        Initialize the RAG text processor.
        
        Args:
            dataset_dir: Directory containing .txt files
            metadata_csv: Path to CSV file with metadata
            chroma_db_dir: Directory to store ChromaDB
            chunk_size: Size of text chunks in characters
        """
        self.dataset_dir = Path(dataset_dir)
        self.metadata_csv = Path(metadata_csv)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.chunk_size = chunk_size
        
        # Load environment variables (no longer needed for local embeddings, but kept for compatibility)
        load_dotenv()
        
        # Initialize local embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Lightweight and fast model
        )
        
        # Initialize ChromaDB
        self.vector_store = Chroma(
            collection_name="greenpeace_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_db_dir)
        )
        
        # Load metadata
        self.metadata_df = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata CSV file."""
        try:
            df = pd.read_csv(self.metadata_csv)
            logger.info(f"Loaded metadata for {len(df)} documents")
            logger.info(f"Metadata columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading metadata CSV: {e}")
            raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _read_text_file(self, file_path: Path) -> Optional[str]:
        """
        Read text file content safely.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _get_file_metadata(self, filename: str) -> Optional[Dict]:
        """
        Get metadata for a file from the CSV.
        
        Args:
            filename: Name of the file (without .txt extension)
            
        Returns:
            Dictionary with metadata or None if not found
        """
        # Remove .txt extension for matching
        file_id = filename.replace('.txt', '')
        
        # Find matching row in metadata
        matching_rows = self.metadata_df[self.metadata_df['id'] == file_id]
        
        if matching_rows.empty:
            logger.warning(f"No metadata found for file: {filename}")
            return None
        
        # Return first matching row as dictionary
        metadata = matching_rows.iloc[0].to_dict()
        return metadata
    
    def _create_documents(self, file_path: Path, text_chunks: List[str], 
                         metadata: Dict) -> List[Document]:
        """
        Create LangChain Document objects from text chunks.
        
        Args:
            file_path: Path to original file
            text_chunks: List of text chunks
            metadata: File metadata from CSV
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for i, chunk in enumerate(text_chunks):
            # Create metadata for this chunk
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'source_file': str(file_path.name),
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(text_chunks)
            })
            
            # Create document
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_file(self, file_path: Path) -> Tuple[int, int]:
        """
        Process a single text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (chunks_created, chunks_indexed)
        """
        logger.info(f"Processing file: {file_path.name}")
        
        # Read file content
        content = self._read_text_file(file_path)
        if not content:
            return 0, 0
        
        # Get metadata
        metadata = self._get_file_metadata(file_path.name)
        
        # Create chunks
        chunks = self._chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        # Create documents
        documents = self._create_documents(file_path, chunks, metadata)
        
        # Generate UUIDs for the documents
        uuids = [str(uuid4()) for _ in documents]
        
        try:
            # Add to vector store
            self.vector_store.add_documents(documents, ids=uuids)
            logger.info(f"Successfully indexed {len(documents)} chunks from {file_path.name}")
            return len(chunks), len(documents)
        except Exception as e:
            logger.error(f"Error indexing documents from {file_path.name}: {e}")
            return len(chunks), 0
    
    def get_valid_files(self) -> List[Path]:
        """
        Get list of .txt files that exist in dataset and have metadata.
        
        Returns:
            List of valid file paths
        """
        valid_files = []
        
        # Get all .txt files in dataset directory
        txt_files = list(self.dataset_dir.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} .txt files in {self.dataset_dir}")
        
        # Filter files that have metadata
        for file_path in txt_files:
            file_id = file_path.name.replace('.txt', '')
            if file_id in self.metadata_df['id'].values:
                valid_files.append(file_path)
            else:
                logger.warning(f"Skipping {file_path.name} - no metadata found")
        
        logger.info(f"Found {len(valid_files)} files with metadata")
        return valid_files
    
    def process_all_files(self) -> Dict[str, int]:
        """
        Process all valid files in the dataset.
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting batch processing of all files...")
        
        valid_files = self.get_valid_files()
        
        stats = {
            'total_files': len(valid_files),
            'processed_files': 0,
            'total_chunks': 0,
            'indexed_chunks': 0,
            'failed_files': 0
        }
        
        for file_path in valid_files:
            try:
                chunks_created, chunks_indexed = self.process_file(file_path)
                
                stats['processed_files'] += 1
                stats['total_chunks'] += chunks_created
                stats['indexed_chunks'] += chunks_indexed
                
                if chunks_indexed == 0:
                    stats['failed_files'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                stats['failed_files'] += 1
        
        logger.info("Batch processing completed!")
        logger.info(f"Statistics: {stats}")
        
        return stats
    
    def search_similar(self, query: str, k: int = 5, 
                      category_filter: Optional[str] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            category_filter: Optional category filter
            
        Returns:
            List of similar documents
        """
        filter_dict = {}
        if category_filter:
            filter_dict["category"] = category_filter
        
        results = self.vector_store.similarity_search(
            query, 
            k=k, 
            filter=filter_dict if filter_dict else None
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get some sample documents to show categories
            sample_docs = self.vector_store.similarity_search("climate", k=10)
            categories = set()
            for doc in sample_docs:
                if 'category' in doc.metadata:
                    categories.add(doc.metadata['category'])
            
            return {
                'total_documents': count,
                'sample_categories': list(categories)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    """Main function to run the RAG text processor."""
    logger.info("=== RAG Text Processor for Greenpeace Dataset (Local Embeddings) ===")
    
    # Check if we're in the right directory
    if not os.path.exists("dataset"):
        logger.error("Dataset directory not found. Please run from the correct directory.")
        return
    
    if not os.path.exists("greenpeace/greenpeace.csv"):
        logger.error("Metadata CSV file not found. Please ensure greenpeace/greenpeace.csv exists.")
        return
    
    # Initialize processor
    try:
        processor = RAGTextProcessor()
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return
    
    # Process all files
    try:
        stats = processor.process_all_files()
        
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files found: {stats['total_files']}")
        print(f"Files processed: {stats['processed_files']}")
        print(f"Files failed: {stats['failed_files']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Chunks indexed: {stats['indexed_chunks']}")
        print(f"ChromaDB location: {processor.chroma_db_dir}")
        
        # Get collection stats
        collection_stats = processor.get_collection_stats()
        print(f"\nCollection statistics:")
        for key, value in collection_stats.items():
            print(f"  {key}: {value}")
        
        # Example search
        print("\n" + "="*50)
        print("EXAMPLE SEARCH")
        print("="*50)
        results = processor.search_similar("climate change", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"   Category: {doc.metadata.get('category', 'Unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return
    
    logger.info("RAG text processing completed successfully!")


if __name__ == "__main__":
    main()