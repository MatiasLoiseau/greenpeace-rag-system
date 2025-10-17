#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Question Answering System for Greenpeace Dataset
==================================================

This script uses the previously created ChromaDB index to answer questions
about the Greenpeace dataset using Google's Gemini model.

Features:
- Uses local embeddings for retrieval (no API cost)
- Gemini 1.5 Flash for generation (better quality, still free tier)
- Context-aware responses with source citations
- Configurable number of relevant chunks (increased to 8)

Usage:
    python rag_qa_system.py

Requirements:
    - Existing ChromaDB from rag_text_processor.py
    - .env file with GOOGLE_API_KEY
    - Conda environment 'llm' activated
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGQuestionAnswering:
    """RAG system for answering questions about Greenpeace documents."""
    
    def __init__(self, 
                 chroma_db_dir: str = "chroma_db_rag",
                 model_name: str = "llama3.2",  # Optimized Ollama model  
                 max_tokens: int = 2048,  # Increased for better responses
                 top_k: int = 8):  # More context for better accuracy
        """
        Initialize the RAG QA system.
        
        Args:
            chroma_db_dir: Directory with the ChromaDB
            model_name: Ollama model to use (llama3.2 with optimized parameters)
            max_tokens: Maximum tokens for response (increased for completeness)
            top_k: Number of relevant chunks to retrieve (8 for better context)
        """
        # Load environment variables
        load_dotenv()
        
        self.chroma_db_dir = Path(chroma_db_dir)
        self.top_k = top_k
        
        # Initialize embeddings (same as used for indexing)
        logger.info("Initializing embeddings...")
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB
        logger.info(f"Loading ChromaDB from {self.chroma_db_dir}")
        if not self.chroma_db_dir.exists():
            raise FileNotFoundError(f"ChromaDB not found at {self.chroma_db_dir}. Run rag_text_processor.py first.")
        
        self.vector_store = Chroma(
            collection_name="greenpeace_documents",
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_db_dir)
        )
        
        # Initialize Ollama model (fallback while resolving Gemini API issues)
        logger.info(f"Initializing Ollama model: {model_name}")
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=model_name,
                temperature=0.2,  # Optimized temperature
                num_predict=max_tokens  # Increased for more complete responses
            )
            logger.info(f"Ollama model {model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            logger.info("Make sure Ollama is running with: ollama serve")
            logger.info(f"And the model is available: ollama pull {model_name}")
            raise
        
        # Prompt template will be built dynamically
        pass
        
        # Create retriever with improved search parameters
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        logger.info("RAG system initialized successfully!")
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents for the prompt."""
        formatted_chunks = []
        
        for i, doc in enumerate(docs, 1):
            # Get metadata
            source = doc.metadata.get('source_file', 'Unknown')
            category = doc.metadata.get('category', 'Unknown')
            
            # Format chunk
            chunk_text = f"[Documento {i}] (Fuente: {source}, Categoría: {category})\n{doc.page_content}\n"
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)
    
    def _build_rag_prompt(self, context: str, question: str) -> str:
        """Construye el prompt para el RAG usando Gemini 1.5 Pro."""
        return f"""You are a highly knowledgeable environmental policy expert and Greenpeace specialist. 
Your task is to provide comprehensive, accurate, and well-structured answers based STRICTLY on the provided context.

CONTEXT FROM GREENPEACE DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. **Accuracy**: Base your answer ONLY on the provided context. If information is missing, explicitly state "I don't have sufficient information in the documents to answer that question."

2. **Comprehensiveness**: Provide a detailed, complete answer that addresses all aspects of the question when information is available.

3. **Structure**: Organize your response with clear sections or bullet points when appropriate.

4. **Evidence**: Include specific facts, statistics, dates, and examples from the context.

5. **Sources**: Reference specific documents or sources when citing information.

6. **Multiple Perspectives**: If the context contains different viewpoints or approaches, present them fairly.

7. **Language**: Answer in English with clear, professional language suitable for policy makers and researchers.

8. **Context Integration**: Synthesize information from multiple documents when relevant to provide a complete picture.

DETAILED ANSWER:"""
    
    def ask_question(self, question: str, 
                    category_filter: Optional[str] = None,
                    show_sources: bool = True) -> Dict:
        """
        Ask a question to the RAG system.
        
        Args:
            question: The question to ask
            category_filter: Optional category to filter results
            show_sources: Whether to include source information in response
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Apply category filter if specified
            if category_filter:
                # Get filtered retriever
                filtered_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.top_k,
                        "filter": {"category": category_filter}
                    }
                )
                retrieved_docs = filtered_retriever.invoke(question)
            else:
                retrieved_docs = self.retriever.invoke(question)
            
            # Format context and generate answer
            context = self._format_docs(retrieved_docs)
            prompt = self._build_rag_prompt(context, question)
            answer = self.llm.invoke(prompt)
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer,
                "num_sources": len(retrieved_docs),
                "category_filter": category_filter
            }
            
            if show_sources:
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "source_file": doc.metadata.get('source_file', 'Unknown'),
                        "category": doc.metadata.get('category', 'Unknown'),
                        "chunk_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
                response["sources"] = sources
            
            logger.info("Question processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error al procesar la pregunta: {str(e)}",
                "error": True
            }
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories in the database."""
        try:
            # Get sample documents to extract categories
            sample_docs = self.vector_store.similarity_search("climate", k=50)
            categories = set()
            
            for doc in sample_docs:
                if 'category' in doc.metadata:
                    categories.add(doc.metadata['category'])
            
            return sorted(list(categories))
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def interactive_mode(self):
        """Start interactive question-answering session."""
        print("\n" + "="*60)
        print("🌍 RAG Question Answering - Dataset Greenpeace")
        print("="*60)
        print("Escribe 'quit' o 'salir' para terminar")
        print("Escribe 'categories' para ver las categorías disponibles")
        print("Usa 'category:NOMBRE' antes de tu pregunta para filtrar por categoría")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 Tu pregunta: ").strip()
                
                if user_input.lower() in ['quit', 'salir', 'exit']:
                    print("¡Hasta luego! 👋")
                    break
                
                if user_input.lower() == 'categories':
                    categories = self.get_available_categories()
                    print("\n📂 Categorías disponibles:")
                    for cat in categories:
                        print(f"  - {cat}")
                    continue
                
                if not user_input:
                    print("Por favor, escribe una pregunta.")
                    continue
                
                # Check for category filter
                category_filter = None
                if user_input.startswith('category:'):
                    parts = user_input.split(':', 1)
                    if len(parts) == 2:
                        category_filter = parts[0].replace('category', '').strip()
                        user_input = parts[1].strip()
                
                # Process question
                print("\n🔍 Buscando información relevante...")
                response = self.ask_question(user_input, category_filter=category_filter)
                
                if response.get('error'):
                    print(f"❌ {response['answer']}")
                    continue
                
                # Display answer
                print(f"\n🤖 **Respuesta:**")
                print(response['answer'])
                
                # Display sources
                if 'sources' in response and response['sources']:
                    print(f"\n📚 **Fuentes consultadas** ({response['num_sources']} documentos):")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['source_file']} (Categoría: {source['category']})")
                        print(f"     Preview: {source['chunk_preview']}")
                
                if response.get('category_filter'):
                    print(f"\n🏷️  Filtrado por categoría: {response['category_filter']}")
                
            except KeyboardInterrupt:
                print("\n\n¡Hasta luego! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the RAG QA system."""
    print("Inicializando sistema RAG...")
    
    try:
        # Initialize RAG system
        rag_qa = RAGQuestionAnswering()
        
        # Show system info
        print(f"\n✅ Sistema listo!")
        print(f"📊 Categorías disponibles: {len(rag_qa.get_available_categories())}")
        print(f"🔍 Chunks por consulta: {rag_qa.top_k}")
        print(f"🤖 Modelo: {rag_qa.llm.model} (Ollama con parámetros optimizados)")
        print(f"🎯 Configuración mejorada: temperatura 0.2, tokens 2048, retrieval top_k 8")
        
        # Example questions
        print("\n🧪 Probando con preguntas de ejemplo...")
        
        example_questions = [
            "¿Qué dice Greenpeace sobre el cambio climático?",
            "¿Cuáles son los principales problemas con los combustibles fósiles?",
            "¿Qué información hay sobre Bitcoin y su impacto ambiental?"
        ]
        
        for i, question in enumerate(example_questions, 1):
            print(f"\n--- Ejemplo {i} ---")
            print(f"Pregunta: {question}")
            
            response = rag_qa.ask_question(question, show_sources=False)
            print(f"Respuesta: {response['answer'][:200]}...")
            print(f"Fuentes: {response['num_sources']} documentos")
        
        # Start interactive mode
        rag_qa.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Ejecuta primero: python rag_text_processor.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    main()