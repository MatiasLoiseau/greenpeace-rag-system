#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Question Answering System for Greenpeace Dataset
==================================================

This script uses the previously created ChromaDB index to answer questions
about the Greenpeace dataset using Google's Gemini model.

Features:
- Uses local embeddings for retrieval (no API cost)
- Gemini 2.0 Flash for generation (free tier friendly)
- Context-aware responses with source citations
- Configurable number of relevant chunks

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
                 model_name: str = "gemini-2.0-flash-exp",  # Free tier model
                 max_tokens: int = 1000,  # Conservative for free tier
                 top_k: int = 5):
        """
        Initialize the RAG QA system.
        
        Args:
            chroma_db_dir: Directory with the ChromaDB
            model_name: Gemini model to use
            max_tokens: Maximum tokens for response (keep low for free tier)
            top_k: Number of relevant chunks to retrieve
        """
        # Load environment variables
        load_dotenv()
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
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
        
        # Initialize Gemini model (free tier friendly settings)
        logger.info(f"Initializing {model_name} model...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for factual responses
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """Eres un asistente experto en temas ambientales y de Greenpeace. 
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, di "No tengo información suficiente en los documentos para responder esa pregunta."

Contexto de documentos de Greenpeace:
{context}

Pregunta: {question}

Instrucciones:
- Responde en español
- Sé preciso y factual
- Cita las fuentes cuando sea relevante
- Si hay múltiples perspectivas en el contexto, menciónalas
- Mantén la respuesta concisa pero informativa

Respuesta:"""
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
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
                
                # Create temporary chain with filtered retriever
                temp_chain = (
                    {"context": filtered_retriever | self._format_docs, "question": RunnablePassthrough()}
                    | self.prompt_template
                    | self.llm
                    | StrOutputParser()
                )
                
                answer = temp_chain.invoke(question)
                retrieved_docs = filtered_retriever.invoke(question)
            else:
                answer = self.rag_chain.invoke(question)
                retrieved_docs = self.retriever.invoke(question)
            
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
        print(f"🤖 Modelo: Gemini 2.0 Flash (versión gratuita)")
        
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