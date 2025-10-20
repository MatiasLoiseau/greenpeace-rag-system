#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para fragmentar documentos Markdown del dataset de Greenpeace
y almacenarlos en ChromaDB para uso en RAG.
"""

import os
import glob
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_markdown_files(dataset_path):
    """
    Carga todos los archivos markdown del dataset.
    
    Args:
        dataset_path: Ruta al directorio del dataset
        
    Returns:
        Lista de tuplas (filepath, content)
    """
    md_files = glob.glob(os.path.join(dataset_path, "*.md"))
    print(f"Encontrados {len(md_files)} archivos markdown")
    
    documents = []
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append((filepath, content))
        except Exception as e:
            print(f"Error leyendo {filepath}: {e}")
    
    return documents

def fragment_documents(documents):
    """
    Fragmenta documentos usando técnicas específicas para Markdown.
    
    Args:
        documents: Lista de tuplas (filepath, content)
        
    Returns:
        Lista de documentos fragmentados con metadata
    """
    # Configurar splitters
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3")
    ]
    
    # Primer paso: dividir por headers de Markdown
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    # Segundo paso: dividir chunks grandes recursivamente
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_splits = []
    
    for filepath, content in documents:
        filename = os.path.basename(filepath)
        print(f"Procesando: {filename}")
        
        try:
            # Primera fragmentación por headers
            md_header_splits = markdown_splitter.split_text(content)
            
            # Agregar metadata de archivo
            for doc in md_header_splits:
                doc.metadata['source'] = filename
                doc.metadata['filepath'] = filepath
            
            # Segunda fragmentación recursiva si los chunks son muy grandes
            splits = text_splitter.split_documents(md_header_splits)
            
            all_splits.extend(splits)
            print(f"  - Generados {len(splits)} fragmentos")
            
        except Exception as e:
            print(f"  - Error procesando {filename}: {e}")
    
    return all_splits

def create_vector_store(documents, persist_directory="./chroma_db"):
    """
    Crea y persiste un vector store en ChromaDB.
    
    Args:
        documents: Lista de documentos fragmentados
        persist_directory: Directorio donde persistir la base de datos
        
    Returns:
        Vector store de ChromaDB
    """
    print(f"\nCreando embeddings con HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Creando vector store con {len(documents)} documentos...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="greenpeace_docs"
    )
    
    print(f"Vector store persistido en: {persist_directory}")
    return vectorstore

def main():
    """Función principal"""
    # Configuración
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "dataset"
    chroma_path = base_dir / "chroma_db"
    
    print("=" * 60)
    print("FRAGMENTACIÓN Y ALMACENAMIENTO EN CHROMADB")
    print("=" * 60)
    
    # Paso 1: Cargar archivos markdown
    print(f"\n1. Cargando archivos desde: {dataset_path}")
    documents = load_markdown_files(dataset_path)
    
    if not documents:
        print("No se encontraron documentos. Verificar la ruta del dataset.")
        return
    
    # Paso 2: Fragmentar documentos
    print(f"\n2. Fragmentando {len(documents)} documentos...")
    splits = fragment_documents(documents)
    
    print(f"\nTotal de fragmentos generados: {len(splits)}")
    
    # Mostrar ejemplo de fragmento
    if splits:
        print("\nEjemplo de fragmento:")
        print("-" * 60)
        print(f"Contenido: {splits[0].page_content[:200]}...")
        print(f"Metadata: {splits[0].metadata}")
        print("-" * 60)
    
    # Paso 3: Crear y persistir vector store
    print(f"\n3. Creando vector store en ChromaDB...")
    vectorstore = create_vector_store(splits, persist_directory=str(chroma_path))
    
    # Paso 4: Probar el vector store
    print("\n4. Probando búsqueda en el vector store...")
    test_query = "What are the main threats to bees?"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nResultados para query: '{test_query}'")
    for i, doc in enumerate(results, 1):
        print(f"\nResultado {i}:")
        print(f"Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"Contenido: {doc.page_content[:200]}...")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nPara usar el vector store en tu RAG:")
    print(f"  from langchain_community.vectorstores import Chroma")
    print(f"  from langchain_huggingface import HuggingFaceEmbeddings")
    print(f"  ")
    print(f"  embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')")
    print(f"  vectorstore = Chroma(")
    print(f"      persist_directory='{chroma_path}',")
    print(f"      embedding_function=embeddings,")
    print(f"      collection_name='greenpeace_docs'")
    print(f"  )")

if __name__ == "__main__":
    main()
