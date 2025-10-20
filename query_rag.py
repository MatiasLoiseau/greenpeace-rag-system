#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para consultar el RAG de documentos de Greenpeace
usando ChromaDB y opcionalmente un LLM local (Ollama).
"""

import sys
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(persist_directory="./chroma_db"):
    """
    Carga el vector store desde ChromaDB.
    
    Args:
        persist_directory: Directorio donde está persistida la base de datos
        
    Returns:
        Vector store de ChromaDB
    """
    print("Cargando vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="greenpeace_docs"
    )
    
    return vectorstore

def search_documents(vectorstore, query, k=5):
    """
    Busca documentos similares a la query.
    
    Args:
        vectorstore: Vector store de ChromaDB
        query: Consulta de búsqueda
        k: Número de documentos a retornar
        
    Returns:
        Lista de documentos encontrados
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

def search_with_scores(vectorstore, query, k=5):
    """
    Busca documentos con scores de similitud.
    
    Args:
        vectorstore: Vector store de ChromaDB
        query: Consulta de búsqueda
        k: Número de documentos a retornar
        
    Returns:
        Lista de tuplas (documento, score)
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

def query_with_llm(vectorstore, query, model="llama3.2"):
    """
    Consulta usando RAG con un LLM local (Ollama).
    
    Args:
        vectorstore: Vector store de ChromaDB
        query: Consulta de búsqueda
        model: Modelo de Ollama a usar
        
    Returns:
        Respuesta generada por el LLM
    """
    try:
        from langchain_community.llms import Ollama
        from langchain.chains import RetrievalQA
        
        print(f"Inicializando modelo {model}...")
        llm = Ollama(model=model)
        
        # Crear cadena de QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        print("Generando respuesta...")
        result = qa_chain.invoke({"query": query})
        
        return result
        
    except ImportError:
        print("Para usar LLM necesitas instalar: pip install langchain-community")
        return None
    except Exception as e:
        print(f"Error al usar LLM: {e}")
        print("Asegúrate de tener Ollama corriendo y el modelo instalado.")
        return None

def interactive_mode(vectorstore):
    """Modo interactivo de consulta."""
    print("\n" + "=" * 70)
    print("MODO INTERACTIVO - RAG Greenpeace")
    print("=" * 70)
    print("\nComandos disponibles:")
    print("  - Escribe tu consulta para buscar documentos")
    print("  - 'llm: <consulta>' para usar Ollama")
    print("  - 'quit' o 'exit' para salir")
    print("\n" + "=" * 70 + "\n")
    
    while True:
        try:
            query = input("\n🔍 Consulta: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'salir']:
                print("\n¡Hasta luego!")
                break
            
            # Verificar si se quiere usar LLM
            use_llm = query.lower().startswith("llm:")
            
            if use_llm:
                query = query[4:].strip()
                result = query_with_llm(vectorstore, query)
                
                if result:
                    print("\n" + "=" * 70)
                    print("RESPUESTA DEL LLM:")
                    print("=" * 70)
                    print(result['result'])
                    
                    print("\n" + "-" * 70)
                    print("FUENTES:")
                    print("-" * 70)
                    for i, doc in enumerate(result['source_documents'], 1):
                        print(f"\n{i}. {doc.metadata.get('source', 'N/A')}")
                        print(f"   {doc.page_content[:150]}...")
            else:
                # Búsqueda simple
                results = search_with_scores(vectorstore, query, k=5)
                
                print("\n" + "=" * 70)
                print(f"RESULTADOS (Top {len(results)}):")
                print("=" * 70)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n{i}. Score: {score:.4f}")
                    print(f"   Fuente: {doc.metadata.get('source', 'N/A')}")
                    headers = {k: v for k, v in doc.metadata.items() 
                              if k.startswith('H') and k != 'source'}
                    if headers:
                        print(f"   Headers: {headers}")
                    print(f"   Contenido: {doc.page_content[:200]}...")
                    
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Función principal"""
    base_dir = Path(__file__).parent
    chroma_path = base_dir / "chroma_db"
    
    # Verificar que existe la base de datos
    if not chroma_path.exists():
        print(f"❌ Error: No se encontró la base de datos en {chroma_path}")
        print("Ejecuta primero 'python process_dataset.py' para crear la base de datos.")
        return
    
    # Cargar vector store
    vectorstore = load_vectorstore(str(chroma_path))
    
    # Si se pasa una query como argumento, hacer búsqueda simple
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nBuscando: {query}\n")
        
        results = search_with_scores(vectorstore, query, k=5)
        
        print("=" * 70)
        print(f"RESULTADOS (Top {len(results)}):")
        print("=" * 70)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Fuente: {doc.metadata.get('source', 'N/A')}")
            headers = {k: v for k, v in doc.metadata.items() 
                      if k.startswith('H') and k != 'source'}
            if headers:
                print(f"   Headers: {headers}")
            print(f"   Contenido: {doc.page_content[:300]}...")
    else:
        # Modo interactivo
        interactive_mode(vectorstore)

if __name__ == "__main__":
    main()
