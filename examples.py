#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplos de uso del RAG de Greenpeace
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).parent
CHROMA_PATH = BASE_DIR / "chroma_db"

def ejemplo_1_busqueda_basica():
    """Ejemplo 1: Búsqueda básica de documentos"""
    print("\n" + "="*70)
    print("EJEMPLO 1: Búsqueda Básica")
    print("="*70)
    
    # Cargar embeddings y vectorstore
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    # Realizar búsqueda
    query = "What are the impacts of climate change on oceans?"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"\nConsulta: {query}")
    print(f"\nEncontrados {len(results)} documentos:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"   Contenido: {doc.page_content[:200]}...")
        print()

def ejemplo_2_busqueda_con_scores():
    """Ejemplo 2: Búsqueda con scores de similitud"""
    print("\n" + "="*70)
    print("EJEMPLO 2: Búsqueda con Scores")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    query = "nuclear power plant accidents"
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    print(f"\nConsulta: {query}")
    print(f"\nResultados ordenados por similitud:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"   Contenido: {doc.page_content[:150]}...")
        print()

def ejemplo_3_filtrado_por_metadata():
    """Ejemplo 3: Búsqueda con filtrado por metadata"""
    print("\n" + "="*70)
    print("EJEMPLO 3: Filtrado por Metadata")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    # Buscar solo en archivos que contengan "bees" en el nombre
    query = "pesticides effects"
    results = vectorstore.similarity_search(query, k=10)
    
    # Filtrar resultados
    bee_docs = [doc for doc in results if 'bee' in doc.metadata.get('source', '').lower()]
    
    print(f"\nConsulta: {query}")
    print(f"Filtro: Solo documentos sobre abejas")
    print(f"\nEncontrados {len(bee_docs)} documentos:\n")
    
    for i, doc in enumerate(bee_docs[:3], 1):
        print(f"{i}. Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"   Headers: {[v for k,v in doc.metadata.items() if k.startswith('H')]}")
        print(f"   Contenido: {doc.page_content[:200]}...")
        print()

def ejemplo_4_retriever():
    """Ejemplo 4: Usar como retriever para RAG"""
    print("\n" + "="*70)
    print("EJEMPLO 4: Uso como Retriever")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    # Crear retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    query = "Amazon rainforest deforestation"
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nConsulta: {query}")
    print(f"\nDocumentos recuperados: {len(docs)}\n")
    
    for i, doc in enumerate(docs[:3], 1):
        print(f"{i}. {doc.metadata.get('source', 'N/A')}")
        print(f"   {doc.page_content[:150]}...")
        print()

def ejemplo_5_mmr_search():
    """Ejemplo 5: Búsqueda con Maximum Marginal Relevance (diversidad)"""
    print("\n" + "="*70)
    print("EJEMPLO 5: Búsqueda MMR (Maximiza Diversidad)")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    query = "renewable energy"
    
    # MMR busca documentos relevantes pero diversos
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=5, 
        fetch_k=20  # Busca 20 candidatos, retorna 5 más diversos
    )
    
    print(f"\nConsulta: {query}")
    print(f"MMR retorna documentos relevantes pero diversos\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. Fuente: {doc.metadata.get('source', 'N/A')}")
        print(f"   Contenido: {doc.page_content[:150]}...")
        print()

def ejemplo_6_estadisticas():
    """Ejemplo 6: Estadísticas del vector store"""
    print("\n" + "="*70)
    print("EJEMPLO 6: Estadísticas del Vector Store")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    # Obtener colección
    collection = vectorstore._collection
    
    print(f"\nNombre de colección: {collection.name}")
    print(f"Total de documentos: {collection.count()}")
    
    # Obtener algunos metadatos únicos
    sample_docs = vectorstore.similarity_search("test", k=100)
    sources = set(doc.metadata.get('source', 'N/A') for doc in sample_docs)
    
    print(f"Archivos únicos en muestra: {len(sources)}")
    print(f"\nEjemplos de archivos:")
    for source in list(sources)[:5]:
        print(f"  - {source}")

def ejemplo_7_rag_simple():
    """Ejemplo 7: RAG simple sin LLM (solo concatenación de contexto)"""
    print("\n" + "="*70)
    print("EJEMPLO 7: RAG Simple (Contexto Concatenado)")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings,
        collection_name='greenpeace_docs'
    )
    
    query = "What is the impact of plastic pollution on marine life?"
    docs = vectorstore.similarity_search(query, k=3)
    
    # Crear contexto
    context = "\n\n---\n\n".join([
        f"[Fuente: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    print(f"\nConsulta: {query}")
    print(f"\n{'='*70}")
    print("CONTEXTO PARA LLM:")
    print('='*70)
    print(context[:1000] + "...")
    
    prompt_template = f"""Basándote en el siguiente contexto, responde la pregunta.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""
    
    print(f"\n{'='*70}")
    print("PROMPT PARA LLM:")
    print('='*70)
    print(prompt_template[:500] + "...")

def main():
    """Ejecutar todos los ejemplos"""
    ejemplos = [
        ejemplo_1_busqueda_basica,
        ejemplo_2_busqueda_con_scores,
        ejemplo_3_filtrado_por_metadata,
        ejemplo_4_retriever,
        ejemplo_5_mmr_search,
        ejemplo_6_estadisticas,
        ejemplo_7_rag_simple,
    ]
    
    print("\n" + "="*70)
    print("EJEMPLOS DE USO DEL RAG DE GREENPEACE")
    print("="*70)
    
    for i, ejemplo in enumerate(ejemplos, 1):
        input(f"\nPresiona Enter para ejecutar Ejemplo {i}...")
        ejemplo()
    
    print("\n" + "="*70)
    print("FIN DE LOS EJEMPLOS")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
