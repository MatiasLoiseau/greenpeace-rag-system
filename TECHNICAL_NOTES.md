# Notas Técnicas - Greenpeace RAG System

## Resumen del Sistema

### Arquitectura

```
┌─────────────────┐
│  Dataset (MD)   │
│  332 archivos   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Fragmentación en 2 Pasos           │
│  1. MarkdownHeaderTextSplitter      │
│  2. RecursiveCharacterTextSplitter  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  27,287 Fragmentos                  │
│  Chunk size: 1000 chars             │
│  Overlap: 100 chars                 │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Embeddings                         │
│  Modelo: all-MiniLM-L6-v2           │
│  Dimensiones: 384                   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  ChromaDB                           │
│  Collection: greenpeace_docs        │
│  Persistent Storage                 │
└─────────────────────────────────────┘
```

## Decisiones de Diseño

### 1. Estrategia de Fragmentación

**Por qué dos pasos?**

1. **MarkdownHeaderTextSplitter (Paso 1)**:
   - Preserva la estructura semántica del documento
   - Mantiene el contexto de headers en metadata
   - Respeta la jerarquía del contenido (#, ##, ###)
   - Genera chunks significativos basados en secciones

2. **RecursiveCharacterTextSplitter (Paso 2)**:
   - Fragmenta secciones muy largas (>1000 chars)
   - Evita chunks que excedan el contexto útil
   - Mantiene overlap para continuidad semántica
   - Prioriza mantener párrafos completos

**Alternativas consideradas**:
- Solo MarkdownTextSplitter: Generaría chunks muy grandes en algunos documentos
- Solo RecursiveCharacterTextSplitter: Perdería contexto de headers y estructura
- SemanticChunker: Más lento, requiere más recursos, resultados similares

### 2. Modelo de Embeddings

**Seleccionado**: `all-MiniLM-L6-v2`

**Ventajas**:
- Rápido (61 MB, ~6-8ms por embedding)
- Buen balance precisión/velocidad
- 384 dimensiones (menos almacenamiento que mpnet-base)
- Ampliamente usado y testeado

**Alternativas**:
- `all-mpnet-base-v2`: Más preciso (768 dim) pero 2x más lento
- `paraphrase-multilingual-MiniLM-L12-v2`: Para multi-idioma
- Embeddings de OpenAI: Requiere API key, costos

### 3. Vector Store

**Seleccionado**: ChromaDB

**Ventajas**:
- Persistencia local (no requiere servidor)
- Simple de usar con LangChain
- Soporta filtrado por metadata
- Open source y gratis
- Buena performance para datasets medianos (<100k docs)

**Alternativas**:
- Pinecone: Requiere cuenta, costos
- Weaviate: Más complejo de configurar
- FAISS: No tiene persistencia nativa, sin metadata filtering

## Configuración Óptima

### Tamaño de Chunks

```python
chunk_size=1000      # Óptimo para embeddings de 384 dim
chunk_overlap=100    # ~10% overlap para continuidad
```

**Razonamiento**:
- 1000 chars ≈ 150-250 tokens
- Suficiente contexto para embeddings significativos
- No tan grande como para diluir la semántica
- 100 chars overlap asegura continuidad entre chunks

### Headers a Procesar

```python
headers_to_split_on = [
    ("#", "H1"),     # Títulos principales
    ("##", "H2"),    # Secciones
    ("###", "H3")    # Subsecciones
]
```

**Por qué no H4, H5, H6?**
- El dataset no usa headers profundos consistentemente
- H3 ya proporciona suficiente granularidad
- Headers más profundos generarían demasiados chunks pequeños

## Métricas de Performance

### Procesamiento (MacBook Pro M1)
- Tiempo total: ~8-10 minutos
- Velocidad: ~33 archivos/minuto
- Embeddings: ~45 docs/segundo

### Búsqueda
- Latencia promedio: 50-100ms para k=5
- Con MMR: 100-150ms
- Primera carga: ~2-3 segundos (carga de modelos)

### Almacenamiento
```
chroma_db/
├── Total: ~180 MB
├── Embeddings: ~150 MB (27k × 384 × 4 bytes)
├── Metadata: ~20 MB
└── Índice: ~10 MB
```

## Metadata Schema

Cada fragmento incluye:

```python
{
    'source': 'nombre-archivo.md',        # Archivo origen
    'filepath': '/path/to/file.md',       # Ruta completa
    'H1': 'Título Principal',             # Opcional
    'H2': 'Sección',                      # Opcional
    'H3': 'Subsección'                    # Opcional
}
```

## Casos de Uso

### 1. Búsqueda de Documentos
```python
results = vectorstore.similarity_search("query", k=5)
```
**Uso**: Encontrar documentos relevantes sobre un tema

### 2. Q&A con LLM
```python
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
answer = qa_chain.invoke({"query": "question"})
```
**Uso**: Responder preguntas basadas en el dataset

### 3. Análisis de Cobertura
```python
results = vectorstore.similarity_search("topic", k=100)
sources = set(doc.metadata['source'] for doc in results)
```
**Uso**: Ver qué documentos cubren un tema específico

### 4. Extracción de Información
```python
results = vectorstore.max_marginal_relevance_search("query", k=10, fetch_k=50)
```
**Uso**: Obtener información diversa sobre un tema

## Limitaciones y Consideraciones

### 1. Multilenguaje
- El modelo `all-MiniLM-L6-v2` está optimizado para inglés
- Si el dataset tiene documentos en español, considerar:
  - `paraphrase-multilingual-MiniLM-L12-v2`
  - Embeddings multilenguaje de OpenAI

### 2. Fragmentación Semántica
- La fragmentación actual es estructural (headers + tamaño)
- Para mejorar: `SemanticChunker` (más lento pero mejor contexto)

### 3. Re-indexación
- Agregar documentos nuevos requiere re-procesar
- Considerar proceso incremental para datasets dinámicos

### 4. Escalabilidad
- ChromaDB funciona bien hasta ~100k-1M documentos
- Para más: considerar Pinecone, Weaviate o Qdrant

## Mejoras Futuras

### Corto Plazo
1. [ ] Proceso incremental de nuevos documentos
2. [ ] Interfaz web para consultas (Streamlit/Gradio)
3. [ ] Caché de consultas frecuentes
4. [ ] Logging y analytics de queries

### Medio Plazo
1. [ ] Reranking de resultados con modelo cross-encoder
2. [ ] Hybrid search (keyword + semantic)
3. [ ] Filtros dinámicos por metadata
4. [ ] Exportar resultados a formatos estructurados

### Largo Plazo
1. [ ] Fine-tuning del modelo de embeddings
2. [ ] Graph RAG para relaciones entre documentos
3. [ ] Multi-modal search (si hay imágenes)
4. [ ] Auto-evaluación de calidad de respuestas

## Troubleshooting Común

### Problema: Resultados poco relevantes
**Soluciones**:
1. Ajustar k (número de documentos)
2. Usar MMR para diversidad
3. Mejorar el query (más específico)
4. Considerar modelo de embeddings más grande

### Problema: Chunks muy largos/cortos
**Soluciones**:
1. Ajustar `chunk_size` en process_dataset.py
2. Modificar `chunk_overlap`
3. Revisar si headers están bien formateados en MD

### Problema: Búsqueda lenta
**Soluciones**:
1. Reducir k (documentos retornados)
2. No usar MMR si no es necesario
3. Considerar índice aproximado (HNSW)
4. Caché de embeddings de queries comunes

## Referencias

- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- ChromaDB Docs: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- RAG Best Practices: https://www.pinecone.io/learn/retrieval-augmented-generation/
