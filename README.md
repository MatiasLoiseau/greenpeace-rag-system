# Greenpeace RAG System

Sistema de Recuperación Aumentada por Generación (RAG) para documentos de Greenpeace.

## Descripción

Este proyecto procesa documentos en formato Markdown de Greenpeace, los fragmenta usando técnicas específicas para Markdown y los almacena en ChromaDB para consultas eficientes usando embeddings.

## Estructura del Proyecto

```
greenpeace-rag-system/
├── dataset/                  # Archivos markdown originales
├── chroma_db/               # Base de datos vectorial (generada)
├── docs/                    # Documentación y ejemplos
├── process_dataset.py       # Script para procesar y almacenar documentos
├── query_rag.py            # Script para consultar el RAG
└── requirements.txt        # Dependencias del proyecto
```

## Instalación

1. Activar el entorno conda:
```bash
conda activate llm
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Procesar el Dataset

Este paso solo necesita ejecutarse una vez (o cuando se actualice el dataset):

```bash
python process_dataset.py
```

Este script:
- Lee todos los archivos `.md` del directorio `dataset/`
- Los fragmenta usando `MarkdownHeaderTextSplitter` (respetando headers H1, H2, H3)
- Aplica `RecursiveCharacterTextSplitter` para chunks más grandes (max 1000 caracteres)
- Genera embeddings usando `all-MiniLM-L6-v2` de HuggingFace
- Almacena todo en ChromaDB en el directorio `chroma_db/`

**Estadísticas del procesamiento:**
- 332 archivos markdown procesados
- 27,287 fragmentos generados
- Embeddings de 384 dimensiones

### 2. Consultar el RAG

#### Modo Interactivo

```bash
python query_rag.py
```

Comandos disponibles:
- **Búsqueda simple**: Escribe tu consulta directamente
- **Búsqueda con LLM**: Usa el prefijo `llm:` para generar respuestas con Ollama
- **Salir**: `quit` o `exit`

Ejemplo:
```
🔍 Consulta: What are the main threats to bees?
# Muestra los 5 documentos más relevantes con scores

🔍 Consulta: llm: Explain the main threats to bees
# Genera una respuesta usando Ollama basada en los documentos
```

#### Modo Línea de Comandos

```bash
python query_rag.py "your query here"
```

Ejemplo:
```bash
python query_rag.py "climate change impacts on forests"
```

### 3. Usar en tu Código

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Cargar embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Cargar vector store
vectorstore = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embeddings,
    collection_name='greenpeace_docs'
)

# Buscar documentos
results = vectorstore.similarity_search("your query", k=5)

# Buscar con scores
results_with_scores = vectorstore.similarity_search_with_score("your query", k=5)

# Usar como retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("your query")
```

## Técnicas de Fragmentación

El sistema usa una estrategia de fragmentación en dos pasos:

### 1. MarkdownHeaderTextSplitter
- Divide documentos respetando la jerarquía de headers (#, ##, ###)
- Preserva el contexto de los headers en los metadatos
- Mantiene la estructura semántica del documento

### 2. RecursiveCharacterTextSplitter
- Fragmenta chunks grandes (> 1000 caracteres)
- Overlap de 100 caracteres entre chunks
- Mantiene párrafos y frases completas cuando es posible

## Metadatos

Cada fragmento incluye:
- `source`: Nombre del archivo original
- `filepath`: Ruta completa al archivo
- `H1`, `H2`, `H3`: Headers jerárquicos del documento (cuando aplican)

## Integración con Ollama

Para usar el modo LLM con Ollama:

1. Asegúrate de tener Ollama instalado y corriendo
2. Descarga un modelo (ej: `ollama pull llama3.2`)
3. Usa el prefijo `llm:` en el modo interactivo

```bash
python query_rag.py
🔍 Consulta: llm: What are the environmental impacts of palm oil production?
```

## Configuración Avanzada

### Cambiar el Modelo de Embeddings

En `process_dataset.py` o `query_rag.py`, modifica:

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

Modelos recomendados:
- `all-MiniLM-L6-v2` (384 dim) - Rápido y eficiente
- `all-mpnet-base-v2` (768 dim) - Más preciso pero más lento
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Para multilenguaje

### Ajustar Tamaño de Chunks

En `process_dataset.py`, línea ~65:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Ajustar según necesidad
    chunk_overlap=100,    # Ajustar overlap
    length_function=len,
    is_separator_regex=False,
)
```

## Troubleshooting

### Error: "No module named 'langchain_huggingface'"
```bash
pip install langchain-huggingface
```

### Error: "Could not connect to Ollama"
- Verifica que Ollama esté corriendo: `ollama serve`
- Verifica que el modelo esté instalado: `ollama list`

### Base de datos no encontrada
```bash
python process_dataset.py  # Re-procesar dataset
```

## Estadísticas del Dataset

- **Total de archivos**: 332 documentos markdown
- **Total de fragmentos**: 27,287 chunks
- **Tamaño promedio de chunk**: ~400-600 caracteres
- **Vector store size**: ~150-200 MB

## Ejemplos de Consultas

1. **Temas ambientales**:
   - "climate change impacts"
   - "deforestation in the Amazon"
   - "ocean plastic pollution"

2. **Especies específicas**:
   - "threats to bees and pollinators"
   - "whale conservation"
   - "coral reef destruction"

3. **Industrias**:
   - "palm oil industry"
   - "fossil fuel companies"
   - "nuclear power plants"

4. **Geografía**:
   - "forest destruction in Indonesia"
   - "Arctic drilling"
   - "Congo basin logging"

## Licencia

Este proyecto procesa documentos de Greenpeace. Asegúrate de respetar las licencias y derechos de autor de los documentos originales.
