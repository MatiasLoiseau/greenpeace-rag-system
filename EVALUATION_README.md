# Sistema de Evaluación del RAG de Greenpeace

Este sistema permite evaluar el rendimiento de tu RAG de manera sistemática y reproducible, generando un conjunto de preguntas de prueba y evaluando múltiples métricas de calidad.

## 📋 Descripción

El sistema consta de dos componentes principales:

1. **`generate_test_questions.py`**: Genera 100 preguntas de evaluación a partir de párrafos aleatorios de los documentos de Greenpeace
2. **`evaluate_rag.py`**: Evalúa el RAG usando las preguntas generadas y calcula métricas de rendimiento

## 🎯 Métricas Evaluadas

El sistema evalúa 4 métricas clave:

1. **Correctitud** (_Correctness_): ¿La respuesta es factualmente correcta comparada con la respuesta de referencia?
2. **Relevancia** (_Relevance_): ¿La respuesta es relevante y responde la pregunta?
3. **Fundamentación** (_Grounding_): ¿La respuesta está basada en los documentos recuperados?
4. **Relevancia de Recuperación** (_Retrieval Relevance_): ¿Los documentos recuperados son relevantes para la pregunta?

## 🚀 Uso

### Requisitos Previos

1. **Ollama con Llama 3.1**: Asegúrate de tener Ollama corriendo y el modelo instalado:
   ```bash
   # Instalar Ollama si no lo tienes
   # Visita: https://ollama.ai
   
   # Descargar Llama 3.1
   ollama pull llama3.1
   
   # Verificar que esté corriendo
   ollama list
   ```

2. **Entorno Conda** (opcional): Si usas el entorno `llm`:
   ```bash
   conda activate llm
   ```

3. **Dependencias Python**:
   ```bash
   pip install langchain langchain-community langchain-huggingface chromadb ollama pydantic tqdm
   ```

4. **Base de datos ChromaDB**: Asegúrate de haber procesado los documentos:
   ```bash
   python process_dataset.py
   ```

### Paso 1: Generar Preguntas de Evaluación

```bash
python generate_test_questions.py
```

Este script:
- Selecciona 100 párrafos aleatorios de los documentos
- Para cada párrafo, genera:
  - Una pregunta original
  - Una respuesta ground truth (respuesta correcta)
  - Fragmentos relevantes del contexto
  - Una pregunta evolucionada (más comprimida e indirecta)
  - Métricas de calidad (groundedness y relevance scores)
- Guarda todo en `test_questions.json`

**Tiempo estimado**: 30-60 minutos dependiendo de tu hardware y conexión con Ollama.

**Nota**: Las preguntas se generan UNA SOLA VEZ. Si el archivo `test_questions.json` ya existe, el script te preguntará si deseas regenerarlas.

### Paso 2: Evaluar el RAG

```bash
# Evaluar con todas las preguntas
python evaluate_rag.py

# Evaluar con un subconjunto (ej: 10 preguntas)
python evaluate_rag.py -n 10

# Cambiar el número de documentos recuperados (default: 3)
python evaluate_rag.py -k 5

# Especificar ruta a ChromaDB
python evaluate_rag.py --chroma-path ./mi_chroma_db
```

**Opciones disponibles**:
- `-n, --num-questions`: Número de preguntas a evaluar (default: todas)
- `-k, --num-docs`: Número de documentos a recuperar del vector store (default: 3)
- `--chroma-path`: Ruta a la base de datos ChromaDB (default: ./chroma_db)

### Paso 3: Analizar Resultados

Los resultados se guardan en el directorio `evaluation_results/`:

1. **`evaluation_YYYYMMDD_HHMMSS.json`**: Resultados completos en formato JSON
   - Métricas agregadas
   - Resultados detallados por pregunta
   - Respuestas generadas y documentos recuperados

2. **`summary_YYYYMMDD_HHMMSS.txt`**: Resumen legible en texto plano
   - Métricas globales
   - Detalles de cada pregunta evaluada
   - Indicadores visuales (✅/❌) por métrica

## 📊 Ejemplo de Salida

```
======================================================================
RESULTADOS
======================================================================

📈 Métricas Globales:
   Total de preguntas: 100
   Correctitud: 85.00% (85/100)
   Relevancia: 92.00% (92/100)
   Fundamentación: 88.00% (88/100)
   Relevancia de Recuperación: 95.00% (95/100)

💾 Resultados guardados en: evaluation_results/evaluation_20250120_143022.json
📄 Resumen guardado en: evaluation_results/summary_20250120_143022.txt
```

## 🔄 Flujo de Trabajo Típico

### Primera vez:
```bash
# 1. Generar preguntas (una sola vez)
python generate_test_questions.py

# 2. Evaluar el RAG base
python evaluate_rag.py
```

### Después de hacer cambios al RAG:
```bash
# Solo re-evaluar (usa las mismas preguntas)
python evaluate_rag.py
```

### Comparar diferentes configuraciones:
```bash
# Probar con k=3
python evaluate_rag.py -k 3

# Probar con k=5
python evaluate_rag.py -k 5

# Probar con k=10
python evaluate_rag.py -k 10
```

## 📁 Estructura de Archivos

```
greenpeace-rag-system/
├── dataset/                           # Documentos originales
├── chroma_db/                         # Base de datos vectorial
├── generate_test_questions.py         # Script para generar preguntas
├── evaluate_rag.py                    # Script para evaluar el RAG
├── test_questions.json                # Preguntas de evaluación (generado)
└── evaluation_results/                # Resultados de evaluaciones (generado)
    ├── evaluation_YYYYMMDD_HHMMSS.json
    └── summary_YYYYMMDD_HHMMSS.txt
```

## 🎓 Metodología

Este sistema de evaluación está basado en las mejores prácticas para evaluar sistemas RAG:

1. **Generación Sintética de Preguntas**: Usa LLMs para generar preguntas realistas a partir de contextos reales
2. **Evolución de Preguntas**: Crea variaciones más desafiantes de las preguntas originales
3. **Filtrado de Calidad**: Solo incluye preguntas con altos scores de fundamentación y relevancia
4. **Evaluación Multidimensional**: Mide diferentes aspectos del rendimiento del RAG
5. **Reproducibilidad**: Las mismas preguntas se usan en todas las evaluaciones para comparaciones justas

## 🔧 Configuración Avanzada

### Cambiar el Modelo LLM

Edita las constantes en los archivos Python:

```python
# En generate_test_questions.py y evaluate_rag.py
MODEL_NAME = "llama3.1"  # Cambia a "llama2", "mistral", etc.
```

### Ajustar Parámetros de Generación

En `generate_test_questions.py`:

```python
# Cambiar el número de preguntas
test_questions = generate_all_questions(num_questions=200)

# Ajustar temperatura del LLM
llm = Ollama(model=model_name, temperature=0.7)
```

### Ajustar Prompts de Evaluación

Los prompts están definidos como constantes en `evaluate_rag.py`. Puedes modificarlos para ajustar los criterios de evaluación.

## ⚠️ Troubleshooting

### Error: "Ollama no está corriendo"
```bash
# Iniciar Ollama
ollama serve
```

### Error: "Modelo no encontrado"
```bash
# Descargar el modelo
ollama pull llama3.1
```

### Error: "No se encontró ChromaDB"
```bash
# Procesar documentos primero
python process_dataset.py
```

### Las preguntas generadas son de baja calidad
- Ajusta los scores mínimos en `generate_test_question()`:
  ```python
  if groundedness_score < 4 or relevance_score < 3:  # Ajustar estos valores
  ```

### La evaluación es muy lenta
- Reduce el número de preguntas: `python evaluate_rag.py -n 20`
- Usa un modelo más rápido (ej: llama2 en lugar de llama3.1)
- Reduce el número de documentos recuperados: `python evaluate_rag.py -k 2`

## 📈 Mejores Prácticas

1. **Genera las preguntas UNA VEZ**: Usa el mismo conjunto de preguntas para todas tus evaluaciones
2. **Guarda los resultados**: No sobrescribas evaluaciones anteriores, compara múltiples ejecuciones
3. **Evalúa incremental**: Empieza con pocas preguntas (-n 10) para probar cambios rápidamente
4. **Documenta cambios**: Anota qué cambios hiciste al RAG antes de cada evaluación
5. **Analiza fallos**: Revisa las preguntas donde el RAG falló para identificar patrones

## 📚 Referencias

- [LangChain RAG Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- Basado en el notebook: `docs/03_rags_evaluación.py`

## 💡 Próximos Pasos

Posibles mejoras al sistema:

- [ ] Agregar más métricas (faithfulness, context recall, etc.)
- [ ] Crear visualizaciones de resultados
- [ ] Implementar comparación automática entre evaluaciones
- [ ] Agregar tests de regresión
- [ ] Soportar múltiples LLMs simultáneamente
- [ ] Crear dashboard interactivo para análisis

---

¿Preguntas o problemas? Revisa los logs de ejecución o consulta la documentación de LangChain y Ollama.
