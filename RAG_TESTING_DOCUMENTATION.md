# RAG Testing and Evaluation System

Sistema completo de testing y evaluación para el sistema RAG de Greenpeace. Incluye generación automática de datasets de prueba, evaluación con LLM juez y métricas de rendimiento detalladas.

## 🎯 Características Principales

### 1. Generación de Dataset de Prueba (`rag_test_generator.py`)
- **Extracción inteligente de párrafos**: Selecciona párrafos relevantes de documentos del dataset
- **Generación de preguntas contextuales**: Usa Gemini para crear preguntas no literales
- **Respuestas de referencia (Ground Truth)**: Genera respuestas basadas en el contexto original
- **Metadata completa**: Guarda referencia al documento fuente, categoría, etc.
- **Configuración flexible**: 50-1000 preguntas según necesidades

### 2. Sistema de Evaluación (`rag_evaluator.py`)
- **LLM Juez**: Evaluación semántica con Gemini 2.0 Flash
- **Métricas multidimensionales**:
  - Similitud semántica (0-10)
  - Precisión factual (0-10)
  - Completitud (0-10)
  - Relevancia (0-10)
- **Evaluación de fuentes**: Precision y recall de documentos recuperados
- **Análisis por categorías**: Rendimiento segmentado por temas
- **Métricas agregadas**: Accuracy, F1-score, distribuciones de rendimiento

### 3. Suite Completa de Testing (`rag_testing_suite.py`)
- **Pipeline automatizado**: Desde generación hasta reporte final
- **Modos de operación**: Quick (desarrollo) y Full (evaluación completa)
- **Verificación de prerequisitos**: Valida configuración antes de ejecutar
- **Reportes duales**: JSON detallado y resumen en texto plano
- **Manejo de errores robusto**: Logging completo y recuperación de fallos

## 🚀 Uso Rápido

### Instalación y Setup
```bash
# 1. Asegurar que el sistema RAG esté configurado
python rag_text_processor.py  # Si no se ejecutó antes

# 2. Verificar que existe GOOGLE_API_KEY
export GOOGLE_API_KEY="tu_api_key_aqui"

# 3. Verificar prerequisitos
ls dataset/  # Debe contener archivos .txt
ls greenpeace/greenpeace.csv  # Metadata debe existir
ls chroma_db_rag/  # Base de datos vectorial debe existir
```

### Ejecución Básica

```bash
# Evaluación completa (1000 preguntas)
python rag_testing_suite.py

# Modo rápido para desarrollo (50 preguntas)
python rag_testing_suite.py --quick

# Número específico de preguntas
python rag_testing_suite.py --questions 100

# Usar dataset existente (no regenerar)
python rag_testing_suite.py --skip-generation
```

### Ejecución por Componentes

```bash
# Solo generar dataset de prueba
python rag_test_generator.py

# Solo ejecutar evaluación (requiere test_dataset.json)
python rag_evaluator.py
```

## 📊 Outputs y Reportes

### Archivos Generados

1. **`test_dataset.json`**: Dataset de preguntas y respuestas de referencia
2. **`test_dataset_stats.json`**: Estadísticas del dataset generado
3. **`rag_evaluation_report_YYYYMMDD_HHMMSS.json`**: Reporte detallado de evaluación
4. **`rag_testing_summary_YYYYMMDD_HHMMSS.txt`**: Resumen ejecutivo en texto plano
5. **Logs**: `rag_testing_suite_YYYYMMDD_HHMMSS.log`

### Estructura del Dataset de Prueba

```json
{
  "id": "qa_0001",
  "question": "¿Cuáles son las principales consecuencias del cambio climático según el documento?",
  "answer": "Según el documento, las principales consecuencias incluyen...",
  "source_paragraph": "El texto original del párrafo...",
  "source_file": "climate-change-impacts.txt",
  "category": "Climate Change",
  "title": "Climate Change Impacts Report",
  "paragraph_length": 456,
  "question_length": 89,
  "answer_length": 234
}
```

### Métricas de Evaluación

#### Métricas Principales (0-10)
- **Similitud Semántica**: Comparación de significado entre respuestas
- **Precisión Factual**: Corrección de hechos e información
- **Completitud**: Cobertura de aspectos importantes de la respuesta
- **Relevancia**: Utilidad y pertinencia para la pregunta

#### Métricas de Recuperación
- **Source Precision**: % de documentos recuperados correctos
- **Source Recall**: % de documentos relevantes recuperados
- **Category Accuracy**: % de coincidencia en categorías

#### Métricas de Rendimiento
- **Overall Score**: Puntuación ponderada general
- **Score Distribution**: Distribución de rendimiento por rangos
- **Error Rate**: % de evaluaciones fallidas
- **Evaluation Time**: Tiempo promedio por pregunta

## 🔧 Configuración y Personalización

### Variables de Configuración

En `rag_test_generator.py`:
```python
class RAGTestGenerator:
    def __init__(self, 
                 target_questions: int = 1000,           # Número de preguntas
                 min_paragraph_length: int = 200,        # Longitud mínima párrafo
                 max_paragraph_length: int = 800):       # Longitud máxima párrafo
```

En `rag_evaluator.py`:
```python
class RAGEvaluator:
    def __init__(self, 
                 max_tokens: int = 800,                  # Tokens para LLM juez
                 top_k: int = 5):                        # Chunks recuperados por RAG
```

### Personalización de Prompts

Los prompts para generación y evaluación están en las clases respectivas y pueden modificarse para:
- Cambiar el estilo de preguntas generadas
- Ajustar criterios de evaluación del LLM juez
- Modificar el idioma de evaluación
- Añadir nuevas dimensiones de evaluación

### Filtros y Segmentación

```python
# Evaluar solo ciertas categorías
evaluator.evaluate_dataset(category_filter="Climate Change")

# Evaluar subset específico
evaluator.evaluate_dataset(max_questions=100)
```

## 📈 Interpretación de Resultados

### Rangos de Puntuación

| Rango | Interpretación | Acción Recomendada |
|-------|---------------|-------------------|
| 9-10  | Excelente     | Mantener configuración |
| 7-8   | Bueno         | Optimizaciones menores |
| 5-6   | Regular       | Mejoras significativas necesarias |
| 3-4   | Pobre         | Revisión mayor requerida |
| 0-2   | Muy Pobre     | Reconfiguración completa |

### Diagnóstico de Problemas

#### Baja Similitud Semántica
- **Causa**: Respuestas conceptualmente diferentes
- **Solución**: Ajustar parámetros de retrieval, mejorar chunking

#### Baja Precisión Factual
- **Causa**: Información incorrecta o desactualizada
- **Solución**: Validar datos fuente, mejorar prompt del RAG

#### Baja Completitud
- **Causa**: Respuestas muy cortas o incompletas
- **Solución**: Aumentar `top_k`, ajustar `max_tokens`

#### Baja Precisión de Fuentes
- **Causa**: Retrieval recupera documentos irrelevantes
- **Solución**: Ajustar similarity threshold, mejorar embeddings

## 🔍 Casos de Uso

### 1. Desarrollo y Testing
```bash
# Testing rápido durante desarrollo
python rag_testing_suite.py --quick

# Testing de cambios específicos
python rag_testing_suite.py --questions 25 --skip-generation
```

### 2. Evaluación de Producción
```bash
# Evaluación completa antes de deployment
python rag_testing_suite.py --questions 1000

# Evaluación continua (usar dataset existente)
python rag_testing_suite.py --skip-generation
```

### 3. Análisis Comparativo
```bash
# Evaluar diferentes configuraciones
python rag_testing_suite.py --questions 200  # Config A
# Cambiar parámetros RAG
python rag_evaluator.py  # Config B con mismo dataset
```

### 4. Debugging Específico
```python
# En Python script personalizado
from rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.evaluate_single_question(test_item)
print(f"Detailed scores: {result}")
```

## ⚠️ Limitaciones y Consideraciones

### Limitaciones del Sistema
1. **Dependencia de LLM**: La calidad de evaluación depende del LLM juez
2. **Subjetividad**: Algunas métricas pueden tener componente subjetivo
3. **Contexto limitado**: Los párrafos pueden perder contexto al extraerse
4. **Sesgo de generación**: Las preguntas reflejan el estilo del LLM generador

### Consideraciones de Costo
- **Modo Quick**: ~$0.50-1.00 USD (50 preguntas)
- **Modo Full**: ~$10.00-20.00 USD (1000 preguntas)
- **Factor principal**: Llamadas a Gemini API para generación y evaluación

### Recomendaciones de Uso
1. **Usar modo Quick** durante desarrollo activo
2. **Evaluar Full** antes de releases importantes
3. **Validar manualmente** una muestra de resultados
4. **Complementar** con testing humano para casos críticos

## 🛠️ Troubleshooting

### Errores Comunes

#### "ChromaDB not found"
```bash
# Solución: Ejecutar procesamiento de texto primero
python rag_text_processor.py
```

#### "GOOGLE_API_KEY not found"
```bash
# Solución: Configurar variable de entorno
export GOOGLE_API_KEY="tu_api_key"
# O crear archivo .env con la key
```

#### "No valid paragraphs found"
```bash
# Solución: Verificar archivos de dataset
ls dataset/*.txt | wc -l  # Debe ser > 10
```

#### "Error parsing LLM evaluation"
- **Causa**: Respuesta inesperada del LLM juez
- **Solución**: Verificar prompts, reintentar con delay

### Performance Issues

#### Evaluación muy lenta
```python
# Reducir max_tokens en evaluador
evaluator = RAGEvaluator(max_tokens=400)  # Default: 800

# Usar menos preguntas para testing
python rag_testing_suite.py --questions 50
```

#### Out of memory
```python
# Procesar en lotes más pequeños
# Modificar batch_size en el código si es necesario
```

## 📚 Referencias y Recursos Adicionales

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Google Gemini API](https://ai.google.dev/)
- [RAG Evaluation Best Practices](https://docs.llamaindex.ai/en/stable/optimizing/evaluation/)

---

Este sistema de testing proporciona una evaluación robusta y automatizada del sistema RAG, permitiendo iteración rápida durante el desarrollo y validación confiable para producción.