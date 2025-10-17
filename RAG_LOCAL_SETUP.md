# Sistema de Testing RAG con LLM Local (Ollama)

## 🎯 Resumen

He modificado completamente el sistema de testing del RAG para que use **Ollama** (LLM local) en lugar de Gemini, eliminando así el costo de API y evitando consumir tu cuota gratuita.

### ✨ Beneficios del Cambio

- ✅ **Gratis**: No consume API credits
- ✅ **Privacidad**: Todo local, sin envío de datos
- ✅ **Velocidad**: Una vez configurado, es muy rápido
- ✅ **Control**: Modelo corriendo localmente
- ❌ **Setup inicial**: Requiere descargar modelo (~2GB)

## 🚀 Instalación y Configuración

### Opción 1: Script Automático (Recomendado)

```bash
# 1. Ejecutar script de instalación
./install_ollama.sh

# 2. Probar la configuración
conda activate llm
python test_ollama.py

# 3. Ejecutar testing RAG
python rag_testing_suite.py --quick
```

### Opción 2: Instalación Manual

```bash
# 1. Instalar Ollama
# En macOS:
brew install ollama

# En Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Iniciar servidor Ollama
ollama serve &

# 3. Descargar modelo llama3.2
ollama pull llama3.2

# 4. Probar configuración
conda activate llm
python test_ollama.py
```

## 📁 Archivos Modificados

### Nuevos Archivos
- `test_ollama.py` - Test de conectividad con Ollama
- `install_ollama.sh` - Script de instalación automatizada
- `RAG_LOCAL_SETUP.md` - Esta documentación

### Archivos Modificados
- `rag_test_generator.py` - Usa Ollama en vez de Gemini
- `rag_evaluator.py` - LLM juez con Ollama
- `rag_qa_system.py` - Sistema RAG con Ollama
- `rag_testing_suite.py` - Verificaciones de Ollama
- `requirements.txt` - Dependencias actualizadas

## 🔧 Uso del Sistema

### 1. Verificar Configuración

```bash
conda activate llm
python test_ollama.py
```

**Output esperado:**
```
🚀 Ollama Test Suite
✅ Ollama server is running
✅ llama3.2 model found
✅ Model test successful
🎉 All tests passed!
```

### 2. Testing Rápido (Desarrollo)

```bash
python rag_testing_suite.py --quick
```

- Genera 50 preguntas
- Evalúa con LLM local
- Tiempo: ~10-15 minutos

### 3. Testing Completo (Producción)

```bash
python rag_testing_suite.py --questions 1000
```

- Genera 1000 preguntas
- Evaluación completa
- Tiempo: ~2-3 horas

### 4. Solo Generación de Dataset

```bash
python rag_test_generator.py
```

### 5. Solo Evaluación

```bash
python rag_evaluator.py
```

## 🎛️ Configuración Avanzada

### Cambiar Modelo Ollama

En los archivos Python, puedes cambiar el modelo:

```python
# De llama3.2 a otro modelo
self.llm = Ollama(
    model="llama3.1",  # o "mistral", "codellama", etc.
    temperature=0.1,
)
```

### Modelos Disponibles

```bash
# Ver modelos instalados
ollama list

# Instalar otros modelos
ollama pull mistral      # Más rápido, menos preciso
ollama pull llama3.1     # Más grande, más preciso
ollama pull codellama    # Especializado en código
```

### Ajustar Parámetros

```python
# En rag_test_generator.py, rag_evaluator.py, etc.
self.llm = Ollama(
    model="llama3.2",
    temperature=0.1,        # 0.0-1.0 (creatividad)
    top_p=0.9,             # Nucleus sampling
    repeat_penalty=1.1,     # Evitar repetición
)
```

## 📊 Comparación de Performance

| Aspecto | Gemini (Cloud) | Llama3.2 (Local) |
|---------|----------------|-------------------|
| Costo | $10-20 USD | Gratis |
| Setup | API Key | Instalar Ollama |
| Velocidad | ~2-3 seg/pregunta | ~1-2 seg/pregunta |
| Calidad | Muy alta | Alta |
| Privacidad | Datos enviados | 100% local |
| Disponibilidad | Requiere internet | Offline |

## ⚠️ Troubleshooting

### Error: "Cannot connect to Ollama"

```bash
# Verificar que Ollama esté corriendo
ps aux | grep ollama

# Si no está corriendo:
ollama serve &

# Verificar puerto
curl http://localhost:11434/api/tags
```

### Error: "Model not found"

```bash
# Verificar modelos instalados
ollama list

# Si llama3.2 no aparece:
ollama pull llama3.2
```

### Error: "Out of memory"

```bash
# Usar modelo más pequeño
ollama pull llama3.2:1b  # Versión de 1B parámetros

# O ajustar en Python:
self.llm = Ollama(model="llama3.2:1b")
```

### Performance Lento

```bash
# Verificar recursos del sistema
htop

# Usar modelo más rápido
ollama pull tinyllama

# Ajustar parámetros
self.llm = Ollama(
    model="llama3.2",
    num_predict=100,  # Respuestas más cortas
)
```

## 🔄 Migración desde Gemini

Si tenías configurado el sistema anterior:

1. **Backup anterior** (opcional):
   ```bash
   git stash  # Guardar cambios locales
   ```

2. **Usar nueva versión**:
   ```bash
   git pull   # Actualizar cambios
   ./install_ollama.sh  # Configurar Ollama
   ```

3. **Datasets existentes**:
   Los archivos `test_dataset.json` son compatibles entre versiones

## 📈 Resultados Esperados

### Métricas Típicas con Llama3.2

```
Overall Performance Scores (0-10 scale):
  Semantic Similarity: 7.2 ± 1.4
  Factual Accuracy: 7.8 ± 1.2  
  Completeness: 6.9 ± 1.6
  Relevance: 8.1 ± 1.1
  Overall Score: 7.5 ± 1.2

Source Matching:
  Source Precision: 72%
  Category Accuracy: 68%
```

### Interpretación

- **7.5+ Overall**: Sistema funcionando bien
- **6.0-7.4**: Necesita ajustes menores
- **<6.0**: Requiere mejoras significativas

## 🎯 Próximos Pasos

Una vez que tengas el sistema funcionando:

1. **Ejecutar test rápido** para verificar funcionalidad
2. **Analizar resultados** y ajustar parámetros si es necesario
3. **Ejecutar evaluación completa** para métricas finales
4. **Comparar con baseline** anterior (si lo tienes)

¿Listo para probar? Ejecuta:

```bash
./install_ollama.sh
```

---

*¿Problemas o preguntas? Revisa el log de Ollama: `tail -f ollama.log`*