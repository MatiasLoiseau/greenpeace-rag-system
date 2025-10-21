#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar preguntas de evaluación para el RAG de Greenpeace.
Toma párrafos aleatorios de los documentos y genera preguntas usando Llama 3.1 local.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tqdm import tqdm

# Configurar para usar el entorno llm de conda
CONDA_ENV = "llm"
MODEL_NAME = "llama3.2"
OUTPUT_FILE = "test_questions.json"

# Modelos Pydantic para estructurar las salidas
class GeneratedQuestion(BaseModel):
    question: str = Field(description="A generated question as a sentence")

class GeneratedAnswer(BaseModel):
    answer: str = Field(description="A generated answer as a sentence")

class RelevantSources(BaseModel):
    sentences: list[str] = Field(description="Relevant sentences from the given context")

class QuestionEvolve(BaseModel):
    question: str = Field(description="Rewritten question with a question mark '?' at the end")

class GroundnessCheck(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    score: int = Field(description="Your evaluation and reasoning for the rating, from 1 to 5.", gt=0, lt=6)

class RelevanceCheck(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    score: int = Field(description="Your evaluation and reasoning for the rating, from 1 to 5.", gt=0, lt=6)


def load_all_documents(dataset_path: Path) -> List[Dict[str, str]]:
    """
    Carga todos los documentos markdown del dataset.
    
    Returns:
        Lista de diccionarios con 'source' y 'content'
    """
    md_files = list(dataset_path.glob("*.md"))
    print(f"📚 Encontrados {len(md_files)} archivos markdown")
    
    documents = []
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():  # Solo agregar si tiene contenido
                    documents.append({
                        'source': filepath.name,
                        'content': content
                    })
        except Exception as e:
            print(f"⚠️ Error leyendo {filepath.name}: {e}")
    
    return documents


def extract_paragraphs(content: str, min_length: int = 100) -> List[str]:
    """
    Extrae párrafos significativos de un documento.
    
    Args:
        content: Contenido del documento
        min_length: Longitud mínima de un párrafo válido
        
    Returns:
        Lista de párrafos
    """
    # Dividir por líneas vacías (párrafos)
    paragraphs = []
    current_para = []
    
    for line in content.split('\n'):
        line = line.strip()
        if line:
            # Ignorar líneas que son solo headers markdown
            if not line.startswith('#'):
                current_para.append(line)
        else:
            if current_para:
                para_text = ' '.join(current_para)
                if len(para_text) >= min_length:
                    paragraphs.append(para_text)
                current_para = []
    
    # Agregar el último párrafo si existe
    if current_para:
        para_text = ' '.join(current_para)
        if len(para_text) >= min_length:
            paragraphs.append(para_text)
    
    return paragraphs


def select_random_contexts(documents: List[Dict[str, str]], 
                          num_contexts: int = 100) -> List[Dict[str, str]]:
    """
    Selecciona párrafos aleatorios de los documentos.
    
    Args:
        documents: Lista de documentos
        num_contexts: Número de contextos a generar
        
    Returns:
        Lista de diccionarios con 'source' y 'context'
    """
    contexts = []
    
    # Extraer todos los párrafos con su fuente
    all_paragraphs = []
    for doc in documents:
        paragraphs = extract_paragraphs(doc['content'])
        for para in paragraphs:
            all_paragraphs.append({
                'source': doc['source'],
                'context': para
            })
    
    print(f"📝 Extraídos {len(all_paragraphs)} párrafos de todos los documentos")
    
    # Seleccionar aleatoriamente
    if len(all_paragraphs) < num_contexts:
        print(f"⚠️ Solo hay {len(all_paragraphs)} párrafos, se usarán todos")
        return all_paragraphs
    
    selected = random.sample(all_paragraphs, num_contexts)
    return selected


def init_llm(model_name: str = MODEL_NAME) -> Ollama:
    """
    Inicializa el LLM local (Ollama con Llama 3.1).
    
    Returns:
        Instancia del LLM
    """
    print(f"🤖 Inicializando {model_name}...")
    try:
        llm = Ollama(model=model_name, temperature=0.7)
        # Probar que funciona
        llm.invoke("Hi")
        print(f"✅ {model_name} inicializado correctamente")
        return llm
    except Exception as e:
        print(f"❌ Error inicializando {model_name}: {e}")
        print("\n💡 Asegúrate de que:")
        print(f"   1. Ollama esté corriendo")
        print(f"   2. El modelo {model_name} esté instalado: ollama pull {model_name}")
        sys.exit(1)


# Prompt Templates (basados en el script de evaluación)
QUESTION_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""<instructions>
Here is some context:
<context>
{context}
</context>
<role>You are a teacher creating a quiz from a given context.</role>
<task>
Your task is to generate 1 question that can be answered using the provided context, following these rules:

<rules>
1. The question should make sense to humans even when read without the given context.
2. The question should be fully answered from the given context.
3. The question should be framed from a part of context that contains important information. It can also be from tables, code, etc.
4. The answer to the question should not contain any links.
5. The question should be of moderate difficulty.
6. The question must be reasonable and must be understood and responded by humans.
7. Do not use phrases like 'provided context', etc. in the question.
8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
9. The question should not contain more than 10 words, make use of abbreviations wherever possible.
</rules>

To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above.

Output only the generated question with a "?" at the end, no other text or characters.
</task>
</instructions>""")

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<instructions>
<role>You are an experienced QA Engineer for building large language model applications.</role>
<task>
It is your task to generate an answer to the following question <question>{question}</question> only based on the <context>{context}</context></task>
The output should be only the answer generated from the context.

<rules>
1. Only use the given context as a source for generating the answer.
2. Be as precise as possible with answering the question.
3. Be concise in answering the question and only answer the question at hand rather than adding extra information.
</rules>

Only output the generated answer as a sentence. No extra characters.
</task>
</instructions>""")

SOURCE_PROMPT = PromptTemplate(
    input_variables=["full_context", "question"],
    template="""<instructions>
<role>You are an experienced QA Engineer for building large language model applications.</role>
<task>
Your task is to extract the relevant sentences from the given context that can potentially help answer the following question. You are not allowed to make any changes to the sentences from the context.

Here is the context:
<context>
{full_context}
</context>

<question>
{question}
</question>

Output only the relevant sentences you found, one sentence per line, without any extra characters or explanations.
</task>
</instructions>""")

EVOLVE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<instructions>
<role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

<task>
It is your task to rewrite the following question in a more indirect and compressed form, following these rules:

<rules>
1. Make the question more indirect
2. Make the question shorter
3. Use abbreviations if possible
4. The question should have between 7 and 10 words
</rules>

<question>
{question}
</question>

Your output should only be the rewritten question with a question mark "?" at the end. Do not provide any other explanation or text.
</task>
</instructions>""")

GROUNDEDNESS_CHECK_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<instructions>
<role>You are an experienced linguistics expert for building testsets for large language model applications.</role>

<task>
You will be given a context and a question related to that context.

Your task is to provide an evaluation of how well the given question can be answered using only the information provided in the context.

<rules>
Rate this on a scale from 1 to 5, where:
1 = The question cannot be answered at all based on the given context
2 = The context provides very little relevant information to answer the question
3 = The context provides some relevant information to partially answer the question
4 = The context provides substantial information to answer most aspects of the question
5 = The context provides all the information needed to fully and unambiguously answer the question
</rules>

First, read through the provided context carefully:

<context>
{context}
</context>

Then read the question:

<question>
{question}
</question>

Evaluate how well you think the question can be answered using only the context information. Provide your reasoning first in an <evaluation> section, explaining what relevant or missing information from the context led you to your evaluation score in only one sentence.
</task>
</instructions>""")

RELEVANCE_CHECK_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<instructions>
You will be given a question related to environmental issues, climate change, and sustainability. Your task is to evaluate how useful this question would be for a student learning about these topics.

To evaluate the usefulness of the question, consider the following criteria:

<rules>
1. Relevance: Is the question directly relevant to environmental issues? Questions that are too broad or unrelated to this domain should receive a lower rating.
2. Practicality: Does the question address a practical problem or use case that students might encounter? Theoretical or overly academic questions may be less useful.
3. Clarity: Is the question clear and well-defined? Ambiguous or vague questions are less useful.
4. Depth: Does the question require a substantive answer that demonstrates understanding of environmental topics? Surface-level questions may be less useful.
5. Applicability: Would answering this question provide insights or knowledge that could be applied to real-world environmental tasks? Questions with limited applicability should receive a lower rating.
</rules>

Here is the question:
<question>
{question}
</question>
</instructions>""")


def generate_question(llm: Ollama, context: str) -> str:
    """Genera una pregunta a partir de un contexto."""
    prompt = QUESTION_PROMPT.format(context=context)
    try:
        response = llm.invoke(prompt)
        # Extraer solo la pregunta, limpiar formato
        question = response.strip()
        if not question.endswith('?'):
            question += '?'
        return question
    except Exception as e:
        print(f"❌ Error generando pregunta: {e}")
        return None


def generate_answer(llm: Ollama, context: str, question: str) -> str:
    """Genera una respuesta a partir de un contexto y pregunta."""
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"❌ Error generando respuesta: {e}")
        return None


def extract_relevant_sources(llm: Ollama, full_context: str, question: str) -> List[str]:
    """Extrae las fuentes relevantes del contexto."""
    prompt = SOURCE_PROMPT.format(full_context=full_context, question=question)
    try:
        response = llm.invoke(prompt)
        # Dividir por líneas
        sentences = [s.strip() for s in response.split('\n') if s.strip()]
        return sentences
    except Exception as e:
        print(f"❌ Error extrayendo fuentes: {e}")
        return []


def evolve_question(llm: Ollama, question: str) -> str:
    """Evoluciona una pregunta para hacerla más indirecta y comprimida."""
    prompt = EVOLVE_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        evolved = response.strip()
        if not evolved.endswith('?'):
            evolved += '?'
        return evolved
    except Exception as e:
        print(f"❌ Error evolucionando pregunta: {e}")
        return question


def check_groundedness(llm: Ollama, context: str, question: str) -> tuple:
    """Verifica qué tan fundamentada está la pregunta en el contexto."""
    prompt = GROUNDEDNESS_CHECK_PROMPT.format(context=context, question=question)
    try:
        response = llm.invoke(prompt)
        # Parsear respuesta para obtener score y explicación
        lines = response.strip().split('\n')
        score = 0
        explanation = ""
        
        for line in lines:
            if 'score' in line.lower() or any(str(i) in line for i in range(1, 6)):
                # Buscar el número
                for i in range(1, 6):
                    if str(i) in line:
                        score = i
                        break
            else:
                explanation += line + " "
        
        return score, explanation.strip()
    except Exception as e:
        print(f"❌ Error verificando fundamentación: {e}")
        return 0, ""


def check_relevance(llm: Ollama, question: str) -> tuple:
    """Verifica la relevancia de la pregunta."""
    prompt = RELEVANCE_CHECK_PROMPT.format(question=question)
    try:
        response = llm.invoke(prompt)
        # Parsear respuesta para obtener score y explicación
        lines = response.strip().split('\n')
        score = 0
        explanation = ""
        
        for line in lines:
            if 'score' in line.lower() or any(str(i) in line for i in range(1, 6)):
                # Buscar el número
                for i in range(1, 6):
                    if str(i) in line:
                        score = i
                        break
            else:
                explanation += line + " "
        
        return score, explanation.strip()
    except Exception as e:
        print(f"❌ Error verificando relevancia: {e}")
        return 0, ""


def generate_test_question(llm: Ollama, context_data: Dict[str, str], 
                          context_index: int) -> Dict:
    """
    Genera una pregunta de prueba completa con todos sus componentes.
    
    Args:
        llm: Instancia del LLM
        context_data: Diccionario con 'source' y 'context'
        context_index: Índice del contexto (para tracking)
        
    Returns:
        Diccionario con toda la información de la pregunta de prueba
    """
    context = context_data['context']
    source = context_data['source']
    
    # 1. Generar pregunta
    print(f"  📝 Generando pregunta...")
    question = generate_question(llm, context)
    if not question:
        return None
    
    # 2. Generar respuesta
    print(f"  💬 Generando respuesta...")
    answer = generate_answer(llm, context, question)
    if not answer:
        return None
    
    # 3. Extraer fragmentos relevantes
    print(f"  🔍 Extrayendo fragmentos relevantes...")
    relevant_chunks = extract_relevant_sources(llm, context, question)
    
    # 4. Evolucionar pregunta
    print(f"  🔄 Evolucionando pregunta...")
    evolved_question = evolve_question(llm, question)
    
    # 5. Verificar fundamentación
    print(f"  ✅ Verificando fundamentación...")
    groundedness_score, groundedness_explanation = check_groundedness(
        llm, context, evolved_question
    )
    
    # 6. Verificar relevancia
    print(f"  ✅ Verificando relevancia...")
    relevance_score, relevance_explanation = check_relevance(llm, evolved_question)
    
    # Solo incluir preguntas con buenos scores
    if groundedness_score < 4 or relevance_score < 3:
        print(f"  ⚠️ Pregunta descartada (groundedness: {groundedness_score}, relevance: {relevance_score})")
        return None
    
    return {
        'id': context_index,
        'source': source,
        'context': context,
        'original_question': question,
        'evolved_question': evolved_question,
        'ground_truth_answer': answer,
        'relevant_chunks': relevant_chunks,
        'groundedness_score': groundedness_score,
        'groundedness_explanation': groundedness_explanation,
        'relevance_score': relevance_score,
        'relevance_explanation': relevance_explanation
    }


def generate_all_questions(num_questions: int = 100) -> List[Dict]:
    """
    Genera todas las preguntas de prueba.
    
    Args:
        num_questions: Número de preguntas a generar
        
    Returns:
        Lista de preguntas de prueba
    """
    print("\n" + "=" * 70)
    print("GENERANDO PREGUNTAS DE EVALUACIÓN PARA RAG")
    print("=" * 70 + "\n")
    
    # Cargar documentos
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "dataset"
    
    if not dataset_path.exists():
        print(f"❌ Error: No se encontró el directorio {dataset_path}")
        sys.exit(1)
    
    documents = load_all_documents(dataset_path)
    if not documents:
        print("❌ No se encontraron documentos")
        sys.exit(1)
    
    # Seleccionar contextos aleatorios
    # Generamos más de los necesarios para filtrar después
    contexts = select_random_contexts(documents, num_contexts=num_questions * 2)
    
    # Inicializar LLM
    llm = init_llm()
    
    # Generar preguntas
    test_questions = []
    attempted = 0
    
    print(f"\n🚀 Generando {num_questions} preguntas de prueba...\n")
    
    with tqdm(total=num_questions, desc="Progreso") as pbar:
        for context_data in contexts:
            if len(test_questions) >= num_questions:
                break
            
            attempted += 1
            print(f"\n--- Intento {attempted} (exitosos: {len(test_questions)}/{num_questions}) ---")
            print(f"📄 Fuente: {context_data['source']}")
            
            try:
                question_data = generate_test_question(llm, context_data, attempted)
                
                if question_data:
                    test_questions.append(question_data)
                    pbar.update(1)
                    print(f"  ✅ Pregunta aceptada!")
                    print(f"  Q: {question_data['evolved_question']}")
                    print(f"  A: {question_data['ground_truth_answer'][:100]}...")
                
            except Exception as e:
                print(f"  ❌ Error procesando contexto: {e}")
                continue
    
    print(f"\n✅ Generadas {len(test_questions)} preguntas de prueba")
    return test_questions


def save_test_questions(test_questions: List[Dict], output_file: str = OUTPUT_FILE):
    """Guarda las preguntas de prueba en un archivo JSON."""
    output_path = Path(__file__).parent / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Preguntas guardadas en: {output_path}")
    
    # Mostrar estadísticas
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)
    print(f"Total de preguntas: {len(test_questions)}")
    
    if test_questions:
        avg_groundedness = sum(q['groundedness_score'] for q in test_questions) / len(test_questions)
        avg_relevance = sum(q['relevance_score'] for q in test_questions) / len(test_questions)
        
        print(f"Groundedness promedio: {avg_groundedness:.2f}/5")
        print(f"Relevancia promedio: {avg_relevance:.2f}/5")
        
        print("\nFuentes únicas:")
        sources = set(q['source'] for q in test_questions)
        print(f"  {len(sources)} documentos únicos")


def load_test_questions(input_file: str = OUTPUT_FILE) -> List[Dict]:
    """
    Carga preguntas de prueba desde un archivo JSON.
    
    Returns:
        Lista de preguntas de prueba o None si no existe el archivo
    """
    input_path = Path(__file__).parent / input_file
    
    if not input_path.exists():
        return None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    print(f"✅ Cargadas {len(test_questions)} preguntas desde {input_path}")
    return test_questions


def main():
    """Función principal."""
    # Verificar si ya existen preguntas
    existing_questions = load_test_questions()
    
    if existing_questions:
        print("\n" + "=" * 70)
        print(f"✅ Archivo de preguntas encontrado: {OUTPUT_FILE}")
        print(f"   Contiene {len(existing_questions)} preguntas")
        print("=" * 70)
        
        response = input("\n¿Deseas regenerar las preguntas? (s/N): ").strip().lower()
        if response != 's':
            print("\n✅ Usando preguntas existentes")
            return existing_questions
    
    # Generar nuevas preguntas
    test_questions = generate_all_questions(num_questions=100)
    
    # Guardar
    save_test_questions(test_questions)
    
    return test_questions


if __name__ == "__main__":
    main()
