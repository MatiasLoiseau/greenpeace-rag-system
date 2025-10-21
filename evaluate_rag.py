#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para evaluar el RAG de Greenpeace usando preguntas de prueba generadas.
Mide: correctitud, relevancia, fundamentación y relevancia de recuperación.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# Configuración
MODEL_NAME = "llama3.2"
QUESTIONS_FILE = "test_questions.json"
RESULTS_DIR = "evaluation_results"

# Modelos Pydantic para las evaluaciones
class CorrectnessGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    is_correct: bool = Field(description="True if the answer is correct, False otherwise.")

class RelevanceGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    is_relevant: bool = Field(description="True if the answer addresses the question, False otherwise")

class GroundedGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    is_grounded: bool = Field(description="True if the answer is grounded on the documents, False otherwise")

class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score.")
    is_relevant: bool = Field(description="True if the retrieved documents are relevant to the question, False otherwise.")


# Prompt Templates para evaluación
CORRECTNESS_PROMPT = PromptTemplate(
    input_variables=["question", "ground_truth_answer", "answer"],
    template="""<instructions>
<role>You are a teacher grading a quiz.</role>
<task>
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

<rules>
Here is the grade criteria to follow:
1. Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
2. Ensure that the student answer does not contain any conflicting statements.
3. It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Here is the correctness criteria:
1. A correctness value of True means that the student's answer meets all of the criteria.
2. A correctness value of False means that the student's answer does not meet all of the criteria.
</rules>

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset.

<question>
QUESTION: {question}
</question>
<ground_truth_answer>
GROUND TRUTH ANSWER: {ground_truth_answer}
</ground_truth_answer>
<answer>
STUDENT ANSWER: {answer}
</answer>
</task>
</instructions>""")

RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""<instructions>
<role>You are a teacher grading a quiz.</role>
<task>
You will be given a QUESTION and a STUDENT ANSWER.

<rules>
Here is the grade criteria to follow:
1. Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
2. Ensure the STUDENT ANSWER helps to answer the QUESTION

Here is the relevance criteria:
1. A relevance value of True means that the student's answer meets all of the criteria.
2. A relevance value of False means that the student's answer does not meet all of the criteria.
</rules>

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset.

<question>
QUESTION: {question}
</question>
<answer>
STUDENT ANSWER: {answer}
</answer>
</task>
</instructions>""")

GROUNDED_PROMPT = PromptTemplate(
    input_variables=["doc_string", "answer"],
    template="""<instructions>
<role>You are a teacher grading a quiz.</role>

<task>
You will be given FACTS and a STUDENT ANSWER.

<rules>
Here is the grade criteria to follow:
1. Ensure the STUDENT ANSWER is grounded in the FACTS.
2. Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

And this is the grounded criteria:
1. A grounded value of True means that the student's answer meets all of the criteria.
2. A grounded value of False means that the student's answer does not meet all of the criteria.
</rules>

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset.

<facts>
FACTS: {doc_string}
</facts>

<answer>
STUDENT ANSWER: {answer}
</answer>
</task>
</instructions>""")

RETRIEVAL_RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["doc_string", "question"],
    template="""<instructions>
<role>You are a teacher grading a quiz.</role>
<task>
Your task is to assess the relevance of a given QUESTION based on the FACTS provided by the student.

<rules>
Here is the grade criteria to follow:
1. Your goal is to identify FACTS that are completely unrelated to the QUESTION
2. If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
3. It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

And this is the relevance criteria:
1. A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
2. A relevance value of False means that the FACTS are completely unrelated to the QUESTION.
</rules>

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. Avoid simply stating the correct answer at the outset.

Here are the facts:
<facts>
{doc_string}
</facts>

And this is the question:
<question>{question}</question>
</task>
</instructions>""")


class RAGEvaluator:
    """Clase para evaluar el RAG."""
    
    def __init__(self, chroma_path: str = "./chroma_db", model_name: str = MODEL_NAME):
        """
        Inicializa el evaluador.
        
        Args:
            chroma_path: Ruta a la base de datos ChromaDB
            model_name: Nombre del modelo LLM a usar
        """
        print("🔧 Inicializando evaluador...")
        
        # Cargar vector store
        print("  📚 Cargando vector store...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name="greenpeace_docs"
        )
        
        # Inicializar LLM para generación
        print(f"  🤖 Inicializando LLM para generación ({model_name})...")
        self.llm = Ollama(model=model_name, temperature=0.3)
        
        # Inicializar LLM para evaluación (puede ser el mismo o diferente)
        print(f"  🤖 Inicializando LLM para evaluación ({model_name})...")
        self.eval_llm = Ollama(model=model_name, temperature=0.1)
        
        print("✅ Evaluador inicializado\n")
    
    def retrieve(self, query: str, k: int = 3) -> List:
        """
        Recupera documentos relevantes para una consulta.
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos
        """
        return self.vectorstore.similarity_search(query, k=k)
    
    def generate(self, query: str, relevant_documents: List) -> str:
        """
        Genera una respuesta usando RAG.
        
        Args:
            query: Consulta del usuario
            relevant_documents: Documentos relevantes
            
        Returns:
            Respuesta generada
        """
        # Crear contexto a partir de los documentos
        documents = "\n\n".join(doc.page_content for doc in relevant_documents)
        
        # Prompt simple para generación
        prompt = f"""Question: {query}

Documents:
{documents}

Please answer the question based only on the provided documents. Be concise and accurate."""
        
        # Generar respuesta
        answer = self.llm.invoke(prompt)
        return answer.strip()
    
    def answer(self, query: str, k: int = 3) -> Tuple[str, List]:
        """
        Genera una respuesta completa usando RAG.
        
        Args:
            query: Consulta del usuario
            k: Número de documentos a recuperar
            
        Returns:
            Tupla (respuesta, documentos_recuperados)
        """
        relevant_documents = self.retrieve(query, k=k)
        answer = self.generate(query, relevant_documents)
        return answer, relevant_documents
    
    def evaluate_correctness(self, question: str, ground_truth: str, answer: str) -> Dict:
        """Evalúa la correctitud de una respuesta."""
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth_answer=ground_truth,
            answer=answer
        )
        
        try:
            response = self.eval_llm.invoke(prompt)
            # Parsear respuesta
            is_correct = 'true' in response.lower() and 'false' not in response.lower().split('true')[0]
            
            return {
                'is_correct': is_correct,
                'explanation': response.strip()
            }
        except Exception as e:
            print(f"❌ Error evaluando correctitud: {e}")
            return {'is_correct': False, 'explanation': str(e)}
    
    def evaluate_relevance(self, question: str, answer: str) -> Dict:
        """Evalúa la relevancia de una respuesta."""
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        
        try:
            response = self.eval_llm.invoke(prompt)
            # Parsear respuesta
            is_relevant = 'true' in response.lower() and 'false' not in response.lower().split('true')[0]
            
            return {
                'is_relevant': is_relevant,
                'explanation': response.strip()
            }
        except Exception as e:
            print(f"❌ Error evaluando relevancia: {e}")
            return {'is_relevant': False, 'explanation': str(e)}
    
    def evaluate_grounding(self, answer: str, documents: List) -> Dict:
        """Evalúa si la respuesta está fundamentada en los documentos."""
        doc_string = "\n\n".join(doc.page_content for doc in documents)
        prompt = GROUNDED_PROMPT.format(doc_string=doc_string, answer=answer)
        
        try:
            response = self.eval_llm.invoke(prompt)
            # Parsear respuesta
            is_grounded = 'true' in response.lower() and 'false' not in response.lower().split('true')[0]
            
            return {
                'is_grounded': is_grounded,
                'explanation': response.strip()
            }
        except Exception as e:
            print(f"❌ Error evaluando fundamentación: {e}")
            return {'is_grounded': False, 'explanation': str(e)}
    
    def evaluate_retrieval_relevance(self, question: str, documents: List) -> Dict:
        """Evalúa la relevancia de los documentos recuperados."""
        doc_string = "\n\n".join(doc.page_content for doc in documents)
        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(doc_string=doc_string, question=question)
        
        try:
            response = self.eval_llm.invoke(prompt)
            # Parsear respuesta
            is_relevant = 'true' in response.lower() and 'false' not in response.lower().split('true')[0]
            
            return {
                'is_relevant': is_relevant,
                'explanation': response.strip()
            }
        except Exception as e:
            print(f"❌ Error evaluando relevancia de recuperación: {e}")
            return {'is_relevant': False, 'explanation': str(e)}
    
    def evaluate_single_question(self, question_data: Dict, k: int = 3) -> Dict:
        """
        Evalúa una pregunta completa.
        
        Args:
            question_data: Datos de la pregunta de prueba
            k: Número de documentos a recuperar
            
        Returns:
            Diccionario con resultados de evaluación
        """
        question = question_data['evolved_question']
        ground_truth = question_data['ground_truth_answer']
        
        # Generar respuesta con RAG
        answer, retrieved_docs = self.answer(question, k=k)
        
        # Evaluar
        correctness = self.evaluate_correctness(question, ground_truth, answer)
        relevance = self.evaluate_relevance(question, answer)
        grounding = self.evaluate_grounding(answer, retrieved_docs)
        retrieval_relevance = self.evaluate_retrieval_relevance(question, retrieved_docs)
        
        return {
            'question_id': question_data['id'],
            'question': question,
            'ground_truth': ground_truth,
            'generated_answer': answer,
            'retrieved_docs': [doc.page_content[:200] + "..." for doc in retrieved_docs],
            'correctness': correctness,
            'relevance': relevance,
            'grounding': grounding,
            'retrieval_relevance': retrieval_relevance
        }


def load_test_questions(questions_file: str = QUESTIONS_FILE) -> List[Dict]:
    """Carga las preguntas de prueba."""
    questions_path = Path(__file__).parent / questions_file
    
    if not questions_path.exists():
        print(f"❌ No se encontró el archivo {questions_file}")
        print("\n💡 Primero ejecuta 'python generate_test_questions.py' para crear las preguntas")
        return None
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    print(f"✅ Cargadas {len(test_questions)} preguntas de prueba\n")
    return test_questions


def evaluate_rag(test_questions: List[Dict], 
                num_questions: int = None,
                k: int = 3,
                chroma_path: str = "./chroma_db") -> Dict:
    """
    Evalúa el RAG con las preguntas de prueba.
    
    Args:
        test_questions: Lista de preguntas de prueba
        num_questions: Número de preguntas a evaluar (None = todas)
        k: Número de documentos a recuperar
        chroma_path: Ruta a ChromaDB
        
    Returns:
        Diccionario con resultados de evaluación
    """
    print("=" * 70)
    print("EVALUACIÓN DEL RAG")
    print("=" * 70 + "\n")
    
    # Inicializar evaluador
    evaluator = RAGEvaluator(chroma_path=chroma_path)
    
    # Seleccionar preguntas a evaluar
    if num_questions:
        questions_to_eval = test_questions[:num_questions]
    else:
        questions_to_eval = test_questions
    
    print(f"📊 Evaluando {len(questions_to_eval)} preguntas...\n")
    
    # Evaluar cada pregunta
    results = []
    
    for i, question_data in enumerate(tqdm(questions_to_eval, desc="Evaluando"), 1):
        try:
            result = evaluator.evaluate_single_question(question_data, k=k)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluando pregunta {i}: {e}")
            continue
    
    # Calcular métricas agregadas
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    
    total = len(results)
    if total == 0:
        print("❌ No se pudieron evaluar preguntas")
        return None
    
    correctness_count = sum(1 for r in results if r['correctness']['is_correct'])
    relevance_count = sum(1 for r in results if r['relevance']['is_relevant'])
    grounding_count = sum(1 for r in results if r['grounding']['is_grounded'])
    retrieval_count = sum(1 for r in results if r['retrieval_relevance']['is_relevant'])
    
    metrics = {
        'total_questions': total,
        'correctness': {
            'score': correctness_count / total,
            'count': correctness_count
        },
        'relevance': {
            'score': relevance_count / total,
            'count': relevance_count
        },
        'grounding': {
            'score': grounding_count / total,
            'count': grounding_count
        },
        'retrieval_relevance': {
            'score': retrieval_count / total,
            'count': retrieval_count
        }
    }
    
    print(f"\n📈 Métricas Globales:")
    print(f"   Total de preguntas: {total}")
    print(f"   Correctitud: {metrics['correctness']['score']:.2%} ({correctness_count}/{total})")
    print(f"   Relevancia: {metrics['relevance']['score']:.2%} ({relevance_count}/{total})")
    print(f"   Fundamentación: {metrics['grounding']['score']:.2%} ({grounding_count}/{total})")
    print(f"   Relevancia de Recuperación: {metrics['retrieval_relevance']['score']:.2%} ({retrieval_count}/{total})")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'k': k,
            'num_questions': total
        },
        'metrics': metrics,
        'results': results
    }


def save_evaluation_results(evaluation_results: Dict, 
                           results_dir: str = RESULTS_DIR):
    """Guarda los resultados de evaluación."""
    # Crear directorio si no existe
    results_path = Path(__file__).parent / results_dir
    results_path.mkdir(exist_ok=True)
    
    # Crear nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}.json"
    filepath = results_path / filename
    
    # Guardar
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {filepath}")
    
    # También guardar un resumen legible
    summary_file = results_path / f"summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMEN DE EVALUACIÓN DEL RAG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha: {evaluation_results['timestamp']}\n")
        f.write(f"Configuración: k={evaluation_results['config']['k']}\n")
        f.write(f"Preguntas evaluadas: {evaluation_results['config']['num_questions']}\n\n")
        
        metrics = evaluation_results['metrics']
        f.write("MÉTRICAS:\n")
        f.write(f"  Correctitud: {metrics['correctness']['score']:.2%}\n")
        f.write(f"  Relevancia: {metrics['relevance']['score']:.2%}\n")
        f.write(f"  Fundamentación: {metrics['grounding']['score']:.2%}\n")
        f.write(f"  Relevancia de Recuperación: {metrics['retrieval_relevance']['score']:.2%}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("DETALLES POR PREGUNTA\n")
        f.write("=" * 70 + "\n\n")
        
        for i, result in enumerate(evaluation_results['results'], 1):
            f.write(f"\n--- Pregunta {i} ---\n")
            f.write(f"Q: {result['question']}\n")
            f.write(f"Ground Truth: {result['ground_truth'][:100]}...\n")
            f.write(f"Generated: {result['generated_answer'][:100]}...\n")
            f.write(f"Correctitud: {'✅' if result['correctness']['is_correct'] else '❌'}\n")
            f.write(f"Relevancia: {'✅' if result['relevance']['is_relevant'] else '❌'}\n")
            f.write(f"Fundamentación: {'✅' if result['grounding']['is_grounded'] else '❌'}\n")
            f.write(f"Recuperación: {'✅' if result['retrieval_relevance']['is_relevant'] else '❌'}\n")
    
    print(f"📄 Resumen guardado en: {summary_file}")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar el RAG de Greenpeace')
    parser.add_argument('-n', '--num-questions', type=int, default=None,
                       help='Número de preguntas a evaluar (default: todas)')
    parser.add_argument('-k', '--num-docs', type=int, default=3,
                       help='Número de documentos a recuperar (default: 3)')
    parser.add_argument('--chroma-path', type=str, default='./chroma_db',
                       help='Ruta a ChromaDB (default: ./chroma_db)')
    
    args = parser.parse_args()
    
    # Cargar preguntas
    test_questions = load_test_questions()
    
    if not test_questions:
        # Si no existen preguntas, intentar generarlas
        print("\n💡 Generando preguntas de prueba primero...")
        from generate_test_questions import main as generate_main
        test_questions = generate_main()
        
        if not test_questions:
            print("❌ No se pudieron generar preguntas")
            sys.exit(1)
    
    # Evaluar RAG
    evaluation_results = evaluate_rag(
        test_questions,
        num_questions=args.num_questions,
        k=args.num_docs,
        chroma_path=args.chroma_path
    )
    
    if not evaluation_results:
        print("❌ La evaluación falló")
        sys.exit(1)
    
    # Guardar resultados
    save_evaluation_results(evaluation_results)
    
    print("\n✅ Evaluación completada exitosamente")


if __name__ == "__main__":
    main()
