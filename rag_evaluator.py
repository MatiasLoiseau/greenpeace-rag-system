#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG System Evaluator with LLM Judge
==================================

Sistema completo para evaluar el rendimiento del RAG usando un LLM como juez.
Compara las respuestas del RAG con el ground truth y calcula métricas de evaluación.

Features:
- LLM juez para evaluación semántica de respuestas
- Métricas de precisión, recall, accuracy y F1-score
- Evaluación de relevancia de fuentes recuperadas
- Análisis detallado por categorías
- Reportes completos de evaluación

Usage:
    python rag_evaluator.py

Requirements:
    - test_dataset.json generado por rag_test_generator.py
    - Sistema RAG funcionando (rag_qa_system.py)
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Local imports
from rag_qa_system import RAGQuestionAnswering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Resultado de evaluación para una pregunta individual."""
    question_id: str
    question: str
    ground_truth_answer: str
    rag_answer: str
    ground_truth_source: str
    rag_sources: List[str]
    
    # Scores del LLM juez (0-10)
    semantic_similarity_score: float
    factual_accuracy_score: float
    completeness_score: float
    relevance_score: float
    
    # Métricas binarias
    source_match: bool
    category_match: bool
    
    # Metadata
    category: str
    evaluation_time: float
    error_message: Optional[str] = None


class RAGEvaluator:
    """Evaluador completo del sistema RAG con LLM juez."""
    
    def __init__(self, 
                 test_dataset_path: str = "test_dataset.json",
                 rag_system: Optional[RAGQuestionAnswering] = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            test_dataset_path: Path to the test dataset JSON
            rag_system: Pre-initialized RAG system (optional)
        """
        load_dotenv()
        
        self.test_dataset_path = Path(test_dataset_path)
        
        # Load test dataset
        self.test_dataset = self._load_test_dataset()
        logger.info(f"Loaded {len(self.test_dataset)} test questions")
        
        # Initialize RAG system
        if rag_system is None:
            logger.info("Initializing RAG system...")
            self.rag_system = RAGQuestionAnswering()
        else:
            self.rag_system = rag_system
        
        # Initialize local LLM judge
        try:
            self.llm_judge = OllamaLLM(
                model="llama3.2",  # Usar llama3.2 para evaluación
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            logger.info("Local Ollama judge model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama judge: {e}")
            logger.info("Make sure Ollama is running with: ollama serve")
            logger.info("And the model is installed with: ollama pull llama3.2")
            raise
        
        # Evaluation prompt will be built dynamically
        pass
    
    def _load_test_dataset(self) -> List[Dict]:
        """Carga el dataset de prueba desde JSON."""
        try:
            with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            return dataset
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            raise
    
    def _parse_llm_evaluation(self, evaluation_text: str) -> Dict[str, float]:
        """
        Parsea la respuesta del LLM juez y extrae las puntuaciones.
        
        Args:
            evaluation_text: Texto de respuesta del LLM juez
            
        Returns:
            Diccionario con las puntuaciones
        """
        scores = {
            'semantic_similarity': 0.0,
            'factual_accuracy': 0.0,
            'completeness': 0.0,
            'relevance': 0.0,
            'justification': ''
        }
        
        try:
            lines = evaluation_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if 'SIMILARITY' in key or 'SIMILITUD' in key:
                        scores['semantic_similarity'] = float(value)
                    elif 'ACCURACY' in key or 'FACTUAL' in key or 'PRECISION' in key:
                        scores['factual_accuracy'] = float(value)
                    elif 'COMPLETENESS' in key or 'COMPLETITUD' in key:
                        scores['completeness'] = float(value)
                    elif 'RELEVANCE' in key or 'RELEVANCIA' in key:
                        scores['relevance'] = float(value)
                    elif 'JUSTIFICATION' in key or 'JUSTIFICACION' in key:
                        scores['justification'] = value
            
        except Exception as e:
            logger.warning(f"Error parsing LLM evaluation: {e}")
            # Return default scores if parsing fails
        
        return scores
    
    def _build_evaluation_prompt(self, question: str, ground_truth: str, 
                               rag_answer: str, original_context: str) -> str:
        """Construye el prompt de evaluación para el LLM juez."""
        return f"""You are an expert evaluator of question-answering systems. Your task is to compare two answers to a question and evaluate how good the system response is compared to the reference answer.

QUESTION: {question}

REFERENCE ANSWER (Ground Truth): {ground_truth}

RAG SYSTEM ANSWER: {rag_answer}

ORIGINAL CONTEXT: {original_context}

Evaluate the RAG system answer on the following dimensions and provide a score from 0 to 10 for each:

1. SEMANTIC SIMILARITY: How similar is the meaning of both answers?
   - 10: Identical or nearly identical meaning
   - 7-9: Very similar, small differences
   - 4-6: Moderately similar, some important differences
   - 1-3: Little similarity, significant differences
   - 0: Completely different

2. FACTUAL ACCURACY: Does the system answer contain correct information?
   - 10: All facts are correct
   - 7-9: Most facts correct, minor errors
   - 4-6: Some facts correct, some errors
   - 1-3: Few facts correct, many errors
   - 0: Incorrect or unrelated information

3. COMPLETENESS: Does the system answer cover all important aspects?
   - 10: Covers all aspects of the reference answer
   - 7-9: Covers most important aspects
   - 4-6: Covers some aspects, moderate missing information
   - 1-3: Covers few aspects, much missing information
   - 0: Doesn't cover main aspects

4. RELEVANCE: How relevant and useful is the answer for the question?
   - 10: Completely relevant and useful
   - 7-9: Very relevant with useful information
   - 4-6: Moderately relevant
   - 1-3: Little relevance
   - 0: Irrelevant or doesn't answer the question

RESPONSE FORMAT (respond EXACTLY in this format):
SEMANTIC_SIMILARITY: [score]
FACTUAL_ACCURACY: [score]
COMPLETENESS: [score]
RELEVANCE: [score]
JUSTIFICATION: [brief explanation in 1-2 sentences]

Example:
SEMANTIC_SIMILARITY: 8
FACTUAL_ACCURACY: 9
COMPLETENESS: 7
RELEVANCE: 8
JUSTIFICATION: The system answer correctly captures the main concepts but omits some specific details mentioned in the reference."""
    
    def _evaluate_source_match(self, ground_truth_source: str, 
                             rag_sources: List[str]) -> Tuple[bool, bool]:
        """
        Evalúa si las fuentes coinciden.
        
        Args:
            ground_truth_source: Archivo fuente de la respuesta de referencia
            rag_sources: Lista de fuentes devueltas por el RAG
            
        Returns:
            Tuple (exact_match, category_match)
        """
        # Exact file match
        source_files = [src.get('source_file', '') for src in rag_sources if isinstance(src, dict)]
        exact_match = ground_truth_source in source_files
        
        # Category match (same category as ground truth)
        # This would require loading metadata to check categories
        # For now, we'll use a simple heuristic
        category_match = exact_match  # Simplified
        
        return exact_match, category_match
    
    def evaluate_single_question(self, test_item: Dict) -> EvaluationResult:
        """
        Evalúa una sola pregunta del dataset de prueba.
        
        Args:
            test_item: Elemento del dataset de prueba
            
        Returns:
            Resultado de la evaluación
        """
        start_time = datetime.now()
        
        try:
            question = test_item['question']
            question_id = test_item['id']
            
            logger.info(f"Evaluating question {question_id}: {question[:50]}...")
            
            # Get RAG response
            rag_response = self.rag_system.ask_question(question, show_sources=True)
            
            if rag_response.get('error'):
                return EvaluationResult(
                    question_id=question_id,
                    question=question,
                    ground_truth_answer=test_item['answer'],
                    rag_answer="ERROR",
                    ground_truth_source=test_item['source_file'],
                    rag_sources=[],
                    semantic_similarity_score=0.0,
                    factual_accuracy_score=0.0,
                    completeness_score=0.0,
                    relevance_score=0.0,
                    source_match=False,
                    category_match=False,
                    category=test_item['category'],
                    evaluation_time=(datetime.now() - start_time).total_seconds(),
                    error_message=rag_response['answer']
                )
            
            # Get LLM evaluation
            prompt = self._build_evaluation_prompt(
                question=question,
                ground_truth=test_item['answer'],
                rag_answer=rag_response['answer'],
                original_context=test_item['source_paragraph'][:500]  # Limit context length
            )
            
            evaluation_text = self.llm_judge.invoke(prompt)
            
            # Parse LLM evaluation
            scores = self._parse_llm_evaluation(evaluation_text)
            
            # Evaluate source matching
            source_match, category_match = self._evaluate_source_match(
                test_item['source_file'],
                rag_response.get('sources', [])
            )
            
            # Create evaluation result
            result = EvaluationResult(
                question_id=question_id,
                question=question,
                ground_truth_answer=test_item['answer'],
                rag_answer=rag_response['answer'],
                ground_truth_source=test_item['source_file'],
                rag_sources=rag_response.get('sources', []),
                semantic_similarity_score=scores['semantic_similarity'],
                factual_accuracy_score=scores['factual_accuracy'],
                completeness_score=scores['completeness'],
                relevance_score=scores['relevance'],
                source_match=source_match,
                category_match=category_match,
                category=test_item['category'],
                evaluation_time=(datetime.now() - start_time).total_seconds()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {e}")
            return EvaluationResult(
                question_id=question_id,
                question=test_item.get('question', ''),
                ground_truth_answer=test_item.get('answer', ''),
                rag_answer="ERROR",
                ground_truth_source=test_item.get('source_file', ''),
                rag_sources=[],
                semantic_similarity_score=0.0,
                factual_accuracy_score=0.0,
                completeness_score=0.0,
                relevance_score=0.0,
                source_match=False,
                category_match=False,
                category=test_item.get('category', 'Unknown'),
                evaluation_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    def evaluate_dataset(self, max_questions: Optional[int] = None) -> List[EvaluationResult]:
        """
        Evalúa el dataset completo de preguntas.
        
        Args:
            max_questions: Número máximo de preguntas a evaluar (None para todas)
            
        Returns:
            Lista de resultados de evaluación
        """
        logger.info(f"Starting evaluation of RAG system...")
        
        questions_to_evaluate = self.test_dataset
        if max_questions:
            questions_to_evaluate = questions_to_evaluate[:max_questions]
        
        logger.info(f"Evaluating {len(questions_to_evaluate)} questions...")
        
        results = []
        for i, test_item in enumerate(questions_to_evaluate):
            try:
                result = self.evaluate_single_question(test_item)
                results.append(result)
                
                # Log progress every 10 questions
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(questions_to_evaluate)} evaluations")
                    
            except Exception as e:
                logger.error(f"Failed to evaluate question {i+1}: {e}")
                continue
        
        logger.info(f"Evaluation completed! Processed {len(results)} questions")
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """
        Calcula métricas de evaluación basadas en los resultados.
        
        Args:
            results: Lista de resultados de evaluación
            
        Returns:
            Diccionario con métricas calculadas
        """
        if not results:
            return {'error': 'No results to calculate metrics'}
        
        # Filter out error results
        valid_results = [r for r in results if r.error_message is None]
        
        if not valid_results:
            return {'error': 'No valid results to calculate metrics'}
        
        # Calculate average scores
        semantic_scores = [r.semantic_similarity_score for r in valid_results]
        factual_scores = [r.factual_accuracy_score for r in valid_results]
        completeness_scores = [r.completeness_score for r in valid_results]
        relevance_scores = [r.relevance_score for r in valid_results]
        
        # Source matching metrics
        source_matches = sum(1 for r in valid_results if r.source_match)
        category_matches = sum(1 for r in valid_results if r.category_match)
        
        # Overall score (weighted average)
        overall_scores = []
        for r in valid_results:
            weighted_score = (
                r.semantic_similarity_score * 0.3 +
                r.factual_accuracy_score * 0.3 +
                r.completeness_score * 0.2 +
                r.relevance_score * 0.2
            )
            overall_scores.append(weighted_score)
        
        # Calculate metrics
        total_valid = len(valid_results)
        total_questions = len(results)
        
        metrics = {
            'total_questions_evaluated': total_questions,
            'valid_evaluations': total_valid,
            'error_rate': (total_questions - total_valid) / total_questions if total_questions > 0 else 0,
            
            # Score metrics (0-10 scale)
            'semantic_similarity': {
                'mean': np.mean(semantic_scores),
                'std': np.std(semantic_scores),
                'min': min(semantic_scores),
                'max': max(semantic_scores)
            },
            'factual_accuracy': {
                'mean': np.mean(factual_scores),
                'std': np.std(factual_scores),
                'min': min(factual_scores),
                'max': max(factual_scores)
            },
            'completeness': {
                'mean': np.mean(completeness_scores),
                'std': np.std(completeness_scores),
                'min': min(completeness_scores),
                'max': max(completeness_scores)
            },
            'relevance': {
                'mean': np.mean(relevance_scores),
                'std': np.std(relevance_scores),
                'min': min(relevance_scores),
                'max': max(relevance_scores)
            },
            'overall_score': {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores)
            },
            
            # Source matching metrics
            'source_precision': source_matches / total_valid if total_valid > 0 else 0,
            'source_recall': source_matches / total_valid if total_valid > 0 else 0,  # Simplified
            'category_accuracy': category_matches / total_valid if total_valid > 0 else 0,
            
            # Performance metrics
            'average_evaluation_time': np.mean([r.evaluation_time for r in valid_results]),
            'total_evaluation_time': sum(r.evaluation_time for r in results),
            
            # Score distributions (percentage in each range)
            'score_distribution': self._calculate_score_distribution(overall_scores)
        }
        
        return metrics
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict:
        """Calcula la distribución de puntuaciones."""
        if not scores:
            return {}
        
        ranges = {
            'excellent_9_10': sum(1 for s in scores if 9 <= s <= 10),
            'good_7_8': sum(1 for s in scores if 7 <= s < 9),
            'fair_5_6': sum(1 for s in scores if 5 <= s < 7),
            'poor_3_4': sum(1 for s in scores if 3 <= s < 5),
            'very_poor_0_2': sum(1 for s in scores if 0 <= s < 3)
        }
        
        total = len(scores)
        return {k: v / total * 100 for k, v in ranges.items()}
    
    def generate_report(self, results: List[EvaluationResult], 
                       output_path: str = "rag_evaluation_report.json") -> Dict:
        """
        Genera un reporte completo de la evaluación.
        
        Args:
            results: Resultados de la evaluación
            output_path: Ruta para guardar el reporte
            
        Returns:
            Diccionario con el reporte completo
        """
        metrics = self.calculate_metrics(results)
        
        # Category breakdown
        category_breakdown = {}
        for result in results:
            if result.error_message is None:
                cat = result.category
                if cat not in category_breakdown:
                    category_breakdown[cat] = {
                        'count': 0,
                        'semantic_similarity': [],
                        'factual_accuracy': [],
                        'completeness': [],
                        'relevance': [],
                        'source_matches': 0
                    }
                
                category_breakdown[cat]['count'] += 1
                category_breakdown[cat]['semantic_similarity'].append(result.semantic_similarity_score)
                category_breakdown[cat]['factual_accuracy'].append(result.factual_accuracy_score)
                category_breakdown[cat]['completeness'].append(result.completeness_score)
                category_breakdown[cat]['relevance'].append(result.relevance_score)
                if result.source_match:
                    category_breakdown[cat]['source_matches'] += 1
        
        # Calculate averages for each category
        for cat, data in category_breakdown.items():
            if data['count'] > 0:
                data['avg_semantic_similarity'] = np.mean(data['semantic_similarity'])
                data['avg_factual_accuracy'] = np.mean(data['factual_accuracy'])
                data['avg_completeness'] = np.mean(data['completeness'])
                data['avg_relevance'] = np.mean(data['relevance'])
                data['source_precision'] = data['source_matches'] / data['count']
        
        # Create full report
        report = {
            'evaluation_metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'test_dataset_path': str(self.test_dataset_path),
                'total_test_questions': len(self.test_dataset),
                'questions_evaluated': len(results)
            },
            'overall_metrics': metrics,
            'category_breakdown': category_breakdown,
            'sample_results': [
                {
                    'question_id': r.question_id,
                    'question': r.question,
                    'ground_truth': r.ground_truth_answer[:200] + "..." if len(r.ground_truth_answer) > 200 else r.ground_truth_answer,
                    'rag_answer': r.rag_answer[:200] + "..." if len(r.rag_answer) > 200 else r.rag_answer,
                    'scores': {
                        'semantic_similarity': r.semantic_similarity_score,
                        'factual_accuracy': r.factual_accuracy_score,
                        'completeness': r.completeness_score,
                        'relevance': r.relevance_score
                    },
                    'source_match': r.source_match,
                    'category': r.category
                }
                for r in results[:5] if r.error_message is None  # First 5 valid results
            ]
        }
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
        
        return report


def main():
    """Función principal para ejecutar la evaluación del RAG."""
    logger.info("=== RAG System Evaluator ===")
    
    # Check if test dataset exists
    test_dataset_path = "test_dataset.json"
    if not os.path.exists(test_dataset_path):
        logger.error(f"Test dataset not found at {test_dataset_path}")
        logger.error("Please run rag_test_generator.py first to generate the test dataset")
        return
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator(test_dataset_path)
        
        # Run evaluation (limit to 50 questions for demo)
        max_questions = 50  # Change to None for full evaluation
        results = evaluator.evaluate_dataset(max_questions=max_questions)
        
        # Generate and display report
        report = evaluator.generate_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        metrics = report['overall_metrics']
        print(f"Questions evaluated: {metrics['valid_evaluations']}/{metrics['total_questions_evaluated']}")
        print(f"Error rate: {metrics['error_rate']:.2%}")
        print(f"\nOverall Performance Scores (0-10 scale):")
        print(f"  Semantic Similarity: {metrics['semantic_similarity']['mean']:.2f} ± {metrics['semantic_similarity']['std']:.2f}")
        print(f"  Factual Accuracy: {metrics['factual_accuracy']['mean']:.2f} ± {metrics['factual_accuracy']['std']:.2f}")
        print(f"  Completeness: {metrics['completeness']['mean']:.2f} ± {metrics['completeness']['std']:.2f}")
        print(f"  Relevance: {metrics['relevance']['mean']:.2f} ± {metrics['relevance']['std']:.2f}")
        print(f"  Overall Score: {metrics['overall_score']['mean']:.2f} ± {metrics['overall_score']['std']:.2f}")
        
        print(f"\nSource Matching:")
        print(f"  Source Precision: {metrics['source_precision']:.2%}")
        print(f"  Category Accuracy: {metrics['category_accuracy']:.2%}")
        
        print(f"\nPerformance:")
        print(f"  Average evaluation time: {metrics['average_evaluation_time']:.2f} seconds")
        print(f"  Total evaluation time: {metrics['total_evaluation_time']:.1f} seconds")
        
        # Score distribution
        dist = metrics['score_distribution']
        print(f"\nScore Distribution:")
        print(f"  Excellent (9-10): {dist.get('excellent_9_10', 0):.1f}%")
        print(f"  Good (7-8): {dist.get('good_7_8', 0):.1f}%")
        print(f"  Fair (5-6): {dist.get('fair_5_6', 0):.1f}%")
        print(f"  Poor (3-4): {dist.get('poor_3_4', 0):.1f}%")
        print(f"  Very Poor (0-2): {dist.get('very_poor_0_2', 0):.1f}%")
        
        print(f"\n✅ Full report saved as 'rag_evaluation_report.json'")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()