#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Testing Suite - Complete Evaluation Pipeline
==============================================

Sistema completo de testing para evaluar el sistema RAG de Greenpeace.
Ejecuta todo el pipeline: generación de dataset de prueba, evaluación y reportes.

Features:
- Generación automática de dataset de prueba
- Evaluación completa con LLM juez
- Métricas detalladas y reportes
- Análisis por categorías y comparaciones
- Modo rápido para testing durante desarrollo

Usage:
    python rag_testing_suite.py [--quick] [--questions N] [--skip-generation]

Options:
    --quick: Modo rápido (menos preguntas para testing rápido)
    --questions N: Número específico de preguntas a generar/evaluar
    --skip-generation: Omitir generación si ya existe el dataset
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Local imports
from rag_test_generator import RAGTestGenerator
from rag_evaluator import RAGEvaluator
from rag_qa_system import RAGQuestionAnswering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_testing_suite_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGTestingSuite:
    """Suite completa de testing para el sistema RAG."""
    
    def __init__(self, 
                 quick_mode: bool = False,
                 num_questions: Optional[int] = None,
                 skip_generation: bool = False):
        """
        Initialize the testing suite.
        
        Args:
            quick_mode: Si True, usar configuración rápida para development
            num_questions: Número específico de preguntas (override quick_mode)
            skip_generation: Si True, omitir generación si existe dataset
        """
        self.quick_mode = quick_mode
        self.skip_generation = skip_generation
        
        # Determine number of questions
        if num_questions is not None:
            self.num_questions = num_questions
        elif quick_mode:
            self.num_questions = 50  # Rápido para desarrollo
        else:
            self.num_questions = 1000  # Completo para evaluación final
        
        # Paths
        self.test_dataset_path = "test_dataset.json"
        self.evaluation_report_path = f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.summary_report_path = f"rag_testing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logger.info(f"RAG Testing Suite initialized:")
        logger.info(f"  Mode: {'Quick' if quick_mode else 'Full'}")
        logger.info(f"  Questions to generate/evaluate: {self.num_questions}")
        logger.info(f"  Skip generation: {skip_generation}")
    
    def check_prerequisites(self) -> bool:
        """
        Verifica que todos los prerequisitos estén en su lugar.
        
        Returns:
            True si todo está listo, False si falta algo
        """
        logger.info("Checking prerequisites...")
        
        # Load environment variables first
        from dotenv import load_dotenv
        load_dotenv()
        
        issues = []
        
        # Check dataset directory
        if not os.path.exists("dataset"):
            issues.append("❌ Dataset directory not found")
        else:
            txt_files = list(Path("dataset").glob("*.txt"))
            if len(txt_files) < 10:
                issues.append(f"⚠️  Only {len(txt_files)} text files found in dataset (might be insufficient)")
        
        # Check metadata
        if not os.path.exists("greenpeace/greenpeace.csv"):
            issues.append("❌ Metadata CSV not found (greenpeace/greenpeace.csv)")
        
        # Check ChromaDB
        if not os.path.exists("chroma_db_rag"):
            issues.append("❌ ChromaDB not found (run rag_text_processor.py first)")
        
        # Check Ollama availability (instead of API key)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                issues.append("⚠️  Ollama server not responding (run: ollama serve)")
        except Exception:
            issues.append("❌ Ollama server not available (install Ollama and run: ollama serve)")
        
        if issues:
            logger.error("Prerequisites not met:")
            for issue in issues:
                logger.error(f"  {issue}")
            return False
        
        logger.info("✅ All prerequisites met!")
        return True
    
    def generate_test_dataset(self) -> bool:
        """
        Genera el dataset de prueba.
        
        Returns:
            True si la generación fue exitosa
        """
        # Check if we should skip generation
        if self.skip_generation and os.path.exists(self.test_dataset_path):
            logger.info(f"Skipping dataset generation - {self.test_dataset_path} already exists")
            
            # Verify existing dataset
            try:
                with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
                    existing_dataset = json.load(f)
                logger.info(f"Using existing dataset with {len(existing_dataset)} questions")
                return True
            except Exception as e:
                logger.warning(f"Error reading existing dataset: {e}. Will regenerate.")
        
        logger.info(f"Generating test dataset with {self.num_questions} questions...")
        
        try:
            # Initialize generator
            generator = RAGTestGenerator(target_questions=self.num_questions)
            
            # Generate dataset
            dataset = generator.generate_test_dataset()
            
            if not dataset:
                logger.error("Dataset generation failed - no questions generated")
                return False
            
            # Save dataset
            generator.save_dataset(dataset, self.test_dataset_path)
            
            logger.info(f"✅ Dataset generation completed: {len(dataset)} questions saved")
            return True
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            return False
    
    def run_evaluation(self) -> Optional[Dict]:
        """
        Ejecuta la evaluación del RAG.
        
        Returns:
            Diccionario con los resultados de la evaluación o None si falló
        """
        logger.info("Starting RAG evaluation...")
        
        try:
            # Check if test dataset exists
            if not os.path.exists(self.test_dataset_path):
                logger.error(f"Test dataset not found at {self.test_dataset_path}")
                return None
            
            # Initialize evaluator
            evaluator = RAGEvaluator(self.test_dataset_path)
            
            # Determine how many questions to evaluate
            with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
                test_dataset = json.load(f)
            
            max_eval_questions = min(len(test_dataset), self.num_questions) if not self.quick_mode else min(len(test_dataset), 20)
            
            logger.info(f"Evaluating {max_eval_questions} questions from dataset of {len(test_dataset)}")
            
            # Run evaluation
            results = evaluator.evaluate_dataset(max_questions=max_eval_questions)
            
            if not results:
                logger.error("Evaluation failed - no results generated")
                return None
            
            # Generate report
            report = evaluator.generate_report(results, self.evaluation_report_path)
            
            logger.info(f"✅ Evaluation completed: {len(results)} questions evaluated")
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def generate_summary_report(self, evaluation_report: Dict) -> str:
        """
        Genera un reporte de resumen en texto plano.
        
        Args:
            evaluation_report: Reporte de evaluación completo
            
        Returns:
            Texto del reporte de resumen
        """
        summary_lines = []
        
        # Header
        summary_lines.append("="*70)
        summary_lines.append("RAG SYSTEM EVALUATION SUMMARY REPORT")
        summary_lines.append("="*70)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Mode: {'Quick' if self.quick_mode else 'Full'} Testing")
        summary_lines.append("")
        
        # Metadata
        meta = evaluation_report.get('evaluation_metadata', {})
        summary_lines.append("EVALUATION METADATA")
        summary_lines.append("-" * 30)
        summary_lines.append(f"Total test questions available: {meta.get('total_test_questions', 'Unknown')}")
        summary_lines.append(f"Questions evaluated: {meta.get('questions_evaluated', 'Unknown')}")
        summary_lines.append(f"Test dataset: {meta.get('test_dataset_path', 'Unknown')}")
        summary_lines.append("")
        
        # Overall metrics
        metrics = evaluation_report.get('overall_metrics', {})
        if metrics:
            summary_lines.append("OVERALL PERFORMANCE METRICS")
            summary_lines.append("-" * 40)
            summary_lines.append(f"Valid evaluations: {metrics.get('valid_evaluations', 0)}/{metrics.get('total_questions_evaluated', 0)}")
            summary_lines.append(f"Error rate: {metrics.get('error_rate', 0)*100:.1f}%")
            summary_lines.append("")
            
            # Score metrics (0-10 scale)
            summary_lines.append("Performance Scores (0-10 scale):")
            for metric_name, display_name in [
                ('semantic_similarity', 'Semantic Similarity'),
                ('factual_accuracy', 'Factual Accuracy'),
                ('completeness', 'Completeness'),
                ('relevance', 'Relevance'),
                ('overall_score', 'Overall Score')
            ]:
                metric_data = metrics.get(metric_name, {})
                if metric_data:
                    mean = metric_data.get('mean', 0)
                    std = metric_data.get('std', 0)
                    summary_lines.append(f"  {display_name:20}: {mean:.2f} ± {std:.2f}")
            
            summary_lines.append("")
            
            # Source matching
            summary_lines.append("Source Matching Performance:")
            summary_lines.append(f"  Source Precision: {metrics.get('source_precision', 0)*100:.1f}%")
            summary_lines.append(f"  Category Accuracy: {metrics.get('category_accuracy', 0)*100:.1f}%")
            summary_lines.append("")
            
            # Performance timing
            summary_lines.append("Performance Timing:")
            summary_lines.append(f"  Average time per question: {metrics.get('average_evaluation_time', 0):.2f} seconds")
            summary_lines.append(f"  Total evaluation time: {metrics.get('total_evaluation_time', 0):.1f} seconds")
            summary_lines.append("")
            
            # Score distribution
            dist = metrics.get('score_distribution', {})
            if dist:
                summary_lines.append("Score Distribution:")
                summary_lines.append(f"  Excellent (9-10): {dist.get('excellent_9_10', 0):.1f}%")
                summary_lines.append(f"  Good (7-8): {dist.get('good_7_8', 0):.1f}%")
                summary_lines.append(f"  Fair (5-6): {dist.get('fair_5_6', 0):.1f}%")
                summary_lines.append(f"  Poor (3-4): {dist.get('poor_3_4', 0):.1f}%")
                summary_lines.append(f"  Very Poor (0-2): {dist.get('very_poor_0_2', 0):.1f}%")
                summary_lines.append("")
        
        # Category breakdown
        categories = evaluation_report.get('category_breakdown', {})
        if categories:
            summary_lines.append("PERFORMANCE BY CATEGORY")
            summary_lines.append("-" * 35)
            for cat, data in sorted(categories.items(), key=lambda x: x[1].get('count', 0), reverse=True):
                summary_lines.append(f"\n{cat}:")
                summary_lines.append(f"  Questions evaluated: {data.get('count', 0)}")
                summary_lines.append(f"  Avg semantic similarity: {data.get('avg_semantic_similarity', 0):.2f}")
                summary_lines.append(f"  Avg factual accuracy: {data.get('avg_factual_accuracy', 0):.2f}")
                summary_lines.append(f"  Avg completeness: {data.get('avg_completeness', 0):.2f}")
                summary_lines.append(f"  Avg relevance: {data.get('avg_relevance', 0):.2f}")
                summary_lines.append(f"  Source precision: {data.get('source_precision', 0)*100:.1f}%")
            summary_lines.append("")
        
        # Recommendations
        summary_lines.append("RECOMMENDATIONS")
        summary_lines.append("-" * 20)
        
        if metrics:
            overall_score = metrics.get('overall_score', {}).get('mean', 0)
            
            if overall_score >= 8:
                summary_lines.append("✅ EXCELLENT: The RAG system is performing very well!")
            elif overall_score >= 6:
                summary_lines.append("✨ GOOD: The RAG system shows solid performance with room for improvement.")
            elif overall_score >= 4:
                summary_lines.append("⚠️  FAIR: The RAG system needs significant improvements.")
            else:
                summary_lines.append("❌ POOR: The RAG system requires major fixes.")
            
            summary_lines.append("")
            
            # Specific recommendations
            factual_score = metrics.get('factual_accuracy', {}).get('mean', 0)
            completeness_score = metrics.get('completeness', {}).get('mean', 0)
            source_precision = metrics.get('source_precision', 0)
            
            if factual_score < 6:
                summary_lines.append("• Improve factual accuracy - consider better chunking or retrieval parameters")
            
            if completeness_score < 6:
                summary_lines.append("• Improve response completeness - consider retrieving more context chunks")
            
            if source_precision < 0.5:
                summary_lines.append("• Improve source retrieval - consider adjusting similarity thresholds")
            
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.1:
                summary_lines.append("• Reduce error rate - investigate system stability issues")
        
        summary_lines.append("")
        summary_lines.append("="*70)
        
        return "\n".join(summary_lines)
    
    def run_complete_evaluation(self) -> bool:
        """
        Ejecuta el pipeline completo de evaluación.
        
        Returns:
            True si todo el proceso fue exitoso
        """
        start_time = time.time()
        
        logger.info("🚀 Starting RAG Testing Suite - Complete Evaluation Pipeline")
        logger.info("="*60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.error("Cannot proceed - prerequisites not met")
            return False
        
        # Step 2: Generate test dataset
        logger.info("\n📝 STEP 1: Generating Test Dataset")
        logger.info("-" * 40)
        if not self.generate_test_dataset():
            logger.error("Dataset generation failed")
            return False
        
        # Step 3: Run evaluation
        logger.info("\n🔍 STEP 2: Running RAG Evaluation")
        logger.info("-" * 40)
        evaluation_report = self.run_evaluation()
        if not evaluation_report:
            logger.error("RAG evaluation failed")
            return False
        
        # Step 4: Generate summary report
        logger.info("\n📊 STEP 3: Generating Summary Report")
        logger.info("-" * 40)
        try:
            summary_text = self.generate_summary_report(evaluation_report)
            
            # Save summary
            with open(self.summary_report_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            logger.info(f"Summary report saved to: {self.summary_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            # Don't fail the entire process for this
        
        # Final summary
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("🎉 RAG TESTING SUITE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.1f} seconds")
        logger.info(f"\nOutput files:")
        logger.info(f"  📄 Test dataset: {self.test_dataset_path}")
        logger.info(f"  📊 Detailed report: {self.evaluation_report_path}")
        logger.info(f"  📝 Summary report: {self.summary_report_path}")
        
        # Quick performance summary
        if evaluation_report and 'overall_metrics' in evaluation_report:
            metrics = evaluation_report['overall_metrics']
            overall_score = metrics.get('overall_score', {}).get('mean', 0)
            
            logger.info(f"\n🎯 QUICK RESULTS:")
            logger.info(f"  Overall Score: {overall_score:.2f}/10")
            logger.info(f"  Questions Evaluated: {metrics.get('valid_evaluations', 0)}")
            logger.info(f"  Source Precision: {metrics.get('source_precision', 0)*100:.1f}%")
        
        return True


def main():
    """Función principal del sistema de testing."""
    parser = argparse.ArgumentParser(description="RAG Testing Suite - Complete Evaluation Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer questions for development)")
    parser.add_argument("--questions", type=int, help="Specific number of questions to generate/evaluate")
    parser.add_argument("--skip-generation", action="store_true", help="Skip dataset generation if file exists")
    
    args = parser.parse_args()
    
    try:
        # Initialize testing suite
        suite = RAGTestingSuite(
            quick_mode=args.quick,
            num_questions=args.questions,
            skip_generation=args.skip_generation
        )
        
        # Run complete evaluation
        success = suite.run_complete_evaluation()
        
        if success:
            print("\n🎉 RAG testing completed successfully!")
            print(f"Check the generated reports for detailed results.")
        else:
            print("\n❌ RAG testing failed. Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()