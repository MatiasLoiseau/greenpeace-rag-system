#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Test Dataset Generator
=========================

Genera un conjunto de 1000 preguntas y respuestas de prueba basadas en párrafos
aleatorios de los documentos del dataset para evaluar el sistema RAG.

Features:
- Selecciona párrafos aleatorios de documentos
- Genera preguntas contextuales usando Gemini
- Crea respuestas de referencia (ground truth)
- Guarda metadata de origen para evaluación
- Evita preguntas demasiado literales

Usage:
    python rag_test_generator.py

Output:
    - test_dataset.json: Dataset de prueba con preguntas, respuestas y metadata
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
import re

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGTestGenerator:
    """Genera dataset de prueba para evaluar el sistema RAG."""
    
    def __init__(self, 
                 dataset_dir: str = "dataset",
                 metadata_csv: str = "greenpeace/greenpeace.csv",
                 target_questions: int = 1000,
                 min_paragraph_length: int = 200,
                 max_paragraph_length: int = 800):
        """
        Initialize the test generator.
        
        Args:
            dataset_dir: Directory containing text files
            metadata_csv: Path to metadata CSV
            target_questions: Number of questions to generate
            min_paragraph_length: Minimum paragraph length in characters
            max_paragraph_length: Maximum paragraph length in characters
        """
        load_dotenv()
        
        self.dataset_dir = Path(dataset_dir)
        self.metadata_csv = Path(metadata_csv)
        self.target_questions = target_questions
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        
        # Load metadata
        self.metadata_df = pd.read_csv(self.metadata_csv)
        logger.info(f"Loaded metadata for {len(self.metadata_df)} documents")
        
        # Initialize local Ollama model
        try:
            self.llm = OllamaLLM(
                model="llama3.2",  # Usar llama3.2 que es más ligero
                temperature=0.7,  # Slightly higher for diverse questions
            )
            logger.info("Local Ollama model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            logger.info("Make sure Ollama is running with: ollama serve")
            logger.info("And the model is installed with: ollama pull llama3.2")
            raise
        
        # Create prompt templates (simplified for Ollama)
        pass  # Templates will be defined as methods
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Divide texto en párrafos usando diferentes delimitadores.
        
        Args:
            text: Texto completo del archivo
            
        Returns:
            Lista de párrafos
        """
        # Primero intentar con dobles saltos de línea
        paragraphs = text.split('\n\n')
        
        # Si no hay suficientes párrafos, intentar con saltos simples
        if len(paragraphs) < 3:
            paragraphs = text.split('\n')
        
        # Filtrar párrafos por longitud
        valid_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if (self.min_paragraph_length <= len(para) <= self.max_paragraph_length 
                and not para.startswith('http') 
                and not para.isdigit()):
                valid_paragraphs.append(para)
        
        return valid_paragraphs
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Lee el contenido de un archivo de texto."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _get_file_metadata(self, filename: str) -> Dict:
        """Obtiene metadata de un archivo desde el CSV."""
        file_id = filename.replace('.txt', '')
        matching_rows = self.metadata_df[self.metadata_df['id'] == file_id]
        
        if matching_rows.empty:
            return {'category': 'Unknown', 'title': filename}
        
        return matching_rows.iloc[0].to_dict()
    
    def _select_random_paragraphs(self) -> List[Dict]:
        """
        Selecciona párrafos aleatorios de todos los archivos del dataset.
        
        Returns:
            Lista de diccionarios con párrafo y metadata
        """
        logger.info("Selecting random paragraphs from dataset...")
        
        all_paragraphs = []
        txt_files = list(self.dataset_dir.glob("*.txt"))
        
        for file_path in txt_files:
            content = self._read_file_content(file_path)
            if not content:
                continue
            
            # Get file metadata
            metadata = self._get_file_metadata(file_path.name)
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(content)
            
            # Add each paragraph with metadata
            for paragraph in paragraphs:
                all_paragraphs.append({
                    'paragraph': paragraph,
                    'source_file': file_path.name,
                    'category': metadata.get('category', 'Unknown'),
                    'title': metadata.get('title', file_path.name),
                    'file_path': str(file_path)
                })
        
        logger.info(f"Found {len(all_paragraphs)} valid paragraphs from {len(txt_files)} files")
        
        # Seleccionar párrafos aleatorios
        if len(all_paragraphs) < self.target_questions:
            logger.warning(f"Only {len(all_paragraphs)} paragraphs available, less than target {self.target_questions}")
            return all_paragraphs
        
        selected = random.sample(all_paragraphs, self.target_questions)
        logger.info(f"Selected {len(selected)} random paragraphs")
        return selected
    
    def _build_question_prompt(self, paragraph_data: Dict) -> str:
        """Construye el prompt para generar preguntas."""
        return f"""You are an expert at generating comprehension questions to evaluate search and response systems.

Your task is to create ONE intelligent and contextual question based on the following paragraph from a Greenpeace document.

Paragraph:
{paragraph_data['paragraph']}

Document: {paragraph_data['source_file']}
Category: {paragraph_data['category']}

IMPORTANT INSTRUCTIONS:
1. The question must require understanding of content, NOT be literal
2. Should be specific but not obvious (avoid questions answered by copying direct text)
3. Can ask about causes, consequences, relationships, implications
4. Must be in ENGLISH
5. Ideal length: 10-25 words
6. Respond ONLY with the question, no additional explanations

Examples of GOOD questions:
- What factors contribute to...?
- What are the implications of...?
- How does X relate to Y according to the document?
- Why is ... important?

Avoid questions like:
- What does the paragraph say about...? (too literal)
- When did ... occur? (too specific)

Question:"""

    def _generate_question(self, paragraph_data: Dict) -> Optional[str]:
        """Genera una pregunta basada en un párrafo."""
        try:
            prompt = self._build_question_prompt(paragraph_data)
            question = self.llm.invoke(prompt)
            return question.strip()
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return None
    
    def _build_answer_prompt(self, paragraph: str, question: str) -> str:
        """Construye el prompt para generar respuestas."""
        return f"""You are an expert in environmental topics and Greenpeace. Answer the following question based ONLY on the provided paragraph.

Paragraph:
{paragraph}

Question: {question}

INSTRUCTIONS:
1. Answer concisely but completely (2-4 sentences)
2. Base your answer ONLY on information from the paragraph
3. If the paragraph doesn't contain enough information, say "The provided information is insufficient to answer completely"
4. Use an informative and objective tone
5. Respond in ENGLISH

Answer:"""

    def _generate_answer(self, paragraph: str, question: str) -> Optional[str]:
        """Genera una respuesta basada en el párrafo y la pregunta."""
        try:
            prompt = self._build_answer_prompt(paragraph, question)
            answer = self.llm.invoke(prompt)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def generate_test_dataset(self) -> List[Dict]:
        """
        Genera el dataset completo de preguntas y respuestas.
        
        Returns:
            Lista de diccionarios con preguntas, respuestas y metadata
        """
        logger.info(f"Starting generation of {self.target_questions} Q&A pairs...")
        
        # Seleccionar párrafos
        selected_paragraphs = self._select_random_paragraphs()
        
        test_dataset = []
        successful_generations = 0
        failed_generations = 0
        
        for i, paragraph_data in enumerate(selected_paragraphs):
            logger.info(f"Processing paragraph {i+1}/{len(selected_paragraphs)}")
            
            try:
                # Generar pregunta
                question = self._generate_question(paragraph_data)
                if not question:
                    failed_generations += 1
                    continue
                
                # Generar respuesta
                answer = self._generate_answer(paragraph_data['paragraph'], question)
                if not answer:
                    failed_generations += 1
                    continue
                
                # Crear entrada del dataset
                qa_pair = {
                    'id': f"qa_{successful_generations + 1:04d}",
                    'question': question,
                    'answer': answer,
                    'source_paragraph': paragraph_data['paragraph'],
                    'source_file': paragraph_data['source_file'],
                    'category': paragraph_data['category'],
                    'title': paragraph_data['title'],
                    'paragraph_length': len(paragraph_data['paragraph']),
                    'question_length': len(question),
                    'answer_length': len(answer)
                }
                
                test_dataset.append(qa_pair)
                successful_generations += 1
                
                # Log progress every 50 items
                if successful_generations % 50 == 0:
                    logger.info(f"Successfully generated {successful_generations} Q&A pairs")
                
            except Exception as e:
                logger.error(f"Error processing paragraph {i+1}: {e}")
                failed_generations += 1
                continue
        
        logger.info(f"Generation completed! Success: {successful_generations}, Failed: {failed_generations}")
        return test_dataset
    
    def save_dataset(self, dataset: List[Dict], output_path: str = "test_dataset.json"):
        """Guarda el dataset en un archivo JSON."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Dataset saved to {output_path}")
            
            # Generate statistics
            stats = self._generate_statistics(dataset)
            stats_path = output_path.replace('.json', '_stats.json')
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Statistics saved to {stats_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def _generate_statistics(self, dataset: List[Dict]) -> Dict:
        """Genera estadísticas del dataset."""
        if not dataset:
            return {'error': 'Empty dataset'}
        
        # Basic stats
        total_items = len(dataset)
        categories = {}
        source_files = {}
        
        question_lengths = []
        answer_lengths = []
        paragraph_lengths = []
        
        for item in dataset:
            # Count by category
            cat = item.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count by source file
            src = item.get('source_file', 'Unknown')
            source_files[src] = source_files.get(src, 0) + 1
            
            # Length statistics
            question_lengths.append(item.get('question_length', 0))
            answer_lengths.append(item.get('answer_length', 0))
            paragraph_lengths.append(item.get('paragraph_length', 0))
        
        stats = {
            'total_qa_pairs': total_items,
            'categories': categories,
            'top_10_source_files': dict(sorted(source_files.items(), key=lambda x: x[1], reverse=True)[:10]),
            'question_length_stats': {
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0,
                'avg': sum(question_lengths) / len(question_lengths) if question_lengths else 0
            },
            'answer_length_stats': {
                'min': min(answer_lengths) if answer_lengths else 0,
                'max': max(answer_lengths) if answer_lengths else 0,
                'avg': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
            },
            'paragraph_length_stats': {
                'min': min(paragraph_lengths) if paragraph_lengths else 0,
                'max': max(paragraph_lengths) if paragraph_lengths else 0,
                'avg': sum(paragraph_lengths) / len(paragraph_lengths) if paragraph_lengths else 0
            }
        }
        
        return stats


def main():
    """Función principal para generar el dataset de prueba."""
    logger.info("=== RAG Test Dataset Generator ===")
    
    try:
        # Initialize generator
        generator = RAGTestGenerator(target_questions=1000)
        
        # Generate dataset
        dataset = generator.generate_test_dataset()
        
        if not dataset:
            logger.error("No dataset generated!")
            return
        
        # Save dataset
        generator.save_dataset(dataset)
        
        # Print summary
        print("\n" + "="*50)
        print("DATASET GENERATION SUMMARY")
        print("="*50)
        print(f"Total Q&A pairs generated: {len(dataset)}")
        
        # Category breakdown
        categories = {}
        for item in dataset:
            cat = item.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nBreakdown by category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} questions")
        
        # Sample questions
        print(f"\nSample questions:")
        for i in range(min(3, len(dataset))):
            item = dataset[i]
            print(f"\n{i+1}. Source: {item['source_file']}")
            print(f"   Question: {item['question']}")
            print(f"   Answer: {item['answer'][:100]}...")
        
        print(f"\n✅ Dataset saved as 'test_dataset.json'")
        print(f"📊 Statistics saved as 'test_dataset_stats.json'")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()