"""
Script para convertir archivos PDF y Markdown del directorio greenpeace/docs 
a formato texto y guardarlos en dataset_txt.
"""

import os
import PyPDF2
import pdfplumber
from pathlib import Path
import json
from datetime import datetime
import logging


class DocumentConverter:
    def __init__(self, source_dir, output_dir, log_file="conversion_log.json"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.log_file = log_file
        self.project_root = Path(__file__).parent  # Directorio raíz del proyecto
        
        # Crear directorio de salida si no existe
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicializar log de errores
        self.error_log = {
            "conversion_date": datetime.now().isoformat(),
            "successful_files": [],
            "failed_files": []
        }
    
    def extract_text_from_pdf_pypdf2(self, pdf_path):
        """Extrae texto de PDF usando PyPDF2."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_pdf_pdfplumber(self, pdf_path):
        """Extrae texto de PDF usando pdfplumber (método alternativo)."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def extract_text_from_markdown(self, md_path):
        """Extrae texto de archivo Markdown."""
        with open(md_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def convert_pdf_to_txt(self, pdf_path, output_path):
        """Convierte PDF a TXT usando múltiples métodos."""
        text = ""
        
        # Intentar primero con PyPDF2
        try:
            text = self.extract_text_from_pdf_pypdf2(pdf_path)
            if text.strip():  # Si se extrajo texto válido
                return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 falló para {pdf_path.name}: {str(e)}")
        
        # Si PyPDF2 falla o no extrae texto, intentar con pdfplumber
        try:
            text = self.extract_text_from_pdf_pdfplumber(pdf_path)
            if text.strip():
                return text
        except Exception as e:
            self.logger.warning(f"pdfplumber falló para {pdf_path.name}: {str(e)}")
        
        # Si ambos métodos fallan
        raise Exception("No se pudo extraer texto con ningún método disponible")
    
    def convert_markdown_to_txt(self, md_path, output_path):
        """Convierte Markdown a TXT."""
        return self.extract_text_from_markdown(md_path)
    
    def process_file(self, file_path):
        """Procesa un archivo individual."""
        file_extension = file_path.suffix.lower()
        output_filename = file_path.stem + ".txt"
        output_path = self.output_dir / output_filename
        
        try:
            if file_extension == '.pdf':
                text_content = self.convert_pdf_to_txt(file_path, output_path)
            elif file_extension == '.md':
                text_content = self.convert_markdown_to_txt(file_path, output_path)
            else:
                raise Exception(f"Tipo de archivo no soportado: {file_extension}")
            
            # Guardar el texto extraído
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text_content)
            
            self.logger.info(f"✓ Convertido exitosamente: {file_path.name}")
            self.error_log["successful_files"].append({
                "original_file": str(file_path.name),
                "output_file": str(output_filename),
                "file_type": file_extension,
                "text_length": len(text_content)
            })
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"✗ Error procesando {file_path.name}: {error_msg}")
            self.error_log["failed_files"].append({
                "original_file": str(file_path.name),
                "file_type": file_extension,
                "error_message": error_msg,
                "error_type": type(e).__name__
            })
            return False
    
    def process_all_files(self):
        """Procesa todos los archivos PDF y Markdown del directorio fuente."""
        supported_extensions = {'.pdf', '.md'}
        
        # Obtener todos los archivos soportados
        files_to_process = [
            f for f in self.source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        self.logger.info(f"Encontrados {len(files_to_process)} archivos para procesar")
        
        successful_count = 0
        failed_count = 0
        
        for file_path in files_to_process:
            if self.process_file(file_path):
                successful_count += 1
            else:
                failed_count += 1
        
        # Guardar log de errores
        self.save_error_log()
        
        # Mostrar resumen
        self.print_summary(successful_count, failed_count, len(files_to_process))
    
    def save_error_log(self):
        """Guarda el log de errores en un archivo JSON."""
        log_path = self.project_root / self.log_file
        with open(log_path, 'w', encoding='utf-8') as log_file:
            json.dump(self.error_log, log_file, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Log de conversión guardado en: {log_path}")
    
    def print_summary(self, successful_count, failed_count, total_count):
        """Imprime un resumen de la conversión."""
        print("\n" + "="*60)
        print("RESUMEN DE CONVERSIÓN")
        print("="*60)
        print(f"Total de archivos procesados: {total_count}")
        print(f"Conversiones exitosas: {successful_count}")
        print(f"Conversiones fallidas: {failed_count}")
        print(f"Tasa de éxito: {(successful_count/total_count)*100:.1f}%")
        
        if failed_count > 0:
            print(f"\nArchivos con errores:")
            for failed_file in self.error_log["failed_files"]:
                print(f"  - {failed_file['original_file']}: {failed_file['error_message']}")
        
        print(f"\nArchivos TXT generados en: {self.output_dir}")
        print(f"Log detallado en: {self.project_root / self.log_file}")
        print("="*60)


def main():
    # Configurar rutas
    current_dir = Path(__file__).parent
    source_directory = current_dir / "greenpeace" / "docs"
    output_directory = current_dir / "dataset_txt"
    
    # Verificar que el directorio fuente existe
    if not source_directory.exists():
        print(f"Error: El directorio {source_directory} no existe")
        return
    
    # Crear y ejecutar el convertidor
    converter = DocumentConverter(source_directory, output_directory)
    converter.process_all_files()


if __name__ == "__main__":
    main()