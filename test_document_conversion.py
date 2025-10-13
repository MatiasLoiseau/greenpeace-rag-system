"""
Test script para validar la conversión de documentos y rastrear errores.
Este script verifica la integridad de las conversiones y genera reportes detallados.
"""

import unittest
import json
import os
from pathlib import Path
from convert_docs_to_txt import DocumentConverter
import tempfile
import shutil


class TestDocumentConverter(unittest.TestCase):
    """Test cases para el convertidor de documentos."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.current_dir = Path(__file__).parent
        self.source_dir = self.current_dir / "greenpeace" / "docs"
        self.output_dir = self.current_dir / "dataset_txt"
        self.converter = DocumentConverter(self.source_dir, self.output_dir)
    
    def test_source_directory_exists(self):
        """Verifica que el directorio fuente existe."""
        self.assertTrue(self.source_dir.exists(), 
                       f"El directorio fuente {self.source_dir} no existe")
    
    def test_output_directory_creation(self):
        """Verifica que se puede crear el directorio de salida."""
        self.assertTrue(self.output_dir.exists(), 
                       f"No se pudo crear el directorio de salida {self.output_dir}")
    
    def test_supported_file_types(self):
        """Verifica que existen archivos PDF y Markdown para procesar."""
        pdf_files = list(self.source_dir.glob("*.pdf"))
        md_files = list(self.source_dir.glob("*.md"))
        
        self.assertGreater(len(pdf_files), 0, "No se encontraron archivos PDF")
        self.assertGreater(len(md_files), 0, "No se encontraron archivos Markdown")
        
        print(f"Encontrados {len(pdf_files)} archivos PDF")
        print(f"Encontrados {len(md_files)} archivos Markdown")
    
    def test_conversion_process(self):
        """Test completo del proceso de conversión."""
        # Ejecutar la conversión
        self.converter.process_all_files()
        
        # Verificar que se generó el log
        log_file = self.output_dir / "conversion_log.json"
        self.assertTrue(log_file.exists(), "No se generó el archivo de log")
        
        # Cargar y verificar el contenido del log
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        self.assertIn('successful_files', log_data)
        self.assertIn('failed_files', log_data)
        self.assertIn('conversion_date', log_data)
        
        # Verificar que se procesaron archivos
        total_processed = len(log_data['successful_files']) + len(log_data['failed_files'])
        self.assertGreater(total_processed, 0, "No se procesó ningún archivo")
        
        return log_data
    
    def test_output_files_integrity(self):
        """Verifica la integridad de los archivos de salida."""
        log_data = self.test_conversion_process()
        
        for successful_file in log_data['successful_files']:
            output_file_path = self.output_dir / successful_file['output_file']
            
            # Verificar que el archivo existe
            self.assertTrue(output_file_path.exists(), 
                          f"El archivo de salida {output_file_path} no existe")
            
            # Verificar que el archivo no está vacío
            file_size = output_file_path.stat().st_size
            self.assertGreater(file_size, 0, 
                             f"El archivo {output_file_path} está vacío")
            
            # Verificar que se puede leer
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.assertGreater(len(content.strip()), 0, 
                                     f"El contenido de {output_file_path} está vacío")
            except Exception as e:
                self.fail(f"No se pudo leer el archivo {output_file_path}: {e}")


class DocumentConversionReport:
    """Genera reportes detallados del proceso de conversión."""
    
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.project_root = Path(__file__).parent  # Directorio raíz del proyecto
        self.load_log_data()
    
    def load_log_data(self):
        """Carga los datos del log de conversión."""
        if not self.log_file_path.exists():
            raise FileNotFoundError(f"Log file {self.log_file_path} no encontrado")
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            self.log_data = json.load(f)
    
    def generate_error_report(self):
        """Genera un reporte detallado de errores."""
        failed_files = self.log_data.get('failed_files', [])
        
        if not failed_files:
            return "✓ No se encontraron errores en la conversión"
        
        report = f"\n🚨 REPORTE DE ERRORES DE CONVERSIÓN\n"
        report += f"{'='*60}\n"
        report += f"Total de archivos con errores: {len(failed_files)}\n"
        report += f"Fecha de conversión: {self.log_data.get('conversion_date', 'N/A')}\n\n"
        
        # Agrupar errores por tipo
        error_types = {}
        for failed_file in failed_files:
            error_type = failed_file.get('error_type', 'Unknown')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failed_file)
        
        for error_type, files in error_types.items():
            report += f"📋 Tipo de Error: {error_type}\n"
            report += f"   Archivos afectados: {len(files)}\n"
            for file_info in files:
                report += f"   - {file_info['original_file']}: {file_info['error_message']}\n"
            report += "\n"
        
        return report
    
    def generate_success_report(self):
        """Genera un reporte de conversiones exitosas."""
        successful_files = self.log_data.get('successful_files', [])
        
        report = f"\n✅ REPORTE DE CONVERSIONES EXITOSAS\n"
        report += f"{'='*60}\n"
        report += f"Total de archivos convertidos exitosamente: {len(successful_files)}\n\n"
        
        # Estadísticas por tipo de archivo
        file_types = {}
        total_text_length = 0
        
        for file_info in successful_files:
            file_type = file_info.get('file_type', 'unknown')
            if file_type not in file_types:
                file_types[file_type] = {'count': 0, 'total_length': 0}
            
            file_types[file_type]['count'] += 1
            text_length = file_info.get('text_length', 0)
            file_types[file_type]['total_length'] += text_length
            total_text_length += text_length
        
        for file_type, stats in file_types.items():
            avg_length = stats['total_length'] / stats['count'] if stats['count'] > 0 else 0
            report += f"📄 Archivos {file_type}:\n"
            report += f"   - Cantidad: {stats['count']}\n"
            report += f"   - Texto total: {stats['total_length']:,} caracteres\n"
            report += f"   - Promedio por archivo: {avg_length:,.0f} caracteres\n\n"
        
        report += f"📊 Total de texto extraído: {total_text_length:,} caracteres\n"
        
        return report
    
    def generate_full_report(self):
        """Genera un reporte completo."""
        successful_count = len(self.log_data.get('successful_files', []))
        failed_count = len(self.log_data.get('failed_files', []))
        total_count = successful_count + failed_count
        
        report = f"\n📋 REPORTE COMPLETO DE CONVERSIÓN DE DOCUMENTOS\n"
        report += f"{'='*70}\n"
        report += f"Fecha de conversión: {self.log_data.get('conversion_date', 'N/A')}\n"
        report += f"Total de archivos procesados: {total_count}\n"
        report += f"Conversiones exitosas: {successful_count}\n"
        report += f"Conversiones fallidas: {failed_count}\n"
        
        if total_count > 0:
            success_rate = (successful_count / total_count) * 100
            report += f"Tasa de éxito: {success_rate:.1f}%\n"
        
        report += self.generate_success_report()
        report += self.generate_error_report()
        
        return report
    
    def save_report_to_file(self, output_file="conversion_report.txt"):
        """Guarda el reporte completo en un archivo."""
        report = self.generate_full_report()
        
        output_path = self.project_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Reporte guardado en: {output_path}")
        return output_path


def run_conversion_tests():
    """Ejecuta todos los tests de conversión."""
    print("🧪 Ejecutando tests de conversión de documentos...\n")
    
    # Crear suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDocumentConverter)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def generate_conversion_report():
    """Genera y muestra el reporte de conversión."""
    current_dir = Path(__file__).parent
    log_file = current_dir / "conversion_log.json"
    
    if not log_file.exists():
        print("❌ No se encontró el archivo de log. Ejecuta primero la conversión.")
        return
    
    # Generar reporte
    reporter = DocumentConversionReport(log_file)
    report = reporter.generate_full_report()
    
    print(report)
    
    # Guardar reporte en archivo
    reporter.save_report_to_file()


def main():
    """Función principal que ejecuta tests y genera reportes."""
    print("🚀 Iniciando validación del sistema de conversión de documentos\n")
    
    # Ejecutar tests
    tests_passed = run_conversion_tests()
    
    if tests_passed:
        print("\n✅ Todos los tests pasaron correctamente!")
        
        # Generar reporte
        print("\n📊 Generando reporte de conversión...")
        generate_conversion_report()
    else:
        print("\n❌ Algunos tests fallaron. Revisa los errores anteriores.")


if __name__ == "__main__":
    main()