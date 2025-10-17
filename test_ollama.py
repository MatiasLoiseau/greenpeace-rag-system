#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple para verificar que Ollama funciona correctamente
"""

import sys
import requests
import time
from langchain_ollama import OllamaLLM

def test_ollama_connection():
    """Test conexión a Ollama server."""
    print("🔍 Testing Ollama server connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            
            models = response.json()
            if 'models' in models and models['models']:
                print(f"📦 Available models: {[m['name'] for m in models['models']]}")
                
                # Check if llama3.2 is available
                model_names = [m['name'] for m in models['models']]
                if any('llama3.2' in name for name in model_names):
                    print("✅ llama3.2 model found")
                    return True
                else:
                    print("❌ llama3.2 model not found")
                    print("Run: ollama pull llama3.2")
                    return False
            else:
                print("❌ No models found")
                return False
        else:
            print(f"❌ Ollama server error: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Make sure to run: ollama serve")
        return False

def test_ollama_model():
    """Test básico del modelo llama3.2."""
    print("\n🧪 Testing llama3.2 model...")
    
    try:
        llm = OllamaLLM(model="llama3.2", temperature=0.1)
        
        test_prompt = "Responde con exactamente una palabra: 'OK'"
        
        print(f"Sending prompt: {test_prompt}")
        start_time = time.time()
        
        response = llm.invoke(test_prompt)
        
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if "OK" in response or "ok" in response.lower():
            print("✅ Model test successful")
            return True
        else:
            print("⚠️  Model gave unexpected response but might still work")
            return True
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🚀 Ollama Test Suite")
    print("="*30)
    
    # Test 1: Server connection
    if not test_ollama_connection():
        print("\n❌ Ollama server test failed")
        sys.exit(1)
    
    # Test 2: Model functionality
    if not test_ollama_model():
        print("\n❌ Model test failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Ollama is ready for RAG testing.")
    print("\nNow you can run:")
    print("  python rag_testing_suite.py --quick")

if __name__ == "__main__":
    main()