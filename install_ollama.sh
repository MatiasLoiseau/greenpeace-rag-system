#!/bin/bash
# Guía de instalación paso a paso para Ollama

echo "🚀 RAG Testing Setup Guide with Ollama"
echo "======================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 Detected macOS"
    
    # Check if Homebrew is installed
    if command -v brew &> /dev/null; then
        echo "🍺 Homebrew found"
        
        # Check if Ollama is already installed
        if command -v ollama &> /dev/null; then
            echo "✅ Ollama already installed"
        else
            echo "📦 Installing Ollama via Homebrew..."
            brew install ollama
        fi
    else
        echo "❌ Homebrew not found"
        echo "Please install Ollama manually:"
        echo "1. Go to https://ollama.com"
        echo "2. Download Ollama for macOS"
        echo "3. Install and run the app"
        exit 1
    fi
else
    # Linux installation
    echo "🐧 Detected Linux"
    
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama already installed"
    else
        echo "📦 Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

echo ""
echo "🔄 Starting Ollama server..."

# Start Ollama server in background
nohup ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

echo "📝 Ollama server started with PID: $OLLAMA_PID"
echo "💾 Logs are being written to ollama.log"

# Wait for server to start
echo "⏳ Waiting for Ollama server to start..."
sleep 5

# Check if server is running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama server is running"
else
    echo "❌ Ollama server failed to start"
    echo "Check ollama.log for errors"
    exit 1
fi

echo ""
echo "📦 Downloading llama3.2 model (this may take several minutes)..."
echo "💡 Model size is approximately 2GB"

ollama pull llama3.2

if [ $? -eq 0 ]; then
    echo "✅ llama3.2 model downloaded successfully"
else
    echo "❌ Failed to download llama3.2 model"
    exit 1
fi

echo ""
echo "🧪 Testing the setup..."

# Test the model
echo "Testing basic functionality..." > test_input.txt
echo "Hello, respond with just 'OK'" >> test_input.txt

TEST_RESPONSE=$(ollama run llama3.2 < test_input.txt 2>/dev/null)
rm test_input.txt

if [[ $TEST_RESPONSE == *"OK"* ]] || [[ $TEST_RESPONSE == *"ok"* ]]; then
    echo "✅ Model test successful"
else
    echo "⚠️  Model test gave: $TEST_RESPONSE"
    echo "But this might still work fine."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Summary:"
echo "  ✅ Ollama server is running (PID: $OLLAMA_PID)"
echo "  ✅ llama3.2 model is available"
echo "  📄 Server logs: ollama.log"
echo ""
echo "🚀 Now you can run:"
echo "  conda activate llm"
echo "  python test_ollama.py              # Test connection"
echo "  python rag_testing_suite.py --quick # Quick RAG test"
echo ""
echo "⏹️  To stop Ollama later:"
echo "  kill $OLLAMA_PID"
echo "  # or"
echo "  pkill -f 'ollama serve'"
echo ""
echo "📝 Note: Keep this terminal open or save the PID to stop Ollama later"