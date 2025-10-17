#!/bin/bash
# Setup script para RAG Testing con Ollama local

echo "🚀 Setting up RAG Testing System with Local Ollama"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install it first:"
    echo ""
    echo "On macOS: brew install ollama"
    echo "On Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Or download from: https://ollama.com"
    exit 1
fi

echo "✅ Ollama found"

# Start Ollama server in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🔄 Starting Ollama server..."
    ollama serve &
    sleep 5
else
    echo "✅ Ollama server already running"
fi

# Check if llama3.2 model is available
echo "🔍 Checking for llama3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "📦 Downloading llama3.2 model (this may take a while)..."
    ollama pull llama3.2
    
    if [ $? -eq 0 ]; then
        echo "✅ llama3.2 model downloaded successfully"
    else
        echo "❌ Failed to download llama3.2 model"
        exit 1
    fi
else
    echo "✅ llama3.2 model already available"
fi

# Test the model
echo "🧪 Testing llama3.2 model..."
test_response=$(echo "Hello, respond with 'OK' if you can understand this." | ollama run llama3.2 2>/dev/null)

if [[ $test_response == *"OK"* ]] || [[ $test_response == *"ok"* ]]; then
    echo "✅ Model test successful"
else
    echo "⚠️  Model test gave unexpected response: $test_response"
    echo "But this might still work fine."
fi

# Install Python requirements if needed
echo "📦 Checking Python dependencies..."
if ! python -c "import langchain_community" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip install langchain-community requests
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Now you can run:"
echo "  python rag_testing_suite.py --quick    # Quick test"
echo "  python rag_testing_suite.py           # Full test"
echo ""
echo "Note: Keep the Ollama server running in the background"
echo "To stop it later: pkill -f 'ollama serve'"