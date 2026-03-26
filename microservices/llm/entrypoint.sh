#!/bin/sh
set -e

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do
  sleep 1
done
echo "Ollama is ready."

# Pull model if not already present
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
  echo "Pulling ${OLLAMA_MODEL}..."
  ollama pull "$OLLAMA_MODEL"
fi

echo "LLM microservice ready."

# Wait for the server process to exit
wait $OLLAMA_PID
