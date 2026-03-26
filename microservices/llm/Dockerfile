FROM ollama/ollama:latest

ARG OLLAMA_MODEL=qwen2.5:3b
ENV OLLAMA_MODEL=${OLLAMA_MODEL}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
