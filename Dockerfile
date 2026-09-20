FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Streamlit is the primary, self-contained UI (runs the full RAG pipeline
# directly). PINECONE_API_KEY and GROQ_API_KEY must be supplied at runtime,
# e.g. `docker run --env-file .env ...` — never baked into the image.
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
