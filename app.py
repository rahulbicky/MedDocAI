"""
Optional JSON API for MedQuery AI.

Streamlit (streamlit.py) is the primary, fully self-contained user interface
and already runs the complete RAG pipeline directly. This Flask app is an
optional, separate headless API for programmatic/automated access to the
same RAG pipeline (e.g. calling it from another service or script) — it is
not required to use the Streamlit app and is not started by default in
Docker. See README.md "API / Backend" section for details.

Usage:
    python app.py
    curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" \
         -d '{"question": "What is Acromegaly?"}'
"""

import os
import sys

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    GROQ_MODEL_NAME,
    PINECONE_INDEX_NAME,
    RETRIEVER_TOP_K,
    download_hugging_face_embeddings,
)
from src.prompt import system_prompt

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY:
    sys.exit("Missing PINECONE_API_KEY. Please add it to your .env file (see .env.example).")
if not GROQ_API_KEY:
    sys.exit("Missing GROQ_API_KEY. Please add it to your .env file (see .env.example).")

app = Flask(__name__)
rag_chain = None
init_error = None

try:
    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_TOP_K}
    )
    chat_model = ChatGroq(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as exc:  # noqa: BLE001 - surfaced via /health, not printed with secrets
    init_error = (
        f"{exc.__class__.__name__}: could not connect to the Pinecone index "
        f"'{PINECONE_INDEX_NAME}'. Run 'python store_index.py' first if you "
        "haven't indexed any documents yet."
    )


@app.route("/health", methods=["GET"])
def health():
    if rag_chain is None:
        return jsonify({"status": "error", "detail": init_error}), 503
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    if rag_chain is None:
        return jsonify({"error": init_error}), 503

    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Request body must include a non-empty 'question' field."}), 400

    try:
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer": response["answer"]})
    except Exception:  # noqa: BLE001 - don't leak internals/tracebacks to API clients
        return jsonify({"error": "Something went wrong while answering your question. Please try again."}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
