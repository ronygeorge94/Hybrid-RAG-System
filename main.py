"""
Main entry point for the RAG system.

Orchestrates configuration loading, model initialization, vector store setup,
and interactive query processing.
"""

import os
import sys
from pathlib import Path
from typing import List
from unittest import result
from src.config_loader import load_config, AppConfig
from src.embedding import get_embedding_model
from src.index import create_or_update_index, load_local_index
from src.llm import get_llm
from src.rag_pipeline import get_rag_chain, generate_answer
from src.ingest import ingest_data
from src.chunking import chunk_documents
from src.retriever import get_retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.utils import load_prompt_text


def main() -> None:
    """
    Initialize and run the RAG system interactively.
    """
    # 1. Load configuration from YAML file
    config: AppConfig = load_config()

    # 2. Initialize embedding and language models
    print("Initializing models...")
    embed_model = get_embedding_model(config.embedding.model_name)
    llm = get_llm(
        provider="openai",
        model_name=config.llm.model_name,
        temperature=config.llm.temperature
    )

    # 3. Manage vector store
    vdb_path: str = config.paths.vector_store_dir
    index_file: str = os.path.join(vdb_path, "index.faiss")

    # Process documents once for efficiency
    print("Processing documents...")
    raw_docs: List[Document] = ingest_data(config.paths.raw_data_dir, config.ingestion.allowed_extensions)
    chunks: List[Document] = chunk_documents(
        raw_docs, 
        chunk_size=config.chunking.size, 
        chunk_overlap=config.chunking.overlap
    )

    if not os.path.exists(index_file):
        print("Building knowledge base...")
        vector_db: FAISS = create_or_update_index(chunks, embed_model, vdb_path)
    else:
        print("Loading existing vector index...")
        vector_db: FAISS = load_local_index(vdb_path, embed_model)

    # 4. Retrieval & Prompt Setup
    print("Setting up retrieval and prompts...")
    retriever = get_retriever(vector_db, llm, chunks, k=3)
    
    # --- NEW CHANGES START HERE ---
    # Load the custom instructions from the .md file
    # Ensure your config.yaml has: paths -> prompt_file: "configs/prompts/rag_prompt.md"
    prompt_template = load_prompt_text(config.paths.prompt_file)

    # Assemble RAG chain - now passing the custom template
    rag_chain = get_rag_chain(llm, retriever, prompt_template)
    # --- NEW CHANGES END HERE ---

    print("\nRAG system ready. Type 'exit' to quit.")
    
    # Interactive query loop
    while True:
        query: str = input("\nUser Query: ")
        
        if query.lower() in ["exit", "quit"]:
            break
            
        if not query.strip():
            continue

        # main.py inside the While loop:

        print("Searching and generating answer...")

        # Execute the chain
        result = rag_chain.invoke(query)

        answer = result["answer"]
        sources = result["context"] # This is a list of Document objects

        print(f"\nAssistant: {answer}")

        # Print clickable or clear sources at the bottom
        if sources:
            print("\nSOURCES USED:")
            unique_sources = set([doc.metadata.get("source") for doc in sources])
            for i, source in enumerate(unique_sources):
                print(f"[{i+1}] {os.path.basename(source)}")

if __name__ == "__main__":
    main()