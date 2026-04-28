import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def create_or_update_index(chunks: List[dict], embedding_model, index_path: str):
    """
    Creates a FAISS index from chunks and persists it to local storage.
    
    Args:
        chunks: List of dicts with 'content' and 'metadata'.
        embedding_model: The OpenAIEmbeddings instance.
        index_path: Directory where the FAISS index will be saved.
    """
    # 1. Convert dictionaries into LangChain Document objects
    documents = [
        Document(page_content=c["content"], metadata=c["metadata"]) 
        for c in chunks
    ]
    
    # 2. Build the FAISS index (this calls the OpenAI Embedding API)
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # 3. Save locally to avoid re-embedding next time
    vector_store.save_local(index_path)
    print(f"Index successfully saved to: {index_path}")
    return vector_store

def load_local_index(index_path: str, embedding_model):
    """
    Loads an existing FAISS index from disk.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No index found at {index_path}")
        
    return FAISS.load_local(
        index_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )