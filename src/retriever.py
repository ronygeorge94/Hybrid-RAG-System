from typing import List

from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM


def get_retriever(vector_db: FAISS, llm: BaseLLM, chunks: List[Document], k: int = 3) -> EnsembleRetriever:
    """
    Creates a hybrid retriever that combines vector search (semantic meaning)
    with BM25 search (keyword-based) for improved retrieval performance.

    This function implements an ensemble approach where dense vector embeddings
    are combined with sparse keyword matching to provide more comprehensive
    document retrieval.

    Args:
        vector_db: FAISS vector database containing embedded documents.
        llm: Language model instance (used for potential future enhancements).
        chunks: List of document chunks to build the BM25 keyword index.
        k: Number of top documents to retrieve from each retriever.

    Returns:
        EnsembleRetriever: A hybrid retriever combining vector and keyword search.

    Note:
        The ensemble weights are fixed at [0.7, 0.3] favoring semantic search
        over keyword search. This can be adjusted based on specific use cases.
    """

    # Create standard vector retriever (dense embeddings)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    # Create keyword retriever (sparse BM25)
    # Requires original document chunks to build keyword index
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = k

    # Create ensemble retriever combining both approaches
    # Weights: 70% semantic, 30% keyword-based
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


def get_multi_query_retriever(vector_db: FAISS, llm: BaseLLM) -> MultiQueryRetriever:
    """
    Creates a multi-query retriever that uses an LLM to generate multiple
    variations of the input query to improve recall.

    This approach helps overcome limitations of single-query retrieval by
    exploring different phrasings and perspectives of the same question.

    Args:
        vector_db: FAISS vector database for document retrieval.
        llm: Language model used to generate query variations.

    Returns:
        MultiQueryRetriever: A retriever that generates and uses multiple
        query variations for improved document retrieval.
    """
    return MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm
    )