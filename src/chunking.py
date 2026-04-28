from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def chunk_documents(raw_docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks using recursive character splitting.

    This function uses LangChain's RecursiveCharacterTextSplitter to divide documents
    into manageable chunks while preserving context through overlapping text segments.

    Args:
        raw_docs: List of Document objects to be chunked.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    # Initialize the text splitter with specified chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the documents into chunks
    # The splitter handles Document objects directly
    final_chunks = splitter.split_documents(raw_docs)

    return final_chunks