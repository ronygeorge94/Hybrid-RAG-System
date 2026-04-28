"""
RAG Pipeline Implementation

This module provides the core components for a Retrieval-Augmented Generation (RAG) pipeline
using LangChain. It includes document formatting, chain assembly, and answer generation utilities.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document


def format_docs_with_id(docs):
    """Format docs so the LLM sees IDs for each source."""
    return "\n\n".join(
        f"Source [{i+1}]: {doc.page_content}\nMetadata: {doc.metadata.get('source')}"
        for i, doc in enumerate(docs)
    )

def get_rag_chain(llm, retriever, prompt_text):
    prompt = ChatPromptTemplate.from_template(prompt_text)

    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # FINAL CHAIN: Returns a dict with "answer" and "source_documents"
    full_chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(answer=rag_chain_from_docs)

    return full_chain

def generate_answer(chain, query: str) -> str:
    """
    Generate an answer using the RAG chain for the given query.

    Args:
        chain: The assembled RAG chain.
        query: The user query string.

    Returns:
        The generated answer as a string, or an error message if generation fails.
    """
    try:
        return chain.invoke(query)
    except Exception as e:
        return f"Error generating answer: {str(e)}"