import os
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(file_path: Path) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def load_txt(file_path: Path) -> str:
    """Extracts text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def ingest_data(directory_path: str, allowed_extensions: list = [".pdf", ".txt"]) -> list[Document]:
    """
    Scans a directory and returns a list of LangChain Document objects.
    
    Returns:
        List[Document]: Each document has .page_content and .metadata['source']
    """
    documents = []
    path = Path(directory_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    print(f"Scanning directory: {directory_path}")
    
    for file_path in path.iterdir():
        # Skip directories and non-allowed files
        if file_path.is_dir() or file_path.suffix.lower() not in allowed_extensions:
            continue
            
        content = ""
        if file_path.suffix.lower() == ".pdf":
            content = load_pdf(file_path)
        elif file_path.suffix.lower() == ".txt":
            content = load_txt(file_path)
        
        if content.strip():
            # Create a LangChain Document object
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,  # Just the filename for cleaner citations
                    "full_path": str(file_path) # Useful if you want to open the file later
                }
            )
            documents.append(doc)

    print(f"Successfully ingested {len(documents)} files.")
    return documents