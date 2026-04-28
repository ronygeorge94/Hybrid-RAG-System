from pathlib import Path

# Identify the project root relative to this file (src/utils.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_prompt_text(relative_path: str) -> str:
    """
    Reads a prompt template from a markdown or text file.
    Args:
        relative_path: Path relative to project root (e.g., 'configs/prompts/rag_prompt.md')
    """
    full_path = PROJECT_ROOT / relative_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Prompt file not found at: {full_path}")
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Error reading the prompt file: {e}")