from pydantic import BaseModel, Field
from typing import List
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PathConfig(BaseModel):
    raw_data_dir: str
    vector_store_dir: str
    prompt_file: str

class IngestionConfig(BaseModel):
    allowed_extensions: List[str]

class ChunkingConfig(BaseModel):
    size: int = Field(gt=0,le=1500, description="Chunk size must be between 1 and 1500")
    overlap: int

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str

class LLMConfig(BaseModel):
    model_name: str
    temperature: float

class AppConfig(BaseModel):
    paths: PathConfig
    ingestion: IngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig

def load_config(config_name: str = "config.yaml") -> AppConfig:
    config_path = PROJECT_ROOT / "configs" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    return AppConfig(**config_dict)