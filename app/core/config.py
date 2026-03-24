from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    openai_temperature: float = 0.0

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    similarity_threshold: float = 0.7

    # Vector store
    vector_store_type: str = "faiss"  # "faiss" | "azure"
    faiss_index_path: str = "data/faiss_index"

    # Azure AI Search
    azure_search_endpoint: str = ""
    azure_search_key: str = ""
    azure_search_index_name: str = "rag-documents"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
