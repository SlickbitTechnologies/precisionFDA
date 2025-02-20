from pydantic import BaseModel, Field


class OllamaSettings(BaseModel):
    llm:str = Field(
        default="llama3.2", description="LLM model"
    )
    port:int = Field(
        default=11434,description="Ollama port"
    )
    context_window: int = Field(
        default=8000, description="Context window size"
    )
    temperature: float = Field(
        default=0.1, description="Temperature"
    )
    host:str = Field(
        default='http://localhost',
        description="Ollama host path"
    )
    request_timeout:int = Field(
        default=300,
        description="Request Timeout"
    )
    embed_model:str = Field(
        default="nomic-embed-text"
    )


class RAGSettings(BaseModel):
    ollama: OllamaSettings = OllamaSettings()
