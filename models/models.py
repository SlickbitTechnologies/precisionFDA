from settings.settings import RAGSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

settings =RAGSettings()

def get_local_model(model:str=None):
    llm = Ollama(
        model=model or settings.ollama.llm,
        request_timeout=settings.ollama.request_timeout,
        temperature=0,
    )
    return llm


def get_embedding_model(model:str=None):
    ollama_embedding = OllamaEmbedding(
        model_name=model or settings.ollama.embed_model,
        base_url=f"{settings.ollama.host}:{settings.ollama.port}",
        ollama_additional_kwargs={"mirostat": 0},
    )
    return ollama_embedding