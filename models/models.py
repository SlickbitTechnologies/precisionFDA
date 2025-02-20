from settings.settings import RAGSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

SYSTEM_PROMPT_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

PROMPT ="You are Precision FDA, a helpful assistant. "\
    "Respond based on the retrieved context and chat history.\n\n"\
    "1. If the user greets (e.g., 'Hi', 'Hello', 'Hey'), respond with: Hello, I'm Precision FDA. How can I help you?'\n\n"\
    "2. If the user asks a question and relevant information is found in the documents, provide an accurate response.\n\n"\
    "3. If no relevant information is found, respond with:'Please, ask relative questions.'\n\n"\
    "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context."\
    "Use the previous chat history and the context above to interact and help the user."
settings =RAGSettings()



def get_local_model(model:str=None):
    llm = Ollama(
        model=model or settings.ollama.llm,
        request_timeout=settings.ollama.request_timeout,
        system_prompt=PROMPT
    )
    return llm


def get_embedding_model(model:str=None):
    ollama_embedding = OllamaEmbedding(
        model_name=model or settings.ollama.embed_model,
        base_url=f"{settings.ollama.host}:{settings.ollama.port}",
        ollama_additional_kwargs={"mirostat": 0},
    )
    return ollama_embedding