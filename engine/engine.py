from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.response_synthesizers import ResponseMode

from llama_index.core.query_engine import FLAREInstructQueryEngine
from models.models import get_embedding_model, get_local_model
from settings.settings import RAGSettings
from vector_store.vector_store import  get_or_create_vector_index
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    VectorIndexRetriever,

)

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
llm = get_local_model()

query_gen_prompt_en = PromptTemplate(
    "You are a skilled search query generator, dedicated to providing accurate and relevant search queries that are concise, specific, and unambiguous.\n"
    "Generate {num_queries} unique and diverse search queries, one on each line, related to the following input query:\n"
    "### Original Query: {query}\n"
    "### Please provide search queries that are:\n"
    "- Relevant to the original query\n"
    "- Well-defined and specific\n"
    "- Free of ambiguity and vagueness\n"
    "- Useful for retrieving accurate and relevant search results\n"
    "### Generated Queries:\n"
)
instruct_prompt = PromptTemplate(
            "You are Precision FDA, a helpful chatbot. "
            "Respond based on the retrieved context and chat history.\n\n"
            "1. If the user greets (e.g., 'Hi', 'Hello', 'Hey'), respond with:\n"
            "   'Hello, I'm Precision FDA. How can I help you?'\n\n"
            "2. If the user asks a question and relevant information is found in the documents, provide an accurate response.\n\n"
            "3. If no relevant information is found, respond with:\n"
            "   'Please, ask relative questions.'\n\n"
            "Here are the relevant documents for context:\n"
            "{context_str}\n"
            "Use the previous chat history and the context above to interact and help the user."
        )
vector_index = get_or_create_vector_index()
def get_query_gen_prompt(language: str):
    if language == "vi":
        return ""
    return query_gen_prompt_en

settings = RAGSettings()

def get_retriever():
    try:
        # vector_index = get_or_create_vector_index()
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=10,
            embed_model=get_embedding_model(),
            verbose=True
        )



        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever],
            llm=get_local_model(),
            query_gen_prompt=get_query_gen_prompt('en'),
            similarity_top_k=4,
            num_queries=2,
            verbose=True
        )
        return hybrid_retriever
    except Exception as e:
        print(e)
        return None


def get_chat_engine():
    try:
        return CondensePlusContextChatEngine.from_defaults(
            retriever=get_retriever(),
            llm=get_local_model(),
            memory=ChatMemoryBuffer(
                token_limit=3029
            )
        )
    except Exception as e:
        print(e)
        return None

COMPACT_ACCUMULATE = "compact_accumulate"\
    """
    Compact and accumulate mode first combine text chunks into larger consolidated \
    chunks that more fully utilize the available context window, then accumulate \
    answers for each of them and finally return the concatenation.
    This mode is faster than accumulate since we make fewer calls to the LLM.
    """
def get_query_engine() -> RetrieverQueryEngine:
    query_engine = RetrieverQueryEngine.from_args(
        retriever=vector_index.as_retriever(
            llm=llm,
            similarity_threshold=0.7
        ),
        llm=llm,
        response_mode=ResponseMode.REFINE,
        memory=memory,
        context_prompt=(
            "You are FDA Advisory Committee, a helpful chatbot. "
            "Respond based on the retrieved context and chat history.\n\n"
            "1. If the user greets (e.g., 'Hi', 'Hello', 'Hey'), respond with:\n"
            "   'Hello, how can I help you?'\n\n"
            "2. If the user asks a question and relevant information is found in the documents, provide an accurate response.\n\n"
            "3. If no relevant information is found, respond with:\n"
            "   'Please, ask relative questions.'\n\n"
            "Here are the relevant documents for context:\n"
            "{context_str}\n"
            "Response must detailed and elaborated manner"
            "Important Instructions:\n"
            "- Do not start responses with 'How can I help you?'\n"
            "- Do not end responses with 'Please, ask relative questions if you'd like more information.'\n"
            "- Keep answers precise and relevant without unnecessary introductions or conclusions."

        )
    )
    print(f"Yor are using {ResponseMode.REFINE}")
    return query_engine

def get_flare_query_engine():
    index_query_engine = vector_index.as_query_engine(similarity_top_k=2,llm=llm)
    query_engine = FLAREInstructQueryEngine(
        query_engine=index_query_engine,
        max_iterations=4,
        llm=llm,
        verbose=True,
        instruct_prompt=instruct_prompt
    )
    return query_engine