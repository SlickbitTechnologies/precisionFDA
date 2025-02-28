import streamlit as st
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.ollama import Ollama

import traceback
from models.models import get_local_model
from engine.engine import get_chat_engine, get_query_engine, get_flare_query_engine
from settings.settings import RAGSettings
from vector_store.vector_store import get_or_create_vector_index
from llama_index.core import ServiceContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer

# Set page configuration
settings = RAGSettings()
st.set_page_config(
    layout="wide",
    page_title="Precision FDA",  # Browser tab title
    page_icon='📄',  # Emoji or path to an image file
)
# App header
st.header(
    "Precision FDA", )


def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []


# Load the local Llama 3.2 model

# if st.session_state.history is None:
#     st.session_state.history = []

# Cache the vector index to avoid reloading on every rerun
@st.cache_data
def load_vector_index():
    return get_or_create_vector_index()


# Load the vector index
with st.spinner("Loading document index..."):
    vector_index = load_vector_index()

# Check if the vector index was loaded successfully
if vector_index is None:
    st.error("Failed to load the document index. Please check the logs for more details.")
    st.stop()

# Set up the local Llama model with LlamaIndex
llm = get_local_model()

# Create a service context with the local Llama model

# Create a query engine with the local Llama model
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

query_engine = get_query_engine()

chat_engine = vector_index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about an essay discussing Paul Grahams life."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
    response_mode=ResponseMode.GENERATION
)
condense_engine = get_chat_engine()
# User input
user_query = st.chat_input("Ask a question:", key="user_query")

# Process the query
if user_query:
    with st.spinner("Searching for an answer..."):
        try:
            # Process the query using the local Llama model
            st.session_state.messages.append({"role": 'user', "content": user_query})
            response = query_engine.query(user_query.strip())
            answer = response.response
            if len(answer) > 100:
                answer = answer.replace("Hello, I'm Precision FDA.", "").strip()
                answer = answer.replace("Hello, how can I help you?","").strip()
                answer = answer.replace("how can I help you?","").strip()
            docs = query_engine.retrieve(user_query)
            for doc in docs:
                print(doc.get_score())

            # Display the response
            st.session_state.messages.append({"role": 'assistant', "content": answer})
            # st.write(response.response)  # Assuming `response.response` contains the answer text
        except Exception as e:
            print(e)
            traceback.print_exc()
            st.error(f"An error occurred while processing your query: {e}")


def update_ui():
    if len(st.session_state.messages) > 0:
        for item in st.session_state.messages:
            with st.chat_message(item['role']):
                st.markdown(item['content'])


def main():
    initialize_session_state()
    update_ui()


if __name__ == "__main__":
    main()
