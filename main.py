import streamlit as st

from models.models import get_local_model
from engine.engine import  get_query_engine
from settings.settings import RAGSettings
from vector_store.vector_store import get_or_create_vector_index
from llama_index.core.memory import ChatMemoryBuffer


settings = RAGSettings()
st.set_page_config(
    layout="wide",
    page_title="PrecisionFDA Challenge",  # Browser tab title
    page_icon='📄',  # Emoji or path to an image file
)

col1, col2 = st.columns([1, 2])

with col1:
    st.image('Main.png', width=150)

with col2:
    st.header("PrecisionFDA Challenge" )

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat' not in st.session_state:
        st.session_state.chat = []


@st.cache_data
def load_vector_index():
    return get_or_create_vector_index()



with st.spinner("Loading document index..."):
    vector_index = load_vector_index()

if vector_index is None:
    st.error("Failed to load the document index. Please check the logs for more details.")
    st.stop()


llm = get_local_model()


memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

query_engine = get_query_engine()


user_query = st.chat_input("Ask a question:", key="user_query")

def find_answer(question):
    for chat in st.session_state.chat:
        if chat["question"] == question:
            return chat["answer"]
    return None


if user_query:
    with st.spinner("Searching for an answer..."):
        try:
            st.session_state.messages.append({"role": 'user', "content": user_query})
            query = user_query.strip()

            response = query_engine.query(query)
            answer = response.response
            if len(answer) > 100:
                answer = answer.replace("Hello, I'm Precision FDA.", "").strip()
                answer = answer.replace("Hello, how can I help you?","").strip()
                answer = answer.replace("how can I help you?","").strip()

            st.session_state.messages.append({"role": 'assistant', "content": answer})
            st.session_state.chat.append({
                "question":query,
                "answer":answer
            })
        except Exception as e:
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
