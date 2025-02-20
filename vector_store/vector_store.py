import logging
import traceback
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from models.models import get_embedding_model
from settings.settings import RAGSettings

INDEX_SAVE_DIR = "index_store_dir"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = RAGSettings()

def get_index_store(embed_model):
    logger.info(f"Loading index from {INDEX_SAVE_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_SAVE_DIR)
    # vector_store = storage_context.vector_store
    # logger.info(f"STORAGE CONTEXT {vs}")
    vector_index = load_index_from_storage(
        storage_context
    )



    return vector_index

def get_documents():
    documents = SimpleDirectoryReader(input_files=["MERGED_cosmetic_guidances.pdf"]).load_data()
    return documents

def create_nodes(embed_model,documents):
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    logger.info("Creating nodes from documents...")
    nodes = splitter.get_nodes_from_documents(documents,show_progress=True)
    logger.info(f"Nodes created: {len(nodes)}")
    return nodes

def create_vector_index(nodes,embed_model):
    logger.info("Creating vector store index...")
    vector_index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        show_progress=True
    )
    logger.info("Vector store index created.")

    # Save the index to disk

    vector_index.storage_context.persist(persist_dir=INDEX_SAVE_DIR)
    logger.info(f"Index saved to {INDEX_SAVE_DIR}.")
    return vector_index


def get_or_create_vector_index():
    try:

        logger.info("Loading embedding model...")
        embed_model = get_embedding_model()
        logger.info("Embedding model loaded.")
        if os.path.exists(INDEX_SAVE_DIR):
            return get_index_store(embed_model)

        documents = get_documents()
        if not documents:
            logger.error("No documents found.")
            return 0
        logger.info(f"Documents loaded: {len(documents)}")

        nodes = create_nodes(embed_model,documents)
        vector_index = create_vector_index(nodes,embed_model)
        return vector_index
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return None



