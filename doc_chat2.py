import os
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Set your  API key
#os.environ["KEY"] = "somekey"

# Define input directory
input_dir_path = "/Users/anad/Agents/docs"

# Load data
loader = SimpleDirectoryReader(
    input_dir=input_dir_path,
    required_exts=[".pdf"],
    recursive=True
)
docs = loader.load_data()

# Define embedding model
model_name = "Snowflake/snowflake-arctic-embed-m"  # or any model you prefer
embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

# Set up Qdrant client
collection_name = "chat_with_docs"
client = qdrant_client.QdrantClient(host="localhost", port=6333)

# Create the vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name
)

# Prepare nodes
nodes = docs  # Ensure this is in the format expected by VectorStoreIndex

# Create storage context and index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

# Set up the query engine
qa_prompt_tmpl_str = (
    "Context information is below. \n"
    "----------------------------- \n"
    "{context_str}\n"
    "----------------------------- \n"
    "Given the context information above I want you \n"
    "to think step by step to answer the query in a crisp \n"
    "manner; in case you don't know the answer say \n"
    "'I don't know!'. \n"
    "Query: {query_str}\n"
    "Answer: "
)

# Update prompts for the query engine
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

# Set up the LLM
llm = Ollama(model="llama3", request_timeout=120.0)
Settings.llm = llm  # Specify the LLM to be used

# Query the engine
response = query_engine.query("What is this document about?")
print(response)
