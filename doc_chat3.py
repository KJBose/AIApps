from llama_index.core import SimpleDirectoryReader

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

input_dir_path = "/Users/anad/Agents/docs"
# load data
loader = SimpleDirectoryReader(
    input_dir = input_dir_path,
    required_exts = [".pdf"],
    recursive = True
)

docs = loader.load_data()

# Define your local model
model_name = "CohereForAI/aya-101"  # Or any other model you prefer
embed_model = HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)

collection_name = "chat_with_docs"
client = qdrant_client.QdrantClient(
    host = "localhost",
    port = 6333
)

vector_store = QdrantVectorStore(
    client = client,
    collection_name = collection_name
)

# Prepare nodes (assuming docs are your nodes)
nodes = docs  # You can modify this if your nodes need different processing

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,)

qa_prompt_tmpl_str =(
    "Context information is below. \n"
    "----------------------------- \n"
    "{context_str}\n"
    "----------------------------- \n"
    "Given the conext information above I want you \n"
    "to think step by step to answer the query in a crisp \n"
    "manner, in case you don't know the answer say \n"
    "'I don't know!'. \n"
    "Query: {query_str}\n"
    "Answer: "  
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

response = query_engine.query("what is this document about?")
print(response)

#setting up the llm
llm = Ollama(model = "llama3", request_timeout=120.0)

# ====== Setup a query engine on the index previously created =====
Settings.llm = llm # specifying the llm to be used
query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)