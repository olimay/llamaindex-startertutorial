import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os.path
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
# from llama_index.llms import PaLM
from llama_index.llms import OpenAI

# llm=PaLM()
llm=OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# either way we can now query the index
query_engine = index.as_query_engine(service_context=service_context)
querystring = """Create a five verse ballad with a refrain about what
the author did growing up.
"""
response = query_engine.query(querystring)
print(response)
