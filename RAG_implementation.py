from langchain.vectorstores import FAISS
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import getpass
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain_groq import ChatGroq
import logging

def logging_setup():
      logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def compute_Elidian_Distance(x,y):
    x1 = np.array(x)
    x2 = np.array(y)
    return np.sqrt(np.sum(x1-x2)**2)
    
    
def google_API_embedding():
    os.environ["Google_API_KEY"] = "AIzaSyBy79jZcC5EhWjed0JWVDTY24KfhnhWeJk"
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embedding_dim = len(embedding.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

returned_vector = FAISS.load_local("faiss_index_dir", embedding , allow_dangerous_deserialization=True)

vector_store_new = returned_vector
v_index = vector_store_new.index

stored_vectors = v_index.reconstruct_n(0, v_index.ntotal)

data = input("Enter the question you want answer:")
query_vector = embedding.embed_query(data)

distances = []
for stored_vector in stored_vectors:
    distances.append(compute_Elidian_Distance(query_vector , stored_vector))

top_index = np.argsort(distances)[:1]


for idx in top_index:
    doc_id = vector_store_new.index_to_docstore_id[idx]
    document = vector_store_new.docstore._dict[doc_id]
    print("yes")
    logging.info(f"Distance: {distances[idx]}")
    logging.info("Document Content:\n", document.page_content)


os.environ["GROQ_API_KEY"] = "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"

    
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=100,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

fine_tuned_message = llm.invoke(document.page_content)
only_data =fine_tuned_message.additional_kwargs.get("reasoning_content" ,"")
logging.info("After Fine Tuning:")
logging.info(only_data)