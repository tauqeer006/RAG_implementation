from langchain.vectorstores import FAISS
import os 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import getpass
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain_groq import ChatGroq
import logging
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from transformers import pipeline
from huggingface_hub import login
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def compute_Elidian_Distance(x,y):
    x1 = np.array(x)
    x2 = np.array(y)
    return np.sqrt(np.sum(x1-x2)**2)
    
    
def google_API_embedding():
    os.environ["Google_API_KEY"] = os.getenv("google_key")
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embedding_dim = len(embedding.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}, )
    return embedding
def embedding_model():
  model = "sentence-transformers/all-mpnet-base-v2"
  hf = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)
  return hf
  

def user_interface():
   embedding = embedding_model()

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
    logging.info(f"Document Content:\n{document.page_content}")

    fine_tuing(document)


def fine_tuing(document):
   print("HF_TOKEN from env:", os.getenv("HF_TOKEN")) 
   login(token= os.getenv("HF_TOKEN"))
   pipe = pipeline("text2text-generation", model="google/flan-t5-base", device_map="auto")
   rag_answer = document
   prompt = f"""
   Instruction: Improve the following answer given  to write in 1  sentence.
   
    Answer: {rag_answer}
   Refined Answer:
    """
   response = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]
   print("After fine tuning the text i get the response: ")
   print(response)
   
if __name__ == "__main__":
   user_interface()