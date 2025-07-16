import logging
import sounddevice as sd
from scipy.io.wavfile import write
import whisper 
from langchain.vectorstores import FAISS
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import numpy as np
from transformers import pipeline
from huggingface_hub import login
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,  format="%(asctime)s - %(levelname)s - %(message)s"
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

def Recording(duration = 5):
    filename = "record.wav"
    sample_rate = 16000
    print("Recording being started: ")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording completed")
    write(filename ,  sample_rate , audio )
    print("audio is:",audio)
    speech_to_text()

def embedding_data():
   embedding = google_API_embedding()
   returned_vector = FAISS.load_local("faiss_index_dir", embedding , allow_dangerous_deserialization=True)
   vector_store_new = returned_vector
   v_index = vector_store_new.index
   stored_vectors = v_index.reconstruct_n(0, v_index.ntotal)
   return embedding , vector_store_new , stored_vectors
   
def speech_to_text():

    model = whisper.load_model("small")
    results = model.transcribe("record.wav")
    print("the text i am getting from speech to text model:",results["text"])
    transcript = results["text"]
    print(" Transcription (first 20 seconds):", transcript)
    embedding , vector_store_new , stored_vectors = embedding_data()
    query_vector = embedding.embed_query(transcript)
    distances = []
    for stored_vector in stored_vectors:
     distances.append(compute_Elidian_Distance(query_vector , stored_vector))
     top_index = np.argsort(distances)[:1]

    for idx in top_index:
     doc_id = vector_store_new.index_to_docstore_id[idx]
     document = vector_store_new.docstore._dict[doc_id]
     print("yes")
     print(f"Distance: {distances[idx]}")
     print(document.page_content)
     #logging.info("Document Content:\n", document.page_content)
     pass_to_llm(document)

def pass_to_llm(document):
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
   text_to_speech(response)

def text_to_speech(text):
   engine = pyttsx3.init()
   print("engine running:")
   engine.say(text)
   engine.runAndWait()




if __name__ == "__main__":
   logging.info("Enter the recording second you will record :")
   record = int(input())
   Recording(record)

   