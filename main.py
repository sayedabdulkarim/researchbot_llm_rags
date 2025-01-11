from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import requests


llm = OllamaLLM(
    base_url="http://localhost:11434",  # Ensure the Ollama server is running locally
    model="llama3.2:1b",
    verbose=True
)

# response = llm.invoke("a small joke")
# print(response)

# using custom dataset #

pdf_reader = PyPDFLoader('./RAGPaper1.pdf')
documents = pdf_reader.load()

# Create a text splitter
# RecursiveCharacterTextSplitter is a text splitter that splits the text into chunks,
# trying to keep paragraphs together and avoid losing context over pages.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# Use the text splitter to split documents
chunks = text_splitter.split_documents(documents)

# Generate embeddings using the local LLM
def generate_local_embeddings_via_api(texts, model="llama3.2:1b"):
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model,
        "input": texts
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
            return response.json().get("embedding")  # Check if "embedding" exists
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("Error:", e)
        return None
#

def generate_embedding(text, model="llama3.2:1b"):
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model,
        "input": text
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
            return response.json().get("embedding")  # Check if "embedding" exists
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("Error:", e)
        return None
chunk_texts = [chunk.page_content for chunk in chunks]
# embeddings = generate_local_embeddings_via_api(chunk_texts, model="llama3.2:1b")

# Print results for verification
print("Generated embeddings:")
# for i, embedding in enumerate(embeddings[:5]):  # Display the first 5 embeddings for sanity check
#     print(f"Chunk {i + 1}: {embedding}")

# # Create embeddings for chunks
# chunk_texts = [chunk.page_content for chunk in chunks]
# embeddings = generate_local_embeddings_via_api(chunk_texts, llm)
embedding = generate_embedding(chunk_texts)
if embedding:
    print("Generated Embedding:", embedding)
else:
    print("Failed to generate embedding.")
#
# # Store embeddings in FAISS
# db = FAISS.from_documents(documents=chunks, embedding=embeddings)

print('hello')
# print(chunk_texts)
# print('embeddding')
# print(embeddings)