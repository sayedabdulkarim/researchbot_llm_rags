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
# using custom dataset #

pdf_reader = PyPDFLoader('./RAGPaper1.pdf')
documents = pdf_reader.load()

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# Use the text splitter to split documents
chunks = text_splitter.split_documents(documents)

# Generate embeddings using the local LLM
def generate_embedding(text, model="llama3.2:1b"):
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model,
        "input": text
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            # print("Response:", response.json())
            return response.json()['embeddings']  # Check if "embedding" exists
        else:
            # print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("Error:", e)
        return None
chunk_texts = [chunk.page_content for chunk in chunks]

# Print results for verification
print("start Generating embeddings ....")
# print(chunk_texts, ' chunk texts')
# for i, embedding in enumerate(embeddings[:5]):  # Display the first 5 embeddings for sanity check
#     print(f"Chunk {i + 1}: {embedding}")

# Create embeddings for chunks
arr = ['Cust om Element Suppor t\nR eact no w fully suppor t s cust om element s and handles pr oper ties/\nattribut es consist ent ly .\nI m p r o v e d  T h i r d - P a r t y  S c r i p t  C o m p a t i b i l i t y\nUne xpect ed tags in <head> and <body> ar e skipped during h y dration, \na v oiding mismat ch err ors.\nR eact Ser v er Component s\nSer v er -r ender ed component s t hat e x ecut e at build time or per r equest.\nA ctions\nAsync functions t hat handle f orm submission, err or stat es, and \noptimistic updat es aut omatically .\nuseActionState\nManages f orm stat e, pr o viding degraded e xperiences when Ja v aScript \nis una v ailable.\nuseFormStatus\nAccess t he status of a par ent f orm wit hout pr op drilling.\nuseOptimistic\nSho w optimistic stat e while async r equest s ar e in pr ogr ess.\nA sync Script Suppor t\nR ender async script s an ywher e in y our component tr ee, wit h aut omatic \ndeduplication.\nBett er Err or R epor ting', "A sync Script Suppor t\nR ender async script s an ywher e in y our component tr ee, wit h aut omatic \ndeduplication.\nBett er Err or R epor ting\nAut omatically de-duplicat es err ors, and intr oduces onCaughtError  \nand onUncaughtError  handlers f or r oot component s.\nuseDeferredValue Initial V alue\nT he useDeferredValue  hook no w suppor t s an initial v alue.\nuse\nR eads r esour ces lik e pr omises or cont e xt during r ender , allo wing f or \nconditional use.\nR e f  Callbac k  Cleanup\nR ef callbacks can no w r eturn a cleanup function.\nStr eamlined Cont e x t API\nUse < C o nt e xt > dir ect ly inst ead of < C o nt e xt.Pr o vi de r >.\nH y dr ation Err or D i ff s\nI mpr o v ed err or logging f or h y dration err ors, pr o viding a detailed diff \nwhen mismat ches occur .\nref as a Pr op\nP ass r ef s dir ect ly as pr ops in function component s.\nD ocument M etadata Suppor t\nAut omatically hoist s < titl e>, < m e t a>, and < link > tags t o t he \n<head>.\n' use client '", "ref as a Pr op\nP ass r ef s dir ect ly as pr ops in function component s.\nD ocument M etadata Suppor t\nAut omatically hoist s < titl e>, < m e t a>, and < link > tags t o t he \n<head>.\n' use client '\nMarks code t hat can be r ef er enced b y t he ser v er component and can \nuse client -side R eact f eatur es.\n' use ser v er '\nMarks ser v er -side functions callable fr om client -side code.\nStylesheet s w it h Pr ecedence\nSuppor t f or inser ting stylesheet s wit h pr ecedence in concurr ent \nr endering en vir onment s.\nR esour ce Pr eloading APIs\nP r eload r esour ces lik e f ont s, script s, and styles t o optimi z e \nper f ormance.\nR eact 19  Cheat Sheet b y K ent C.  Dodds EpicR eact.de v"]
embedding = generate_embedding(chunk_texts)
# embedding = generate_embedding("Here is an article about llamas...")
# if embedding:
#     print("Generated Embedding hello:", embedding)
# else:
#     print("Failed to generate embedding.")
#
# # Store embeddings in FAISS
# db = FAISS.from_documents(documents=chunks, embedding=embeddings)

print(embedding, 'hello')
# print(chunk_texts)