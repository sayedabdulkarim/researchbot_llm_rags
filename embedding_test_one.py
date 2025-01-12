import requests

# Function to send a request to the base endpoint
def generate_embedding(text, model="llama3.2:1b"):
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": model,
        "input": text
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            response_data = response.json()  # Store the JSON response as a dictionary
            print("Response:", response_data)  # Print full response for debugging
            return response_data.get("embedding")  # Check if "embedding" exists
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("Error:", e)
        return None


# Preprocess text to remove unnecessary characters
def preprocess_text(text):
    # Example preprocessing: Remove excessive whitespace, newlines, etc.
    return " ".join(text.split())


# Function to split long text into smaller chunks
def split_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Test the function
arr = [
    "Custom Element Support: React now fully supports custom elements and handles properties/attributes consistently.",
    "Marks server-side functions callable from client-side code. Stylesheets with Precedence: Support for inserting stylesheets with precedence in concurrent rendering environments. Resource Preloading APIs: Preload resources like fonts, scripts, and styles to optimize performance. React 19 Cheat Sheet by Kent C. Dodds EpicReact.dev"
]

# Loop through the list and process each input
embeddings = []
for text in arr:
    preprocessed_text = preprocess_text(text)  # Preprocess the text
    chunks = split_text(preprocessed_text, max_length=1000)  # Split text into smaller chunks
    for chunk in chunks:
        embedding = generate_embedding(chunk)
        if embedding:
            embeddings.append(embedding)
            print("Generated Embedding for chunk:", embedding[:10])  # Print the first 10 values for brevity
        else:
            print("Failed to generate embedding for chunk:", chunk)

# Check if embeddings were generated
if embeddings:
    print(f"Generated {len(embeddings)} embeddings.")
else:
    print("No embeddings were generated.")
