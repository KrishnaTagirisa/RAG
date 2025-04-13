# Import necessary libraries
import os                                       # For interacting with the operating system
import chromadb                                 # Chroma DB client for vector storage and retrieval
from dotenv import load_dotenv                  # For loading environment variables from a .env file
from openai import OpenAI                       # OpenAI client for API access
from chromadb.utils import embedding_functions  # Utility to use OpenAI embeddings with Chroma

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI embedding function using a specific model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Initialize Chroma client with persistent storage
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")

# Define or create a collection to store document embeddings
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

# Initialize the OpenAI API client
client = OpenAI(api_key=openai_key)

# Example chat API call (commented out)
# This shows how to call GPT for a chat completion
# resp = client.chat.completions.create(...)

# ============================
# Function Definitions
# ============================

# Load text documents from a given directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Only load .txt files
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Split large text into smaller overlapping chunks for embedding
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # Create overlapping chunks
    return chunks

# ============================
# Load and Preprocess Documents
# ============================

# Set the directory containing the documents
directory_path = "./news_articles"

# Load documents from the directory
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

# Split each document into smaller chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# ============================
# Embedding and Database Insertion
# ============================

# Generate embeddings using OpenAI API
def get_openai_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    print("==== Generating embeddings... ====")
    return embedding

# Generate and attach embeddings to document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_openai_embedding(doc["text"])

# Insert (upsert) chunks into Chroma DB
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[doc["embedding"]]
    )

# ============================
# Query and Response Functions
# ============================

# Query Chroma for relevant document chunks based on a user question
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)

    # Flatten list of lists into a single list of relevant document texts
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

# Generate a concise answer based on retrieved document chunks
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    answer = response.choices[0].message
    return answer

# ============================
# Example Usage
# ============================

# Example user question
question = "tell me about databricks"

# Query the Chroma collection for relevant document chunks
relevant_chunks = query_documents(question)

# Generate and print a response using GPT model
answer = generate_response(question, relevant_chunks)
print(answer)
