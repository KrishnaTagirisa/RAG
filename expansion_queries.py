# === Import Required Libraries ===
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from helper_utils import project_embeddings, word_wrap

# LangChain for text splitting
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# ChromaDB for vector storage and retrieval
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# === Step 1: Environment Setup ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# === Step 2: Read and Preprocess PDF ===
reader = PdfReader("data/microsoft-annual-report.pdf")

# Extract and clean text from PDF pages
pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
full_text = "\n\n".join(pdf_texts)

# === Step 3: Text Splitting ===

# First split by characters and punctuation
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text(full_text)

# Further split using token limits
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256
)

# Token-level split of each character-split chunk
token_split_texts = []
for text in character_split_texts:
    token_split_texts.extend(token_splitter.split_text(text))

# === Step 4: Vector Store (ChromaDB) Setup ===

# Create embedding function
embedding_function = SentenceTransformerEmbeddingFunction()

# Initialize ChromaDB and create a collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection",
    embedding_function=embedding_function
)

# Add split text documents to Chroma with unique IDs
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)

# === Step 5: Querying the Document ===

# Example natural language query
query = "What was the total revenue for the year?"

# Retrieve top 5 matching documents
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# === Step 6: Generate Related (Augmented) Queries ===

def generate_multi_query(query, model="gpt-3.5-turbo"):
    """
    Use LLM to create 5 relevant sub-questions to the original financial query.
    """
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content.strip().split("\n")

# Generate expanded queries
original_query = "What details can you provide about the factors that led to revenue growth?"
aug_queries = generate_multi_query(original_query)

# Display generated sub-questions
for q in aug_queries:
    print("\n", q)

# Combine original + augmented queries
joint_query = [original_query] + aug_queries

# Run the combined queries through Chroma
results = chroma_collection.query(
    query_texts=joint_query,
    n_results=5,
    include=["documents", "embeddings"]
)

retrieved_documents = results["documents"]

# === Step 7: Deduplicate and Display Results ===

# Flatten results and remove duplicates
unique_documents = set()
for documents in retrieved_documents:
    unique_documents.update(documents)

# Print each set of results for every sub-query
for i, documents in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("\nResults:")
    for doc in documents:
        print(word_wrap(doc))
        print()
    print("-" * 100)

# === Step 8: Summarize Final Results with LLM ===

def summary_generated(query, model="gpt-3.5-turbo"):
    """
    Use LLM to summarize the retrieved relevant information.
    """
    prompt = """You are a helpful expert financial research assistant. 
    Provide a summarized version of the provided text."""

    messages = [
        {"role": "system", "content": query},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content

# Generate and display summaries
for i, documents in enumerate(retrieved_documents):
    combined_text = ' '.join([''.join(doc) for doc in documents])
    summarized_answer = summary_generated(combined_text)
    print(word_wrap(summarized_answer))
    print("-" * 100)
