from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (make sure your OpenAI key is in the .env file as OPENAI_API_KEY)
load_dotenv()

# List of URLs to load
urls = [
    'https://www.bbc.co.uk/news/articles/cz95n2837vgo',
    'https://www.bbc.co.uk/news/live/cn4jjw30d5qt',
    'https://www.bbc.co.uk/sport/football/live/c5yr06pwzyyt'
]

# Load and process web content
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
# print(data)       # Consolidated data from the url's

# Split data into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)
print('Total number of split docs:', len(docs))
#print(docs[0])

# Create a vector store with OpenAI embeddings
vector_store = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())

# Create a retriever
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 6})

# Language model for response generation
llm = ChatOpenAI(temperature=0.4, max_tokens=500)

# Prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create chain components
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Ask a question
response = rag_chain.invoke({"input": "Tell me the improvements of Universal theme park planned in UK?"})
print("\nAnswer:", response["answer"])





