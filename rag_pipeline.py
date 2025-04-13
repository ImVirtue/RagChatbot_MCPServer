from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Init components
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
model = ChatOpenAI(model="gpt-4")

template = """
You are a Rag chatbot assistant. Your responsibility is retrieving information from a PDF file and answering the user's questions.
Question: {question}
Context: {context}
Answer:
"""

# Setup prompt template
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load and index PDF (done only once, before answering)
pdf_path = "/home/davidntd/PycharmProjects/MCP/pdfs/work_rules_and_regulations_2016(2).pdf"
loader = PDFPlumberLoader(pdf_path)
documents = loader.load()

# Split and index
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
vector_store.add_documents(chunks)


def ask_from_pdf(question: str) -> str:
    relevant_docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    result = chain.invoke({"question": question, "context": context})
    return result.content
