from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")# Corrected




llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=groq_api_key
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load the CSV data
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)

    # Split documents into chunks
    docs = text_splitter.split_documents(data)

    # Create FAISS vector database from the document chunks
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)

    # Save the vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the FAISS vector database
    vectordb = FAISS.load_local(vectordb_file_path, embeddings,  allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()

    # Define the prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type( llm = llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    return chain

if __name__ == "__main__":
    # Create the vector database if not already created
    create_vector_db()

    # Create the RetrievalQA chain
    chain = get_qa_chain()

    # Run the QA chain with a sample query
    result = chain("Why should I trust Codebasics?")

    # Output the result
    print(result)
