import streamlit as st
import tempfile
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

# Streamlit page config
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat with your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    # Retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # Groq LLM
    llm = ChatGroq(
    groq_api_key=groq_key,
    model_name="qwen/qwen3-32b"
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Question box
    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        with st.spinner("Generating answer..."):

            answer = qa_chain.invoke(question)

            if isinstance(answer, dict):
                answer_text = answer["result"]
            else:
                answer_text = answer

        # Save chat history
        st.session_state.messages.append({
            "question": question,
            "answer": answer_text
        })

    # Display chat history
    if st.session_state.messages:

        st.subheader("Chat History")

        for chat in reversed(st.session_state.messages):

            st.markdown(
                f"""
                **🧑 You:**  
                {chat['question']}
                """
            )

            st.markdown(
                f"""
                **🤖 Bot:**  
                {chat['answer']}
                """
            )

            st.divider()

else:
    st.info("Upload a PDF to start chatting.")