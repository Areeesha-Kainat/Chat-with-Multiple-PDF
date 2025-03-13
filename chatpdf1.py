# import streamlit as st
# from PyPDF2 import PdfReader
# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Ensure API key is set
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

# # Function to extract text from uploaded PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             extracted_text = page.extract_text()
#             if extracted_text:
#                 text += extracted_text
#     return text

# # Function to split text into manageable chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Function to create FAISS vector store for embeddings
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Function to set up the question-answering model
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as accurately and thoroughly as possible based on the provided context.
#     If the answer is not available in the context, respond with: "The answer is not available in the provided context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """

#     # Use ChatGoogleGenerativeAI for correct integration
#     model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Function to handle user input and process queries
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     # Ensure FAISS index exists before loading
#     if not os.path.exists("faiss_index/index.faiss"):
#         st.error("FAISS index not found. Please upload and process PDFs first.")
#         return

#     # Load FAISS index for similarity search
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     # Get the conversational chain and generate a response
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#     # Display the response
#     st.write("Reply: ", response["output_text"])

# # Main function to handle Streamlit UI
# def main():
#     st.set_page_config(page_title="Chat PDF", layout="wide")
#     st.header("Chat with PDF using Gemini 💁")

#     # User input field for questions
#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     # Sidebar for file upload and processing
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit to Process", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Processing Complete!")

# # Run the app
# if __name__ == "__main__":
#     main()



# ===================== updated code ======================
import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Directly set the API key (Not recommended for production)
GOOGLE_API_KEY = "AIzaSyAirGKFXwyaMurrXF1K8ZAFcUwoUN7LqyE"

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create FAISS vector store for embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the question-answering model
def get_conversational_chain():
    prompt_template = """
    Answer the question as accurately and thoroughly as possible based on the provided context.
    If the answer is not available in the context, respond with: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3, google_api_key=GOOGLE_API_KEY)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user input and process queries
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index not found. Please upload and process PDFs first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])

# Main function to handle Streamlit UI
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit to Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")

if __name__ == "__main__":
    main()
