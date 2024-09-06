from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# Define the path to the dataset directory containing PDF and text files
data_path = r"C:\Users\Computing\Downloads\LLM-T5-Model (2)\Flan-T5-Model\RAG-Dataset" 

# Define the path where the FAISS vector database will be saved
Db_faiss_path = "Vector_Data_Base_GTR_T5_Large"

# Custom class to load text files
class TextFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    # Method to load text content from the file and wrap it into a Document object
    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [Document(page_content=text, metadata={'source': self.file_path})]

# Function to create a vector database from documents
def create_vector_db():
    documents = []  # Initialize a list to hold all documents

    # Load PDF documents from the specified directory using DirectoryLoader
    pdf_loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()  # Load all PDFs into documents
    documents.extend(pdf_documents)

    # Load all text files in the specified directory
    txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    for txt_file in txt_files:
        # Use the custom TextFileLoader to load text documents
        txt_loader = TextFileLoader(os.path.join(data_path, txt_file))
        txt_documents = txt_loader.load()
        documents.extend(txt_documents)

    # Split the loaded documents into smaller chunks using a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create embeddings using a pre-trained HuggingFace model (GTR-T5-Large)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                        model_kwargs={'device': 'cuda'})
    # Create a FAISS vector store from the document chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)

    # Save the FAISS database to the specified local path
    db.save_local(Db_faiss_path)

# Entry point for the script execution
if __name__ == '__main__':
    create_vector_db()  # Call the function to create and save the vector database
