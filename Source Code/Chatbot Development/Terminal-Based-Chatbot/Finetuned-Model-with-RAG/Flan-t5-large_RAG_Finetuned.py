import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# System prompt for the model
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the pre-trained model and tokenizer for sequence-to-sequence tasks
model_path = r"C:\Users\Computing\Downloads\Flan-T5-Model\Flan-T5-Model\T5-Finetune-Arunabh"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  # Load the model
tokenizer = AutoTokenizer.from_pretrained(model_path)      # Load the tokenizer

# Initialize embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})  # Initialize embeddings with GPU support

# Load the FAISS vector store from disk
Db_faiss_path = r"C:\Users\Computing\Downloads\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)  # Load FAISS index
# Configuration for text generation to control the sampling strategy and output format
generation_config = GenerationConfig(
    do_sample=True,             # Enable sampling instead of greedy decoding
    top_k=1,                    # Use only the top 1 token (deterministic output)
    temperature=0.1,            # Low temperature to make the output more deterministic and less random
    max_new_tokens=150,         # Limit the maximum number of tokens in the generated response
    pad_token_id=tokenizer.eos_token_id  # Set padding token ID to the model's EOS token ID
)
# Function to retrieve context documents for a given query
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)  # Embed the query
    docs = db.similarity_search_by_vector(query_embedding, k)  # Retrieve documents from FAISS
    ranked_docs = rank_documents(query, docs)  # Rank the retrieved documents
    return ranked_docs[:5]  # Return top 5 documents

# Function to rank documents (currently returns them as is)
def rank_documents(query, docs):
    return docs  # No additional ranking applied

# Function to generate an answer using the model
def generate_answer(query, context):
    # Combine the system prompt, user query, and context for input
    input_text = system_prompt + "\nUser: " + query + "\nBot: Please provide a detailed response."
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)  # Tokenize input
    
    # Generate a response with specified parameters
    outputs = model.generate(**inputs,generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the generated output
    return answer.strip()  # Return the cleaned answer

# Function to get a response including context, sources, and response time
def get_response(question):
    context = retrieve_context(question)  # Get context for the question
    start_time = time.time()  # Record start time for performance measurement
    answer = generate_answer(question, context)  # Generate the answer
    response_time = time.time() - start_time  # Calculate response time
    sources = [doc.metadata['source'] for doc in context]  # Extract source information from context
    return answer, sources, response_time  # Return the answer, sources, and response time

# Main function to interact with the user
def main():
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == 'exit':  # Exit condition
            print("Goodbye!")
            break
        response, sources, response_time = get_response(user_input)  # Get the chatbot response
        print("Bot:", response)  # Print the generated response
        print("Sources:", sources)  # Print the sources of the context
        print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Print response time

if __name__ == "__main__":
    main()  # Run the main function when the script is executed
