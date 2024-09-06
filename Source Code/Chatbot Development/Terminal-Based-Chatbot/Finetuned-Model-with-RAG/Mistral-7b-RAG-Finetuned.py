import time
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Define the system prompt to guide the assistant's behavior and response style
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the tokenizer for the language model
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned")

# Load the fine-tuned model with specific configurations for efficient usage
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned",
    low_cpu_mem_usage=True,      # Optimize for low CPU memory usage
    return_dict=True,            # Return outputs as a dictionary
    torch_dtype=torch.float16,   # Use float16 precision to reduce memory usage
    device_map="cuda"            # Use GPU (CUDA) for model inference
)

# Configure settings for generating responses
generation_config = GenerationConfig(
    do_sample=True,              # Enable sampling to produce diverse outputs
    top_k=1,                     # Restrict sampling to the top-k tokens (k=1 for deterministic output)
    temperature=0.1,             # Control randomness in output; lower values make output more deterministic
    max_new_tokens=150,          # Set the maximum number of new tokens to generate
    pad_token_id=tokenizer.eos_token_id  # Define padding token ID
)

# Initialize embeddings and FAISS vector store for retrieving relevant context
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})  # Use GPU for embeddings

Db_faiss_path = r"C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)  # Load the FAISS database

def retrieve_context(query, k=10):
    """
    Retrieve context documents relevant to the query from the FAISS database.
    
    Args:
        query (str): The user query.
        k (int): Number of nearest neighbors to retrieve.
        
    Returns:
        List[Document]: A list of the top k relevant documents.
    """
    query_embedding = embeddings.embed_query(query)  # Get the embedding for the query
    docs = db.similarity_search_by_vector(query_embedding, k)  # Perform similarity search
    ranked_docs = rank_documents(query, docs)  # Rank the retrieved documents
    return ranked_docs[:5]  # Return the top 5 documents

def rank_documents(query, docs):
    """
    Rank the documents based on relevance (currently returns documents as-is).
    
    Args:
        query (str): The user query.
        docs (List[Document]): List of retrieved documents.
        
    Returns:
        List[Document]: Ranked list of documents.
    """
    return docs  # Currently returns documents without additional ranking due to the FAISS

def generate_answer(query, context):
    """
    Generate an answer based on the user query and context.
    
    Args:
        query (str): The user query.
        context (List[Document]): The context documents.
        
    Returns:
        str: The generated answer.
    """
    context_text = "\n\n".join([doc.page_content for doc in context])  # Combine context documents into a single string
    input_text = system_prompt + "\nContext: " + context_text + "\nUser: " + query + "\nBot:"  # Formulate the input text for the model
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move tensors to GPU
    
    # Generate a response using the model
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the generated output
    
    # Extract the answer part from the response
    answer_part = answer.split("Bot:")[-1].strip()
    return answer_part

def get_response(question):
    """
    Get a response for the user query including context retrieval and response generation.
    
    Args:
        question (str): The user query.
        
    Returns:
        str: The generated answer.
    """
    context = retrieve_context(question)  # Retrieve relevant context
    start_time = time.time()  # Start timer for measuring response time
    answer = generate_answer(question, context)  # Generate the answer
    response_time = time.time() - start_time  # Calculate response time
    sources = [doc.metadata['source'] for doc in context]  # Collect sources of context documents
    return answer

def main():
    """
    Main function to interact with the user via command line input.
    """
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")  # Read user input from the console
        if user_input.lower() == 'exit':  # Check if the user wants to exit
            print("Chatbot: Goodbye!")  # Print exit message
            break
        response = get_response(user_input)  # Get the response for the user input
        print("Chatbot:", response)  # Print the assistant's response

if __name__ == "__main__":
    main()  # Run the main function if this script is executed
