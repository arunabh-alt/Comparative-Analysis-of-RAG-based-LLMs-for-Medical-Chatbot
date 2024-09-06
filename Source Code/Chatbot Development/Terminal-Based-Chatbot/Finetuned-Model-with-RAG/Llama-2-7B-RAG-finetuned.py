import time
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Define the system prompt that guides the assistant's response style and behavior
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the tokenizer for the language model
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\Two Models\llm-llama\llm-llama\llama-2-7b-arunabh")

# Load the model with configurations for efficient memory usage and GPU support
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\Two Models\llm-llama\llm-llama\llama-2-7b-arunabh",
    low_cpu_mem_usage=True,      # Optimize for lower CPU memory usage
    return_dict=True,            # Return outputs as a dictionary
    torch_dtype=torch.float16,   # Use float16 precision to save memory
    device_map="cuda"            # Map the model to GPU for faster processing
)

# Configure settings for text generation
generation_config = GenerationConfig(
    do_sample=True,              # Enable sampling to allow variability in responses
    top_k=1,                     # Restrict sampling to the top-k most likely tokens (k=1 for deterministic output)
    temperature=0.1,             # Control the randomness of responses (lower value results in more deterministic output)
    max_new_tokens=150,          # Maximum number of tokens to generate
    pad_token_id=tokenizer.eos_token_id  # Define the padding token ID to handle sequences of varying lengths
)

# Initialize embeddings and load the FAISS vector store for context retrieval
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})  # Use GPU for embeddings

# Load the FAISS database from disk
Db_faiss_path = r"C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_context(query, k=10):
    """
    Retrieve relevant context documents for the provided query from the FAISS database.

    Args:
        query (str): The user's query.
        k (int): The number of closest documents to retrieve.

    Returns:
        List[Document]: The top k relevant documents.
    """
    query_embedding = embeddings.embed_query(query)  # Compute the query embedding
    docs = db.similarity_search_by_vector(query_embedding, k)  # Retrieve similar documents from FAISS
    ranked_docs = rank_documents(query, docs)  # Optionally rank the documents
    return ranked_docs[:5]  # Return the top 5 documents

def rank_documents(query, docs):
    """
    Rank the retrieved documents based on relevance (currently, this function just returns the documents as-is).

    Args:
        query (str): The user's query.
        docs (List[Document]): The retrieved documents.

    Returns:
        List[Document]: The ranked documents.
    """
    return docs  # No additional ranking is applied; documents are returned in their retrieved order

def generate_answer(query, context):
    """
    Generate an answer to the user's query using the provided context.

    Args:
        query (str): The user's query.
        context (List[Document]): The context documents to inform the answer.

    Returns:
        str: The generated answer.
    """
    # Combine context documents into a single string
    context_text = "\n\n".join([doc.page_content for doc in context])
    # Formulate the input text for the model including the system prompt and user query
    input_text = system_prompt + "\nContext: " + context_text + "\nUser: " + query + "\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize the input text and move to GPU
    outputs = model.generate(**inputs, generation_config=generation_config)  # Generate the model's response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the response from tokens to text
    return answer.strip()  # Return the answer with leading and trailing whitespace removed

def get_response(question):
    """
    Process the user's question, retrieve context, and generate a response.

    Args:
        question (str): The user's question.

    Returns:
        Tuple[str, List[str], float]: The generated answer, sources of the context documents, and response time.
    """
    context = retrieve_context(question)  # Retrieve relevant context for the question
    start_time = time.time()  # Start timing the response generation
    answer = generate_answer(question, context)  # Generate the answer based on the context
    response_time = time.time() - start_time  # Calculate the time taken to generate the response
    sources = [doc.metadata['source'] for doc in context]  # Extract sources of the context documents
    return answer, sources, response_time  # Return the answer, sources, and response time

def main():
    """
    Main function to handle user interaction through command-line input.
    """
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == 'exit':  # Check if the user wants to exit
            print("Goodbye!")  # Print farewell message
            break
        response, sources, response_time = get_response(user_input)  # Get the chatbot's response
        print("Bot:", response)  # Print the response from the chatbot
        print("Sources:", sources)  # Print the sources of the context documents
        print(f"Time Taken to Respond: {response_time:.2f} seconds")  # Print the time taken to generate the response

if __name__ == "__main__":
    main()  # Run the main function if this script is executed
