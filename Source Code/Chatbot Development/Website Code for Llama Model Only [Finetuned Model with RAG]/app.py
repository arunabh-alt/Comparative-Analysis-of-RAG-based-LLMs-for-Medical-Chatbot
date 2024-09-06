from flask import Flask, render_template, request, jsonify
import time
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Initialize Flask app and specify the template folder path for rendering HTML pages
app = Flask(__name__, template_folder=r'C:\Users\Computing\Downloads\Two Models\llm-llama\llm-llama\website\template')

# System prompt used to instruct the model on how to respond to user queries
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the tokenizer from the specified path (Mistral-finetuned model)
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\Project\Llama\llama-2-7b-arunabh")

# Load the fine-tuned model for causal language modeling with specific configurations
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\Project\Llama\llama-2-7b-arunabh",
    low_cpu_mem_usage=True,          # Optimize memory usage for the model
    return_dict=True,                # Ensure the model returns output as a dictionary
    torch_dtype=torch.float16,       # Use half-precision to save GPU memory
    device_map="cuda"                # Map the model to the GPU for faster inference
)

# Configure generation settings for the model
generation_config = GenerationConfig(
    do_sample=True,                  # Enable sampling to introduce variability in the output
    top_k=1,                         # Use only the top 1 token for deterministic output
    temperature=0.1,                 # Low temperature for less random outputs
    max_new_tokens=150,              # Limit the maximum number of tokens generated
    pad_token_id=tokenizer.eos_token_id  # Use the EOS token as the padding token
)

# Load the HuggingFace embeddings model and the FAISS vector store for retrieving relevant documents
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
# Load the FAISS database from the specified path, allowing potentially unsafe deserialization
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

# Function to retrieve context documents from the FAISS vector store based on the user's query
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)  # Convert the query to an embedding
    docs = db.similarity_search_by_vector(query_embedding, k)  # Retrieve top k similar documents
    ranked_docs = rank_documents(query, docs)  # Rank the documents (currently a placeholder)
    return ranked_docs[:5]  # Return the top 5 documents

# Placeholder function for ranking documents (can be expanded with a more complex ranking logic)
def rank_documents(query, docs):
    return docs  # Currently just returns the documents as-is

# Function to generate an answer using the model, based on the query and the retrieved context
def generate_answer(query, context):
    # Combine the system prompt, context, and user query into a single input string
    context_text = "\n\n".join([doc.page_content for doc in context])
    input_text = system_prompt + "\nContext: " + context_text + "\nUser: " + query + "\nBot:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move to GPU

    # Generate a response using the model, without updating gradients (inference mode)
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Decode the generated tokens into a human-readable string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()  # Return the cleaned-up answer

# Function to handle the complete process of getting a response from the chatbot
def get_response(question):
    context = retrieve_context(question)  # Retrieve relevant documents for context
    start_time = time.time()  # Record the start time for response time calculation
    answer = generate_answer(question, context)  # Generate an answer based on the context
    response_time = time.time() - start_time  # Calculate the time taken to generate the response
    sources = [doc.metadata['source'] for doc in context]  # Extract the source information
    
    torch.cuda.empty_cache()  # Clear the GPU cache to free up memory
    # Strip off the "Bot:" prefix if present and return the final answer
    return answer.strip().split("\nBot:")[-1].strip()

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html template

# Flask route to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json  # Get JSON data from the POST request
    question = data.get('question', '')  # Extract the question from the request
    if not question:
        return jsonify({"error": "No question provided"}), 400  # Return error if no question is provided
    
    answer = get_response(question)  # Generate a response to the question
    return jsonify({
        "answer": answer  # Return the answer as a JSON response
    })

# Entry point for the Flask application
if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app in debug mode
