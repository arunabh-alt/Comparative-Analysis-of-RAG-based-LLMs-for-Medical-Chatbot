import re
import time
import csv
import torch
from transformers import AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Define the system prompt that guides the model's responses
system_prompt = """
You are a helpful and informative assistant. Provide only the response text, without the option letter.
"""

# Load the pre-trained tokenizer for encoding text inputs
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\llama-2-7b-arunabh")

# Load the pre-trained model with configurations for memory efficiency and GPU utilization
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\llama-2-7b-arunabh",
    low_cpu_mem_usage=True,      # Optimize model loading to use less CPU memory
    return_dict=True,            # Return the model outputs as a dictionary
    torch_dtype=torch.float16,   # Use float16 precision to reduce memory usage
    device_map="cuda"            # Load the model onto the GPU for faster computation
)

# Configure the generation settings for the model
generation_config = GenerationConfig(
    do_sample=True,              # Enable sampling for response variability
    top_k=1,                     # Limit sampling to the top-1 most probable token (deterministic output)
    temperature=0.1,             # Adjust the randomness of output (low temperature for more deterministic responses)
    max_new_tokens=150,          # Set the maximum number of tokens to generate
    pad_token_id=tokenizer.eos_token_id  # Define the padding token ID
)

# Initialize embeddings and load the FAISS vector store for context retrieval
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

# Retrieve relevant context documents from FAISS based on the query
def retrieve_context(query, k=10):
    """
    Retrieves relevant documents from the FAISS vector store based on the query.

    Args:
        query (str): The query for which relevant documents are retrieved.
        k (int): The number of closest documents to retrieve.

    Returns:
        List[Document]: The top k relevant documents.
    """
    query_embedding = embeddings.embed_query(query)  # Embed the query into a vector
    docs = db.similarity_search_by_vector(query_embedding, k)  # Search for similar documents
    ranked_docs = rank_documents(query, docs)  # Rank the documents (no ranking applied here)
    return ranked_docs[:5]  # Return the top 5 documents

# Rank documents (currently does not change document order)
def rank_documents(query, docs):
    return docs  # No ranking logic applied; return the documents in their retrieved order

# Generate an answer based on the query, context, and few-shot examples
def generate_answer(query, context, few_shot_examples):
    """
    Generates an answer using the context and few-shot examples provided.

    Args:
        query (str): The user's query.
        context (List[Document]): The context documents for generating the answer.
        few_shot_examples (List[Dict]): Few-shot examples to guide the model's response.

    Returns:
        str: The generated answer.
    """
    # Combine context documents into a single text
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    # Format the few-shot examples into a text string
    few_shot_text = "\n\n".join([
        f"User: {ex['Question']}\nBot: {ex['Answer']}"
        for ex in few_shot_examples
    ])
    
    # Formulate the input text for the model, including system prompt, few-shot examples, and query
    input_text = system_prompt + "\n" + few_shot_text + "\nUser: " + query + "\nBot:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move to GPU
    outputs = model.generate(**inputs, generation_config=generation_config)  # Generate the model's response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the generated tokens to text
    
    # Extract and clean the answer part after "Bot:"
    answer_part = answer.split("Bot:")[-1].strip()
    answer_text = re.sub(r"^[A-E]:\s*", "", answer_part).strip()  # Remove any extraneous prefix
    return answer_text

# Load questions and answers from a CSV file
def load_questions_answers(file_path, limit=20):
    """
    Load a specified number of questions and answers from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing questions and answers.
        limit (int): The maximum number of questions to load.

    Returns:
        List[Dict]: A list of dictionaries containing questions and answers.
    """
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            questions.append({
                'question': row['Question'],
                'answer': row['Answer']
            })
    return questions

# Evaluate the model's performance and save results
def evaluate_model(questions, generate_answer, few_shot_examples):
    """
    Evaluate the model's performance on a set of questions and save results to files.

    Args:
        questions (List[Dict]): A list of dictionaries containing questions and answers.
        generate_answer (Callable): The function used to generate answers.
        few_shot_examples (List[Dict]): Few-shot examples to guide the model's responses.

    Returns:
        float: The average time taken to generate responses.
    """
    total = len(questions)
    total_time = 0
    responses_and_truths = []
    
    for qa in questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        start_time = time.time()  # Start timing
        retrieved_docs = retrieve_context(question, k=10)  # Retrieve relevant documents
        context = retrieved_docs[:5]  # Use top 5 documents for context
        generated_answer = generate_answer(question, context, few_shot_examples)  # Generate the answer
        end_time = time.time()  # End timing
        
        # Collect results
        responses_and_truths.append(f"Question: {question}\nGenerated Answer: {generated_answer}\nGround Truth: {ground_truth}\n")
        total_time += (end_time - start_time)
    
    avg_time_per_question = total_time / total if total > 0 else 0  # Calculate average time
    
    # Save the responses and ground truths to a file
    with open(r'C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\Medical Questions\responses_and_truths.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save the average response time to a file
    with open(r'C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\Medical Questions\performance_metrics.txt', 'w', encoding='utf-8') as file:
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")
    
    return avg_time_per_question

# Main function to execute the evaluation process
def main():
    """
    Main function to load data, evaluate the model, and handle file paths.
    """
    questions_file_path = r'C:\Users\Computing\Downloads\llm-llama\llm-llama\llm-llama\Medical Questions\Medical_Data.csv'
    questions = load_questions_answers(questions_file_path, limit=20)
    
    # Define few-shot examples for guiding the model's responses
    few_shot_examples = [
        {
            "Question": "Who is at risk for Lymphocytic Choriomeningitis (LCM)?",
            "Answer": "LCMV infections can occur after exposure to fresh urine, droppings, saliva, or nesting materials from infected rodents.  Transmission may also occur when these materials are directly introduced into broken skin, the nose, the eyes, or the mouth, or presumably, via the bite of an infected rodent. Person-to-person transmission has not been reported, with the exception of vertical transmission from infected mother to fetus, and rarely, through organ transplantation."
        },
        {
            "Question": "How to prevent Parasites - Cysticercosis ?",
            "Answer": """To prevent cysticercosis, the following precautions should be taken:
    
     - Wash your hands with soap and warm water after using the toilet, changing diapers, and before handling food
     - Teach children the importance of washing hands to prevent infection
     - Wash and peel all raw vegetables and fruits before eating
     - Use good food and water safety practices while traveling in developing countries such as: 
      
       - Drink only bottled or boiled (1 minute) water or carbonated (bubbly) drinks in cans or bottles
       - Filter unsafe water through an ""absolute 1 micron or less"" filter AND dissolve iodine tablets in the filtered water; ""absolute 1 micron"" filters can be found in camping and outdoor supply stores
      """
        }
    ]
    
    evaluate_model(questions, generate_answer, few_shot_examples)  # Evaluate the model's performance

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly
