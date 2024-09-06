import json
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from nltk.tokenize import word_tokenize

# System prompt that defines the assistant's behavior and response style
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know.".
"""

# Load the pre-trained sequence-to-sequence model and tokenizer from specified paths
model_path = r"C:\Users\Computing\Downloads\Flan-T5-Model\Flan-T5-Model\T5-Finetune-Arunabh"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Initialize embeddings and load FAISS vector store for document retrieval
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)
# Configure generation settings
generation_config = GenerationConfig(
    do_sample=True,
    top_k=10,
    temperature=0.1,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)
# Function to retrieve relevant context documents using FAISS
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)  # Create an embedding for the query
    docs = db.similarity_search_by_vector(query_embedding, k)  # Search for similar documents
    ranked_docs = rank_documents(query, docs)  # Rank documents (currently a no-op)
    return ranked_docs[:5]  # Return top 5 documents

# Function to rank documents (currently does not alter the order because of FAISS)
def rank_documents(query, docs):
    return docs  # Return documents as-is

# Function to generate an answer using the pre-trained model
def generate_answer(question):
    # Retrieve context documents related to the question
    context = retrieve_context(question, k=10)
    # Combine context documents into a single string
    context_str = " ".join([doc.page_content for doc in context])  # Use 'page_content' or similar attribute
    
    # Construct input text by including system prompt, question, and context
    input_text = f"{system_prompt}\nUser: {question}\nContext: {context_str}\nBot: Please provide a detailed response."
    
    # Tokenize the input text and generate a response
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs,generation_config=generation_config)
         
    
    # Decode the generated text and strip any leading/trailing whitespace
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# Function to load questions and answers from a JSONL file
def load_questions_answers(file_path, limit=5):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:  # Limit the number of loaded questions
                break
            data = json.loads(line.strip())  # Parse JSON line
            questions.append(data)
    return questions

# Function to evaluate the model's performance and save results to text files
def evaluate_model(questions, generate_answer):
    correct = 0
    total = len(questions)
    total_time = 0
    
    responses_and_truths = []
    
    for qa in questions:
        question = qa['question']
        options = qa['options']
        ground_truth = qa['answer']
        
        start_time = time.time()  # Start timing the response generation
        retrieved_docs = retrieve_context(question, k=10)  # Retrieve context
        context = retrieved_docs[:5]
        generated_answer = generate_answer(question)  # Generate answer
        end_time = time.time()  # End timing
        
        # Append results including question, options, generated answer, and ground truth
        responses_and_truths.append(f"Question: {question}\nOptions: {options}\nGenerated Answer: {generated_answer}\nGround Truth: {ground_truth}\n")
        
        # Check if the generated answer matches the ground truth (case-insensitive)
        if generated_answer.strip().lower() == ground_truth.strip().lower():
            correct += 1
        
        total_time += (end_time - start_time)
    
    accuracy = correct / total  # Calculate accuracy
    avg_time_per_question = total_time / total  # Calculate average response time
    
    # Save responses and ground truths to a text file
    with open(r'C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\USMLE Test\responses_and_truths_100_few-shot.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save accuracy and average time per question to a text file
    with open(r'C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\USMLE Test\performance_metrics100_few-shot.txt', 'w', encoding='utf-8') as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")
    
    return {
        "accuracy": accuracy,
        "avg_time_per_question": avg_time_per_question
    }

# Main function to execute the model evaluation
def main():
    questions_file_path = r'C:\Users\Computing\Downloads\archive\MedQA-USMLE\questions\US\test.jsonl'
    questions = load_questions_answers(questions_file_path, limit=5)  # Load a limited number of questions
    metrics = evaluate_model(questions, generate_answer)  # Evaluate the model
    print(f"Metrics: {metrics}")  # Print performance metrics

# Entry point of the script
if __name__ == "__main__":
    main()
