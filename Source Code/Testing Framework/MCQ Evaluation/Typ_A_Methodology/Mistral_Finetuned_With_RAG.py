import re
import time
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, GenerationConfig
from peft import AutoPeftModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# System prompt for the model
system_prompt = """
You are a helpful and informative assistant. Provide only the response text, without the option letter.
"""

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned")
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Configure generation settings for the model
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=20,
    pad_token_id=tokenizer.eos_token_id
)

# Load embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

# Function to retrieve relevant context documents using FAISS
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    ranked_docs = rank_documents(query, docs)
    return ranked_docs[:5]

# Function to rank documents (currently just returns the same documents)
def rank_documents(query, docs):
    return docs 

# Function to generate an answer using the model
def generate_answer(query, options, context, few_shot_examples):
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    few_shot_text = "\n\n".join([
        f"User: {ex['question']}\nOptions: {ex['options']}\nBot: {ex['answer']}"
        for ex in few_shot_examples
    ])
    
    input_text = system_prompt + "\n" + few_shot_text + "\nUser: " + query + "\nOptions: " + str(options) + "\nBot:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and clean the answer part after "Bot:"
    answer_part = answer.split("Bot:")[-1].strip()
    answer_text = re.sub(r"^[A-E]:\s*", "", answer_part).strip()
    return answer_text

# Load questions and answers from a JSONL file
def load_questions_answers(file_path, limit=100):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line.strip())
            questions.append(data)
    return questions

# Parse QA pairs from a file for evaluation
def parse_qa_pairs(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()
    
    # Regular expressions to extract generated answers and ground truths
    generated_re = re.compile(r"Generated Answer: (.*?)\n")
    ground_truth_re = re.compile(r"Ground Truth: (.*?)\n")
    
    generated_answers = generated_re.findall(text)
    ground_truths = ground_truth_re.findall(text)
    
    # Function to clean and normalize text
    def clean_text(text):
        return re.sub(r'\s+', ' ', text.strip())

    generated_answers = [clean_text(ans) for ans in generated_answers]
    ground_truths = [clean_text(ans) for ans in ground_truths]
    
    return generated_answers, ground_truths

# Calculate accuracy of generated answers
def calculate_accuracy(generated_answers, ground_truths):
    def are_similar(text1, text2, threshold=1):
        if not text1 or not text2:  
            return False
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return cosine_sim >= threshold
        except ValueError:
            return False

    correct = 0
    total = len(ground_truths)

    for gen, truth in zip(generated_answers, ground_truths):
        if are_similar(gen, truth):
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Evaluate the model's performance and save results to text files
def evaluate_model(questions, generate_answer, few_shot_examples):
    correct = 0
    total = len(questions)
    total_time = 0
    
    responses_and_truths = []
    
    for qa in questions:
        question = qa['question']
        options = qa['options']
        ground_truth = qa['answer']
        
        start_time = time.time()
        retrieved_docs = retrieve_context(question, k=10)
        context = retrieved_docs[:5]
        generated_answer = generate_answer(question, options, context, few_shot_examples)
        end_time = time.time()
        
        # Append results to responses_and_truths without including context
        responses_and_truths.append(f"Question: {question}\nOptions: {options}\nGenerated Answer: {generated_answer}\nGround Truth: {ground_truth}\n")
        
        if generated_answer.strip().lower() == ground_truth.strip().lower():
            correct += 1
        
        total_time += (end_time - start_time)
    
    accuracy = correct / total if total > 0 else 0
    avg_time_per_question = total_time / total
    
    # Save responses and ground truths to a text file
    with open(r'responses_and_truths.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save accuracy to a text file
    with open(r'performance_metrics.txt', 'w', encoding='utf-8') as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")
    
    return accuracy

# Main function to get response and evaluate the model
def main():
    questions_file_path = r'C:\Users\Computing\Downloads\archive\MedQA-USMLE\questions\US\test.jsonl'
    questions = load_questions_answers(questions_file_path, limit=100)
    
    # Define few-shot examples for the model
    few_shot_examples = [
        {
            "question": "Question: A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
            "options": "{'A': 'Inhibition of thymidine synthesis', 'B': 'Inhibition of proteasome', 'C': 'Hyperstabilization of microtubules', 'D': 'Generation of free radicals', 'E': 'Cross-linking of DNA'}",
            "answer": "Cross-linking of DNA"
        },
        {
            "question": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
            "options": "{'A': 'Disclose the error to the patient but leave it out of the operative report', 'B': 'Disclose the error to the patient and put it in the operative report', 'C': 'Tell the attending that he cannot fail to disclose this mistake', 'D': 'Report the physician to the ethics committee', 'E': 'Refuse to dictate the operative report'}",
            "answer": "Tell the attending that he cannot fail to disclose this mistake"
        }
    ]
    
    accuracy = evaluate_model(questions, generate_answer, few_shot_examples)
    
    # Calculate and print accuracy from the responses and truths file
    file_path = r'responses_and_truths.txt'
    generated_answers, ground_truths = parse_qa_pairs(file_path)
    if len(generated_answers) != len(ground_truths):
        print("Warning: The number of generated answers does not match the number of ground truths.")
    
    accuracy = calculate_accuracy(generated_answers, ground_truths)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
