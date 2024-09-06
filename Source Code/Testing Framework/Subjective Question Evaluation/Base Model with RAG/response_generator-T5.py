import json
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import re
import torch
import re
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, BitsAndBytesConfig, AutoModelForSeq2SeqLM
import os
import time
import json
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
# System prompt for the model
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. 
"""

# Load the pre-trained model and tokenizer
model_name = "google/flan-t5-large"
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)



# Configure generation settings
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.2,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)


# Load embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\Base_Model_With_RAG_Medical\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)
# Configure generation settings
generation_config = GenerationConfig(
    do_sample=True,
    top_k=10,
    temperature=0.1,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)

# Retrieve relevant context documents using FAISS
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    ranked_docs = rank_documents(query, docs)
    return ranked_docs[:5]

# Rank documents (currently just returning the same docs)
def rank_documents(query, docs):
    return docs 

def generate_answer(query, context, few_shot_examples):
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    few_shot_text = "\n\n".join([
        f"User: {ex['Question']}\nBot: {ex['Answer']}"
        for ex in few_shot_examples
    ])
    
    input_text = system_prompt + "\n" + few_shot_text + "\nUser: " + query + "\nBot:"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs =  outputs = model.generate(**inputs,generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part after "Bot:"
    answer_part = answer.split("Bot:")[-1].strip()
    
    # Clean and format the answer to only include the response text
    answer_text = re.sub(r"^[A-E]:\s*", "", answer_part).strip()
    return answer_text

# Load questions and answers from a CSV file
def load_questions_answers(file_path, limit=20):
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

# Evaluate the model's performance and save results to text files
def evaluate_model(questions, generate_answer, few_shot_examples):
    total = len(questions)
    total_time = 0
    
    responses_and_truths = []
    
    for qa in questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        start_time = time.time()
        retrieved_docs = retrieve_context(question, k=10)
        context = retrieved_docs[:5]
        generated_answer = generate_answer(question, context, few_shot_examples)
        end_time = time.time()
        
        # Append results to responses_and_truths without including context
        responses_and_truths.append(f"Question: {question}\nGenerated Answer: {generated_answer}\nGround Truth: {ground_truth}\n")
        
        total_time += (end_time - start_time)
    
    avg_time_per_question = total_time / total if total > 0 else 0
    
    # Save responses and ground truths to a text file
    with open(r'C:\Users\Computing\Downloads\Base_Model_With_RAG_Medical\Responses_Time\responses_and_truths_T5_Base.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save average time to a text file
    with open(r'C:\Users\Computing\Downloads\Base_Model_With_RAG_Medical\Responses_Time\performance_metrics_T5_Base.txt', 'w', encoding='utf-8') as file:
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")
    
    return avg_time_per_question

# Main function to get response and evaluate the model
def main():
    questions_file_path = r'C:\Users\Computing\Downloads\Base_Model_With_RAG_Medical\train.csv'
    questions = load_questions_answers(questions_file_path, limit=20)
    
    # Define few-shot examples
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
                        More on: Handwashing
                        More on: Food and Water Safety"""
        }
    ]
    
    evaluate_model(questions, generate_answer, few_shot_examples)

if __name__ == "__main__":
    main()



