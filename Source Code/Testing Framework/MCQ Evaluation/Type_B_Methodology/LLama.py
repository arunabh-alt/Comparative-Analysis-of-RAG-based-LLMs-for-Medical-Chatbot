import re
import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig,AutoModelForCausalLM
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from auto_gptq import exllama_set_max_input_length
from torch.nn.functional import sigmoid
# Load the pre-trained tokenizer, model, and embeddings
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\New folder\llama-2-7b-arunabh")
model = AutoModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\New folder\llama-2-7b-arunabh",
    torch_dtype=torch.float16,
    device_map="cuda"
)
# max_input_length = 4096  # Adjust this to be greater than the maximum sequence length you expect
# model = exllama_set_max_input_length(model, max_input_length=max_input_length)
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=20,
    pad_token_id=tokenizer.eos_token_id
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\New folder\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    return docs

def generate_answer(query,context):
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    input_text = f"Context: {context_text}\nQuestion: {query} \nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer_part = answer.split("Answer:")[-1].strip()
    
    answer_text = re.sub(r"^[A-E]:\s*", "", answer_part).strip()
    return answer_text

def generate_option_confidences_with_llm(query, options, context):
    context_text = "\n\n".join([doc.page_content for doc in context])
    option_confidences = {}
    
    for option_key, option_text in options.items():
        input_text = f"Context: {context_text}\nQuestion: {query}\nOption: {option_text}"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[-1]  # Extract the logits from the last token

        # Compute the probability of the given option
        option_input_ids = tokenizer(option_text, return_tensors="pt").input_ids.to("cuda")
        option_logits = logits[:, option_input_ids].mean(dim=-1)  # Mean logits for the option tokens
        option_probability = sigmoid(option_logits).mean().item()  # Apply sigmoid and take the mean probability

        option_confidences[option_key] = option_probability

    return option_confidences

def generate_option_embeddings(generated_answer, options):
    # context_docs = retrieve_context(query, k=10)
    # context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    answer_embedding = embeddings.embed_query(generated_answer)
    
    option_embeddings = [embeddings.embed_query(desc) for desc in options.values()]
    similarities = calculate_similarity(answer_embedding, option_embeddings)
    
    return dict(zip(options.keys(), similarities)), option_embeddings

def calculate_similarity(context_embedding, option_embeddings):
    similarities = []
    for opt_emb in option_embeddings:
        similarity = cosine_similarity([context_embedding], [opt_emb])[0][0]
        similarities.append(similarity)
    return similarities

def get_answer_embedding(answer):
    return embeddings.embed_query(answer)

def evaluate_model(questions):
    correct = 0
    responses_and_truths = []
    total_time = 0
    total_questions = len(questions)

    for qa in questions:
        question = qa['question']
        options = qa['options']
        ground_truth = qa['answer']
        
        start_time = time.time()
        context_docs = retrieve_context(question, k=10)
        
        # Generate confidences for each option using the LLM
        option_confidences = generate_option_confidences_with_llm(question, options, context_docs)
        
        # Generate the answer using the LLM
        generated_answer = generate_answer(question, context_docs)
        
        # Generate option embeddings and calculate similarities
        option_similarities, _ = generate_option_embeddings(generated_answer, options)
        
        # Sort options by their confidences
        sorted_options = sorted(option_confidences.items(), key=lambda x: x[1], reverse=True)
        
        # Determine if the selected option matches the ground truth
        most_confident_option = sorted_options[0][0]
        most_confident_option_text = options[most_confident_option]
        formatted_generated_answer = f"Option {most_confident_option}: {most_confident_option_text}"
        
        if most_confident_option.strip().lower() == ground_truth.strip().lower():
            correct += 1
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Prepare the full response text with confidence and similarity
        options_text = "\n".join([
            f"Option {opt}: {desc} - Confidence: {option_confidences[opt]:.4f} - Similarity: {option_similarities[opt]:.4f}" 
            for opt, desc in options.items()
        ])
        
        full_response = (
            f"Question: {question}\n"
            f"Options and Confidences:\n{options_text}\n"
            f"Generated Answer: {generated_answer}\n"
            f"Selected Answer: {formatted_generated_answer}\n"
            f"Ground Truth: {ground_truth}\n"
            f"---\n"
        )
        
        responses_and_truths.append(full_response)
    
    # Save responses and probabilities to a text file
    with open(r'responses_and_probabilities_RAG_Llama_Confidence.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save performance metrics
    with open(r'performance_metrics_RAG_Llama.txt', 'w', encoding='utf-8') as file:
        accuracy = correct / total_questions if total_questions > 0 else 0
        avg_time_per_question = total_time / total_questions if total_questions > 0 else 0
        
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")

    return accuracy

def load_questions_answers(file_path, limit=100):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break;
            data = json.loads(line.strip())
            questions.append(data)
    return questions

def main():
    questions_file_path = r'C:\Users\Computing\Downloads\New folder\test.jsonl'
    questions = load_questions_answers(questions_file_path, limit=100)
    
    accuracy = evaluate_model(questions)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
