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
tokenizer = AutoTokenizer.from_pretrained("/home/computing/Downloads/Test/mistral-finetuned")
model = AutoModelForCausalLM.from_pretrained(
    "/home/computing/Downloads/Test/mistral-finetuned",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# adjusts the maximum input length for the model
max_input_length = 4096  
model = exllama_set_max_input_length(model, max_input_length=max_input_length)

# Configuration for text generation, including sampling strategies.
generation_config = GenerationConfig(
    do_sample=True,  # Enables sampling for more diverse outputs.
    top_k=1,  # Use the top-k sampling strategy with k=1 (greedy sampling).
    temperature=0.1,  # Low temperature for less random outputs.
    max_new_tokens=20,  # Limit the number of generated tokens.
    pad_token_id=tokenizer.eos_token_id  # Use the EOS token ID for padding.
)

# Initialize HuggingFace sentence embeddings model with GPU support.
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', model_kwargs={'device': 'cuda'})

# Load a local FAISS vector database from disk, allowing deserialization of potentially unsafe data.
Db_faiss_path = "/home/computing/Downloads/Test/Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

# Function to retrieve context documents based on query similarity.
def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)  # Embed the query into a vector.
    docs = db.similarity_search_by_vector(query_embedding, k)  # Retrieve the top k similar documents.
    return docs

# Function to generate an answer given a query and retrieved context.
def generate_answer(query, context):
    context_text = "\n\n".join([doc.page_content for doc in context])  # Concatenate all context documents.
    
    input_text = f"Context: {context_text}\nQuestion: {query} \nAnswer:"  # Formulate the input prompt.
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move to GPU.
    outputs = model.generate(**inputs, generation_config=generation_config)  # Generate the output using the model.
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the generated tokens to text.
    
    answer_part = answer.split("Answer:")[-1].strip()  # Extract the answer portion.
    
    # Clean up the answer text by removing any labels (like "A:", "B:", etc.).
    answer_text = re.sub(r"^[A-E]:\s*", "", answer_part).strip()
    return answer_text

# Function to generate confidence scores for each option using the LLM.
def generate_option_confidences_with_llm(query, options, context):
    context_text = "\n\n".join([doc.page_content for doc in context])  # Concatenate all context documents.
    option_confidences = {}  # Dictionary to store confidence scores for each option.
    
    for option_key, option_text in options.items():
        input_text = f"Context: {context_text}\nQuestion: {query}\nOption: {option_text}"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # Tokenize and move to GPU.

        # Generate logits for the input tokens.
        with torch.no_grad():  # Disable gradient computation for inference.
            outputs = model(**inputs)
            logits = outputs.logits[-1]  # Extract the logits from the last token.

        # Compute the probability of the given option.
        option_input_ids = tokenizer(option_text, return_tensors="pt").input_ids.to("cuda")
        option_logits = logits[:, option_input_ids].mean(dim=-1)  # Mean logits for the option tokens.
        option_probability = sigmoid(option_logits).mean().item()  # Apply sigmoid and take the mean probability.

        option_confidences[option_key] = option_probability  # Store the probability in the dictionary.

    return option_confidences

# Function to generate embeddings for each option and calculate similarities to the generated answer.
def generate_option_embeddings(generated_answer, options):
    answer_embedding = embeddings.embed_query(generated_answer)  # Embed the generated answer.
    
    option_embeddings = [embeddings.embed_query(desc) for desc in options.values()]  # Embed each option.
    similarities = calculate_similarity(answer_embedding, option_embeddings)  # Calculate similarities.
    
    return dict(zip(options.keys(), similarities)), option_embeddings  # Return a dictionary of similarities.

# Function to calculate cosine similarity between the answer and each option embedding.
def calculate_similarity(context_embedding, option_embeddings):
    similarities = []
    for opt_emb in option_embeddings:
        similarity = cosine_similarity([context_embedding], [opt_emb])[0][0]  # Calculate cosine similarity.
        similarities.append(similarity)
    return similarities

# Function to get the embedding for a specific answer.
def get_answer_embedding(answer):
    return embeddings.embed_query(answer)  # Embed the answer into a vector.

# Function to evaluate the model's performance on a set of questions.
def evaluate_model(questions):
    responses_and_truths = []  # List to store response details.
    total_time = 0  # Track the total time taken for evaluation.

    for qa in questions:
        question = qa['question']
        options = qa['options']
        ground_truth = qa['answer']
        
        start_time = time.time()  # Start the timer.
        context_docs = retrieve_context(question, k=10)  # Retrieve relevant context documents.
        
        # Generate confidence scores for each option using the LLM.
        option_confidences = generate_option_confidences_with_llm(question, options, context_docs)
        
        # Generate the answer using the LLM.
        generated_answer = generate_answer(question, context_docs)
        
        # Generate option embeddings and calculate similarities.
        option_similarities, _ = generate_option_embeddings(generated_answer, options)
        
        # Sort options by their confidence scores.
        sorted_options = sorted(option_confidences.items(), key=lambda x: x[1], reverse=True)
        
        # Determine the most confident option.
        most_confident_option = sorted_options[0][0]
        most_confident_option_text = options[most_confident_option]
        formatted_generated_answer = f"Option {most_confident_option}: {most_confident_option_text}"
        
        end_time = time.time()  # Stop the timer.
        total_time += (end_time - start_time)  # Accumulate the time taken.
        
        # Prepare the full response text with confidence and similarity information.
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
        
        responses_and_truths.append(full_response)  # Store the response details.
    
    # Save the detailed responses and probabilities to a text file.
    with open(r'responses_and_probabilities_RAG_Mistral_Confidence.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(responses_and_truths))
    
    # Save the average time per question metric.
    with open(r'performance_metrics_RAG_Mistral.txt', 'w', encoding='utf-8') as file:
        avg_time_per_question = total_time / len(questions) if len(questions) > 0 else 0
        file.write(f"Average Time per Question: {avg_time_per_question:.4f} seconds\n")

# Function to load a list of questions and answers from a JSONL file.
def load_questions_answers(file_path, limit=100):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:  # Stop if the limit is reached.
                break
            data = json.loads(line.strip())  # Load each line as a JSON object.
            questions.append(data)  # Append the loaded question to the list.
    return questions

# Main function to load the questions and evaluate the model.
def main():
    questions_file_path = '/home/computing/Downloads/Test/test.jsonl'
    questions = load_questions_answers(questions_file_path, limit=100)  # Load up to 100 questions.
    
    evaluate_model(questions)  # Evaluate the model without returning accuracy.

# Run the main function when this script is executed.
if __name__ == "__main__":
    main()
