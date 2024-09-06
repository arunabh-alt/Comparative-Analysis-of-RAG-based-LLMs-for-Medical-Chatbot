import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model and tokenizer
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

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})

Db_faiss_path = r"C:\Users\Computing\Downloads\LLM-T5-Model (2)\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_context(query, k=5):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    return docs

def generate_answer(query, context):
    # Prepare the input for the model
    input_text = f"question: {query} context: {' '.join([doc.page_content for doc in context])}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Generate the answer using the fine-tuned model
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

def get_response(question):
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    return answer

def main():
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
