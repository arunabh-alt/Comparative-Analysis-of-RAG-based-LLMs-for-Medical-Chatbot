import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers import AutoTokenizer, GenerationConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import time
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

base_model = "meta-llama/Llama-2-7b-chat-hf"
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load and quantize model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)

model = prepare_model_for_kbit_training(model)
model = prepare_model_for_kbit_training(model)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_params)

model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Configure generation settings
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)

# System prompt
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large', 
                                   model_kwargs={'device': 'cuda'})
Db_faiss_path = r"C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\Vector_Data_Base_GTR_T5_Large"
db = FAISS.load_local(Db_faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrieve_context(query, k=10):
    query_embedding = embeddings.embed_query(query)
    docs = db.similarity_search_by_vector(query_embedding, k)
    ranked_docs = rank_documents(query, docs)
    return ranked_docs[:5]

def rank_documents(query, docs):
    return docs 

def generate_answer(query, context):
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    # Create a clear separation in the prompt
    input_text = (
        system_prompt +
        "\nContext: " + context_text + 
        "\n\nUser: " + query + 
        "\nBot:"
    )
    
    # Tokenize and generate response
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    
    # Decode and clean the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ensure the output is only the answer
    return answer.strip().split("\nBot:")[-1].strip()

def get_response(question):
    context = retrieve_context(question)
    start_time = time.time()
    answer = generate_answer(question, context)
    response_time = time.time() - start_time
    sources = [doc.metadata['source'] for doc in context]
    return answer, sources, response_time

def main():
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response, sources, response_time = get_response(user_input)
        print("Bot:", response)
        # print("Sources:", sources)
        print(f"Time Taken to Respond: {response_time:.2f} seconds")

if __name__ == "__main__":
    main()
