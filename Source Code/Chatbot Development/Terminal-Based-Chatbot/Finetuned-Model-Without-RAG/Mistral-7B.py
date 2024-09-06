from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig
import torch
import time

# Define the system prompt that sets the behavior and expectations for the assistant
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the pre-trained tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned")

# Load the pre-trained model for causal language modeling
model = AutoPeftModelForCausalLM.from_pretrained(
    r"C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\Mistral-finetuned",
    low_cpu_mem_usage=True,         # Reduce CPU memory usage during model loading
    return_dict=True,               # Return the output as a dictionary
    torch_dtype=torch.float16,     # Use float16 precision to reduce memory usage
    device_map="cuda"               # Map the model to GPU (CUDA) for faster inference
)

# Configure generation settings for the model
generation_config = GenerationConfig(
    do_sample=True,                # Enable sampling to generate diverse outputs
    top_k=1,                       # Restrict sampling to the top-k tokens (k=1 for deterministic outputs)
    temperature=0.1,               # Control randomness; lower value makes output more deterministic
    max_new_tokens=150,            # Limit the number of new tokens to generate
    pad_token_id=tokenizer.eos_token_id  # Pad token ID for handling padding
)

print("Chatbot: Hi! I'm here to assist you. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")  # Get user input
    
    if user_input.lower() == "exit":  # Check if the user wants to exit
        print("Chatbot: Goodbye!")
        break
    
    # Combine system prompt with user input to create the full input text for the model
    full_prompt = system_prompt + "\nUser: " + user_input + "\nAssistant:"
    
    # Tokenize the input text and move tensors to GPU
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Measure the time taken to generate a response
    start_time = time.time()
    outputs = model.generate(**inputs, generation_config=generation_config)  # Generate a response using the model
    end_time = time.time()
    
    response_time = end_time - start_time  # Calculate response time
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the generated response
    
    # Extract the assistant's response from the generated text
    assistant_response_start = response.find("Assistant:") + len("Assistant:")
    assistant_response = response[assistant_response_start:].strip()
    
    print("Chatbot:", assistant_response)  # Print the assistant's response
    print(f"Response Time: {response_time:.2f} seconds")  # Print the time taken to generate the response
