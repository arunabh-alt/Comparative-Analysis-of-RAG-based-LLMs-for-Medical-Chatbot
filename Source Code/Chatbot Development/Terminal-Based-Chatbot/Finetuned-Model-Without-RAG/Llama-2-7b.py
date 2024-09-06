from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
import torch
import time

# Define the system prompt to guide the chatbot's behavior and response style
system_prompt = """
You are a helpful and informative assistant. Your goal is to answer questions accurately, thoroughly, and naturally. Provide detailed explanations and context when possible. If you do not understand or do not have enough information to answer, simply say - "Sorry, I don't know." Avoid formatting your response as a multiple-choice answer.
"""

# Load the tokenizer and model for Llama-2
model_path = r"C:\Users\Computing\Downloads\Two Models\llm-llama\llm-llama\llama-2-7b-arunabh"

# Initialize the tokenizer from the pre-trained model path, enabling remote code execution
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Initialize the model from the pre-trained model path with specific configurations
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 precision to reduce memory usage
    device_map="cuda"           # Map the model to GPU (CUDA) for faster inference
)

# Configure the generation settings for the model
generation_config = GenerationConfig(
    do_sample=True,              # Enable sampling to generate diverse outputs
    top_k=1,                     # Restrict sampling to the top-k tokens (k=1 for deterministic output)
    temperature=0.1,             # Control randomness in output; lower values make output more deterministic
    max_new_tokens=150,          # Set the maximum number of new tokens to generate
    pad_token_id=tokenizer.eos_token_id  # Set the pad token ID to handle padding in generation
)

print("Chatbot: Hi! I'm here to assist you. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")  # Read user input from the console
    
    # Check if the user wants to exit the chat
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")  # Print exit message
        break
    
    # Combine the system prompt with user input to create the full prompt for the model
    full_prompt = system_prompt + "\nUser: " + user_input + "\nAssistant:"
    
    # Tokenize the combined prompt and move tensors to GPU
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Measure the time taken to generate a response
    start_time = time.time()
    outputs = model.generate(**inputs, generation_config=generation_config)  # Generate the response
    end_time = time.time()
    
    response_time = end_time - start_time  # Calculate the time taken to generate the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the model's output to text
    
    # Extract the assistant's response from the generated text
    assistant_response_start = response.find("Assistant:") + len("Assistant:")
    assistant_response = response[assistant_response_start:].strip()
    
    print("Chatbot:", assistant_response)  # Print the assistant's response
    print(f"Response Time: {response_time:.2f} seconds")  # Print the time taken to generate the response
