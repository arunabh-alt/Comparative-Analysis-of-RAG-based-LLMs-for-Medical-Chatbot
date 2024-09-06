from transformers import T5ForConditionalGeneration, T5Tokenizer, GenerationConfig

# Path to the fine-tuned T5 model on the local system
model = r'C:\Users\Computing\Downloads\Two Models\Flan-T5-Model\Flan-T5-Model\T5-Finetune-Arunabh'

# Load the fine-tuned T5 model for conditional generation of text
finetuned_model = T5ForConditionalGeneration.from_pretrained(model)

# Load the corresponding tokenizer for the fine-tuned T5 model
finetuned_tokenizer = T5Tokenizer.from_pretrained(model)

# Configuration for text generation to control the sampling strategy and output format
generation_config = GenerationConfig(
    do_sample=True,             # Enable sampling instead of greedy decoding
    top_k=1,                    # Use only the top 1 token (deterministic output)
    temperature=0.1,            # Low temperature to make the output more deterministic and less random
    max_new_tokens=150,         # Limit the maximum number of tokens in the generated response
    pad_token_id=finetuned_tokenizer.eos_token_id  # Set padding token ID to the model's EOS token ID
)

# Function to generate a response based on the user's question
def get_response(question):
    # Prepare the input text by prefixing it with a specific instruction
    input_text = "Please answer to this question correctly: " + question
    # Tokenize the input text to prepare it for the model
    inputs = finetuned_tokenizer(input_text, return_tensors="pt")
    # Generate the model's output using the specified generation configuration
    outputs = finetuned_model.generate(**inputs, generation_config=generation_config)
    # Decode the generated output into a human-readable answer
    answer = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer  # Return the final answer

# Main function to interact with the user in a command-line interface
def main():
    print("Welcome to the chatbot. Type your question and press enter.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")  # Capture user input
        if user_input.lower() == 'exit':  # Check if the user wants to exit
            print("Goodbye!")  # Print a goodbye message
            break  # Exit the loop and end the program
        response = get_response(user_input)  # Get the response from the model
        print("Bot:", response)  # Print the chatbot's response

# Entry point for script execution
if __name__ == "__main__":
    main()  # Start the chatbot interaction loop
