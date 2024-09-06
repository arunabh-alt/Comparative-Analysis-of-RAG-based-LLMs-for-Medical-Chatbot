import re

def process_labels(input_file, output_file):
    # Try to open the file with utf-8 encoding
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # If utf-8 fails, open the file with 'errors=replace' to handle encoding issues gracefully
        with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
    
    # Define the labels you want to handle
    labels = ["Question", "Generated Answer", "Ground Truth"]
    
    # Initialize a list to hold the content for each block
    blocks = []
    current_block = {label: "" for label in labels}
    
    # Split the text by newlines to process each line
    lines = text.split('\n')
    
    # Track the current label being processed
    current_label = None
    
    for line in lines:
        # Check if the line starts with one of the labels
        for label in labels:
            if line.startswith(f"{label}:"):
                # Save the current block if it's not empty
                if any(current_block.values()):
                    blocks.append(current_block)
                    current_block = {l: "" for l in labels}
                
                # Update the current label and content
                current_label = label
                current_block[current_label] = line[len(label) + 1:].strip()
                break
        else:
            # Continue appending content if we're already within a label
            if current_label:
                current_block[current_label] += ' ' + line.strip()
    
    # Append the last block if it contains content
    if any(current_block.values()):
        blocks.append(current_block)
    
    # Format the output content
    formatted_text = ""
    for block in blocks:
        for label in labels:
            if block[label]:
                formatted_text += f"{label}: {block[label]}\n"
        formatted_text += "\n"
    
    # Write the formatted text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(formatted_text.strip())

if __name__ == "__main__":
    input_file = r"C:\Users\Computing\Downloads\output_20_questions.txt"  #  input file path
    output_file = r"C:\Users\Computing\Downloads\output_20_questions-cl.txt"  #  desired output file path
    
    process_labels(input_file, output_file)
    print(f"Processed text has been saved to {output_file}")
