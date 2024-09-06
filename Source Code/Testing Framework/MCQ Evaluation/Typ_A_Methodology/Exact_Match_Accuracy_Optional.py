import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Function to parse QA pairs from a text file
def parse_qa_pairs(file_path):
    try:
        # Attempt to open the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try ISO-8859-1
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()
    
    # Define regular expressions to extract generated answers and ground truths from the text
    generated_re = re.compile(r"Generated Answer: (.*?)\n")
    ground_truth_re = re.compile(r"Ground Truth: (.*?)\n")
    
    # Find all generated answers and ground truths using regex
    generated_answers = generated_re.findall(text)
    ground_truths = ground_truth_re.findall(text)
    
    # Function to clean and normalize text by removing extra spaces and trimming
    def clean_text(text):
        return re.sub(r'\s+', ' ', text.strip())

    # Apply cleaning function and ensure the text is not empty
    generated_answers = [clean_text(ans.split()[-1]) for ans in generated_answers if clean_text(ans.split()[-1])]
    ground_truths = [clean_text(ans.split()[-1]) for ans in ground_truths if clean_text(ans.split()[-1])]
    
    return generated_answers, ground_truths

# Function to calculate evaluation metrics: accuracy, precision, recall, and F1-score
def calculate_metrics(generated_answers, ground_truths):
    # Function to check similarity between two texts based on cosine similarity of TF-IDF vectors
    def are_similar(text1, text2, threshold=1):
        if not text1 or not text2:  
            return False
        try:
            # Convert texts to TF-IDF vectors and compute cosine similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return cosine_sim >= threshold
        except ValueError:
            return False

    correct = 0
    total = len(ground_truths)

    # Compare generated answers with ground truths for similarity
    for gen, truth in zip(generated_answers, ground_truths):
        if are_similar(gen, truth):
            correct += 1

    accuracy = correct / total if total > 0 else 0

    # Print the number of correct answers and total questions
    print(f"Total Correct Answer: {correct}")
    print(f"Total Questions: {total}")
    
    return accuracy

# File path to the text file containing the QA pairs
file_path = r"C:\Users\Computing\Desktop\Thesis_Supporting_Document\Results\MCQ Questions\Type A Methodology\MedMCQA\responses_and_truths_Mistral_Model.txt"
# Alternative file path (commented out)
# file_path = r'C:\Users\Computing\Downloads\Two Models\Mistral Ai LLm\Mistral Ai LLm\USMLE\responses_and_truths.txt'

# Parse the QA pairs from the specified file
generated_answers, ground_truths = parse_qa_pairs(file_path)

# Check if the number of generated answers matches the number of ground truths
if len(generated_answers) != len(ground_truths):
    print("Warning: The number of generated answers does not match the number of ground truths.")

# Calculate and print evaluation metrics
accuracy = calculate_metrics(generated_answers, ground_truths)
print(f"Accuracy: {accuracy * 100:.2f}%")
