import re  # Import the regular expression module

def parse_response_file(file_path):
    # List of character encodings to try for reading the file
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

    content = None  # Initialize content as None
    for encoding in encodings:  # Attempt to read the file with different encodings
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()  # Read the file content
            break  # If successful, break out of the loop
        except UnicodeDecodeError:
            continue  # If decoding fails, try the next encoding

    if content is None:  # If the file could not be read with any encoding
        raise ValueError("Unable to decode the file with available encodings.")
    
    # Split content into entries separated by double newlines
    entries = re.split(r'\n\n+', content.strip())

    results = []  # Initialize a list to hold parsed results
    
    for entry in entries:
        # Use regular expressions to extract question, options, selected answer, and ground truth
        question_match = re.search(r'Question: (.*?)\n', entry, re.S)
        options_match = re.findall(r'Option [A-E]: (.*?) - Confidence: ([\d.-]+) - Similarity: ([\d.]+)', entry)
        selected_answer_match = re.search(r'Selected Answer: Option [A-E]: (.*?)\n', entry)
        ground_truth_match = re.search(r'Ground Truth: (.*?)\n', entry)
        
        # Extract and clean the question text
        question = question_match.group(1).strip() if question_match else None
        
        # Parse the options into a dictionary with confidence and similarity scores
        options = {
            f"Option {chr(65+i)}": (opt.strip(), float(confidence), float(similarity))
            for i, (opt, confidence, similarity) in enumerate(options_match)
        }
        
        # Extract the selected answer and ground truth, if available
        selected_answer = selected_answer_match.group(1).strip() if selected_answer_match else None
        ground_truth = ground_truth_match.group(1).strip() if ground_truth_match else None
        
        # Append to results only if all required elements are found
        if question and options and selected_answer and ground_truth:
            results.append({
                'question': question,
                'options': options,
                'selected_answer': selected_answer,
                'ground_truth': ground_truth
            })
        else:
            # Log the entry if some required fields are missing
            print("Skipping entry due to missing values:", question, options, selected_answer, ground_truth)
    
    return results  # Return the list of parsed results


def calculate_metrics(results, k=2):
    total = len(results)  # Total number of valid entries
    rank_based_accuracy = 0  # Initialize accuracy counter
    rank_based_total = 0  # Initialize total rank-based counter
    
    # Initialize lists for different metrics
    rank_list = []  
    reciprocal_ranks = []
    precision_at_k_list = []
    recall_at_k_list = []
    mean_similarity_scores = []

    for result in results:
        # Extract details from each result
        selected_answer = result['selected_answer']
        ground_truth = result['ground_truth']
        options = result['options']
        
        # Skip the entry if any option text or ground truth is None
        if ground_truth is None or any(opt_text is None for opt, (opt_text, _, _) in options.items()):
            print("Skipping entry due to None values")
            continue

        # Sort options based on confidence score in descending order
        sorted_options = sorted(options.items(), key=lambda x: x[1][1], reverse=True)
        
        # Find the rank of the ground truth option
        ground_truth_option = next((opt for opt, (text, _, _) in options.items() if text == ground_truth), None)
        if ground_truth_option:
            ground_truth_rank = next((index + 1 for index, (opt, (text, _, _)) in enumerate(sorted_options) if opt == ground_truth_option), None)
            
            if ground_truth_rank is not None:
                rank_list.append(ground_truth_rank)  # Append rank to the rank list
                
                # Calculate Reciprocal Rank (1/rank)
                reciprocal_rank = 1 / ground_truth_rank
                reciprocal_ranks.append(reciprocal_rank)
                
                # Calculate Precision@k (proportion of relevant items in top k)
                relevant_items = set(opt for opt, (text, _, _) in options.items() if text == ground_truth)
                top_k_options = [opt for opt, _ in sorted_options[:k]]
                relevant_top_k = sum(1 for opt in top_k_options if opt in relevant_items)
                precision_at_k = relevant_top_k / k
                precision_at_k_list.append(precision_at_k)
                
                # Calculate Recall@k (proportion of relevant items in top k out of all relevant items)
                total_relevant_items = len(relevant_items)
                relevant_top_k_count = sum(1 for opt in top_k_options if opt in relevant_items)
                recall_at_k = relevant_top_k_count / total_relevant_items if total_relevant_items > 0 else 0
                recall_at_k_list.append(recall_at_k)
                
                # If the rank is within the threshold (k_threshold), consider it correct
                k_threshold = 3 
                if ground_truth_rank <= k_threshold:
                    rank_based_accuracy += 1
                rank_based_total += 1

        # Calculate the mean similarity score for all options
        similarities = [similarity for _, (_, _, similarity) in options.items()]
        
        if len(similarities) > 0:
            mean_similarity = sum(similarities) / len(similarities)
            mean_similarity_scores.append(mean_similarity)
        else:
            print("Skipping mean similarity calculation due to no available similarities.")

    # Compute final metrics after iterating through all results
    rank_based_accuracy = rank_based_accuracy / rank_based_total if rank_based_total > 0 else 0
    
    # Mean Reciprocal Rank (average of reciprocal ranks)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    
    # Overall mean similarity score (mean of similarity scores across all questions)
    overall_mean_similarity = sum(mean_similarity_scores) / len(mean_similarity_scores) if mean_similarity_scores else 0

    return rank_based_accuracy, rank_list, mean_reciprocal_rank, average_precision_at_k, average_recall_at_k, overall_mean_similarity

# Path to the response.txt file
file_path = r'D:\LLM Project\Ranked_based_Accuracy\responses_and_probabilities_RAG_Llama_MedQA.txt'

# Parsing the response file to extract questions, options, and answers
results = parse_response_file(file_path)

# Calculating the required metrics based on the parsed results
rank_based_accuracy, rank_list, mean_reciprocal_rank, average_precision_at_k, average_recall_at_k, overall_mean_similarity = calculate_metrics(results, k=2)

# Print the calculated metrics
print(f"Lenient Accuracy [LaCC] (K=3): {rank_based_accuracy * 100:.2f}%")
print(f"Ranks of Ground Truths: {rank_list}")
print(f"Mean Reciprocal Rank (MRR): {mean_reciprocal_rank:.2f}")
print(f"Overall Mean Similarity Score: {overall_mean_similarity:.4f}")
