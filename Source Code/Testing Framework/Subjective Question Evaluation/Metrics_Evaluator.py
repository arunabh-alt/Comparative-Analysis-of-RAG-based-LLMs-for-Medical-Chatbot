import re  # Regular expression operations
import numpy as np  # Numerical operations, especially on arrays
import torch  # PyTorch for tensor operations and model handling
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Cosine similarity between vectors
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU score calculation
from nltk.tokenize import word_tokenize  # Tokenizer for BLEU and METEOR
from rouge_score import rouge_scorer  # ROUGE score calculation
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face transformers for model and tokenizer
import bert_score  # BERTScore for evaluating text similarity
import nltk  # Natural Language Toolkit
from nltk.translate import meteor_score  # METEOR score calculation

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt')

def parse_qa_pairs(file_path):
    """
    Reads the file and extracts generated answers and ground truth answers using regular expressions.
    Handles potential Unicode decoding errors by trying different encodings.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            text = file.read()
    
    # Compile regular expressions to extract generated answers and ground truths
    generated_re = re.compile(r"Generated Answer: (.*?)\n")
    ground_truth_re = re.compile(r"Ground Truth: (.*?)\n")
    
    # Find all matching answers
    generated_answers = generated_re.findall(text)
    ground_truths = ground_truth_re.findall(text)
    return generated_answers, ground_truths

def ensure_equal_length(gen_answers, ground_truths):
    """
    Truncates both lists (generated answers and ground truths) to the length of the shorter one.
    This ensures both lists are of equal length to avoid errors in metric calculations.
    """
    min_length = min(len(gen_answers), len(ground_truths))
    return gen_answers[:min_length], ground_truths[:min_length]

def calculate_bertscore(references, bot):
    """
    Calculates BERTScore for the generated answers compared to the references.
    Returns the precision, recall, and F1 score from BERTScore.
    """
    P, R, F1 = bert_score.score(bot, references, lang="en", verbose=True, rescale_with_baseline=True)
    return {
        'Precision': P.mean().item(),
        'Recall': R.mean().item(),
        'F1': F1.mean().item()
    }

def calculate_rouge(references, bot):
    """
    Calculates ROUGE-1 and ROUGE-L scores for the generated answers compared to the references.
    Uses a stemmer to ensure that different forms of the same word are treated as equivalent.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, cand) for ref, cand in zip(references, bot)]
    rouge1 = [score['rouge1'].fmeasure for score in scores]
    rougeL = [score['rougeL'].fmeasure for score in scores]
    return {
        'ROUGE-1': sum(rouge1) / len(rouge1),
        'ROUGE-L': sum(rougeL) / len(rougeL)
    }

def calculate_meteor(references, bot):
    """
    Calculates the METEOR score for the generated answers compared to the references.
    METEOR considers synonyms and stemming, making it robust to variations in wording.
    """
    scores = [meteor_score.single_meteor_score(word_tokenize(ref), word_tokenize(cand)) for ref, cand in zip(references, bot)]
    return {
        'METEOR': sum(scores) / len(scores) if scores else 0
    }

def calculate_bleu(references, bot):
    """
    Calculates the BLEU score for the generated answers compared to the references.
    Uses a smoothing function to handle cases where n-grams might not be present.
    """
    smoothing_function = SmoothingFunction().method4
    scores = [sentence_bleu([word_tokenize(ref)], word_tokenize(cand), smoothing_function=smoothing_function) for ref, cand in zip(references, bot)]
    return {
        'BLEU': sum(scores) / len(scores)
    }

def calculate_perplexity(bot):
    """
    Calculates the perplexity of the generated text using a pre-trained language model (GPT-2).
    Perplexity measures how well a probability model predicts a sample.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    
    # Tokenize the entire generated text
    encodings = tokenizer("\n\n".join(bot), return_tensors="pt")

    max_length = model.config.n_positions  # Maximum length supported by the model
    stride = 512  # Stride size for processing large texts in chunks

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # Calculate target length for each chunk
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask out non-target tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len  # Calculate log likelihood

        lls.append(log_likelihood)

    # Calculate and return perplexity
    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()
    return {
        'Perplexity': ppl
    }

def evaluate_responses(references, bot):
    """
    Evaluates the generated answers against the references using various metrics: BERTScore, ROUGE, METEOR, BLEU, and Perplexity.
    Ensures the lengths of the input lists match before evaluation.
    """
    if len(references) != len(bot):
        print(f"Warning: Number of references ({len(references)}) does not match number of generated answers ({len(bot)}).")
        references, bot = ensure_equal_length(references, bot)

    # Calculate various evaluation metrics
    bert_scores = calculate_bertscore(references, bot)
    rouge_scores = calculate_rouge(references, bot)
    meteor_scores = calculate_meteor(references, bot)
    bleu_scores = calculate_bleu(references, bot)
    perplexity_scores = calculate_perplexity(bot)

    return {
        'BERTScore': bert_scores,
        'ROUGE Score': rouge_scores,
        'METEOR Score': meteor_scores,
        'BLEU Score': bleu_scores,
        'Perplexity Score': perplexity_scores
    }

# Main execution
file_path = r'C:\Users\Computing\Downloads\output_20_questions-cl.txt'
generated_answers, ground_truths = parse_qa_pairs(file_path)

# Ensure both lists are of equal length
if len(generated_answers) != len(ground_truths):
    print(f"Warning: Number of generated answers ({len(generated_answers)}) does not match number of ground truths ({len(ground_truths)}).")
    generated_answers, ground_truths = ensure_equal_length(generated_answers, ground_truths)

# Evaluate the generated answers using various metrics
results = evaluate_responses(ground_truths, generated_answers)

# Print the results for each metric
print(f"BERTScore: {results['BERTScore']}")
print(f"ROUGE Score: {results['ROUGE Score']}")
print(f"METEOR Score: {results['METEOR Score']}")
print(f"BLEU Score: {results['BLEU Score']}")
print(f"Perplexity Score: {results['Perplexity Score']}")
