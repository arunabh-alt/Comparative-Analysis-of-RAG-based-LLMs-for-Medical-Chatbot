import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    logging as transformers_logging  # Avoid conflict with standard logging
)

import pandas as pd
import logging  # Import standard logging
import os
import pandas as pd
import torch
from datasets import Dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, TrainerCallback

# Configure standard logging to record training progress
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Set CUDA device for training; "0" refers to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the pre-trained model and tokenizer from Hugging Face model hub
model_name = "google/flan-t5-large"
new_model = "t5-large-finetuned"
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # Configure quantization to 8-bit for efficiency
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the model for k-bit training, which reduces the model's memory footprint
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)  # Apply LoRA configuration to the model

# Define the path to the dataset CSV file and load it into a DataFrame
csv_file_path = r"C:\Users\Computing\Downloads\MedMCQA\Dataset\train.csv"
df = pd.read_csv(csv_file_path)

# Sample the first 10 rows for demonstration purposes
df_sampled = df.iloc[:10]

# Split the data into training and evaluation sets
train_df = df_sampled.iloc[:8]  # First 80% of the sampled data for training
eval_df = df_sampled.iloc[8:]  # Remaining 20% for evaluation

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Define a function to preprocess the data for model input
def preprocess_function(examples):
    questions = examples['question']
    options_a = examples['opa']
    options_b = examples['opb']
    options_c = examples['opc']
    options_d = examples['opd']
    correct_options = examples['cop']  # Index of the correct option
    explanations = examples.get('exp', [''] * len(questions))  # Use empty string if 'exp' column does not exist

    inputs = []
    labels = []

    # Format input strings and labels for the model
    for question, opa, opb, opc, opd, correct_option, explanation in zip(questions, options_a, options_b, options_c, options_d, correct_options, explanations):
        # Format the input string with options and explanation
        options_formatted = [f"{chr(97+i)}: {opt}" for i, opt in enumerate([opa, opb, opc, opd])]
        input_str = f"question: {question} {' '.join(options_formatted)}"
        
        if explanation:  # Append explanation if it exists
            input_str += f" Explanation: {explanation}"

        inputs.append(input_str)
        
        # Process label (convert correct_option index to letter)
        label_str = chr(97 + correct_option)  # Convert index to letter ('a', 'b', 'c', 'd')
        labels.append(label_str)

    # Tokenize the inputs and labels
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels_tokenized = tokenizer(labels, max_length=35, truncation=True, padding="max_length")["input_ids"]
    
    model_inputs["labels"] = labels_tokenized
    return model_inputs

# Apply the preprocessing function to both training and evaluation datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on train dataset")
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on eval dataset")

# Define training arguments for Seq2Seq model
training_params = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save results
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=1,  # Number of steps to accumulate gradients
    optim="paged_adamw_32bit",  # Optimizer to use
    save_steps=200,  # Save model every 200 steps
    logging_steps=200,  # Log training information every 200 steps
    learning_rate=3e-4,  # Learning rate
    weight_decay=0.001,  # Weight decay for regularization
    fp16=True,  # Use 16-bit floating point precision
    push_to_hub=False,  # Disable pushing to Hugging Face model hub
)

# Define a custom callback to log training progress
class LogCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_entry = {
                'step': state.global_step,
                'loss': logs.get('loss'),
                'learning_rate': logs.get('learning_rate'),
                'epoch': logs.get('epoch'),
                'eval_loss': logs.get('eval_loss'),
                'eval_runtime': logs.get('eval_runtime'),
                'eval_samples_per_second': logs.get('eval_samples_per_second'),
                'eval_steps_per_second': logs.get('eval_steps_per_second'),
                'train_runtime': logs.get('train_runtime'),
                'train_samples_per_second': logs.get('train_samples_per_second'),
                'train_steps_per_second': logs.get('train_steps_per_second'),
                'train_loss': logs.get('train_loss'),
            }
            self.training_logs.append(log_entry)
            logger.info(log_entry)  # Log the entry using standard logging

log_callback = LogCallback()

# Data collator for Seq2Seq training
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize the Seq2SeqTrainer with the specified parameters
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    dataset_text_field="text",  # Specify the text field in the dataset
    max_seq_length=None,  # No maximum sequence length constraint
    tokenizer=tokenizer,
    args=training_params,
    packing=False,  # Disable packing sequences
    callbacks=[log_callback],  # Add the custom logging callback
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Save the training logs to an Excel file
df = pd.DataFrame(log_callback.training_logs)
df.to_excel('training_logs.xlsx', index=False)
