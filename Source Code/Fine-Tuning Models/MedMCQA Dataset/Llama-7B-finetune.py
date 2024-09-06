import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,  # Model class for causal language modeling
    AutoTokenizer,  # Tokenizer class to handle tokenization
    BitsAndBytesConfig,  # Configuration for quantization and memory optimizations
    TrainingArguments,  # Training arguments for model training
    TrainerCallback,  # Base class for implementing custom callbacks during training
    logging as transformers_logging  # Use a different name to avoid conflict with standard logging
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model  # PEFT (Parameter Efficient Fine-Tuning) tools
from trl import SFTTrainer, SFTConfig  # Trainer class for fine-tuning with specific configurations
import pandas as pd
import logging  # Standard logging library

# Configure standard logging to record training progress in 'training_logs.log'
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Define model names and configuration
base_model = "meta-llama/Llama-2-7b-chat-hf"  # Pre-trained base model for fine-tuning
# base_model = "meta-llama/Llama-2-7b-hf"  # Alternative pre-trained base model
new_model = "llama-2-7b-arunabh"  # Directory name for saving the fine-tuned model
compute_dtype = getattr(torch, "float16")  # Set computation dtype to float16

# Configure quantization for efficient model storage and computation
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Data type for computation
    bnb_4bit_use_double_quant=False,  # Disable double quantization
)

# Load and quantize the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,  # Apply quantization configuration
    device_map={"": 0}  # Map model to GPU device (device 0)
)

# Prepare the model for k-bit training (quantized training)
model = prepare_model_for_kbit_training(model)

# Define LoRA (Low-Rank Adaptation) parameters for fine-tuning
peft_params = LoraConfig(
    lora_alpha=16,  # Scaling factor for LoRA
    lora_dropout=0.1,  # Dropout rate for LoRA
    r=64,  # Rank for LoRA
    bias="none",  # No bias term
    task_type="CAUSAL_LM",  # Task type for causal language modeling
)

# Apply LoRA configuration to the model
model = get_peft_model(model, peft_params)

# Update model configuration to disable caching and set pretraining tensor parallelism
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token
tokenizer.padding_side = "right"  # Ensure padding is applied to the right

# Load dataset from CSV file
csv_file_path = r"C:\Users\Computing\Downloads\Llama\Dataset\Dataset\train.csv"
df = pd.read_csv(csv_file_path)

# Sample the first 10,000 rows from the dataset for training
df_sampled = df.iloc[:10000]

# Split the sampled data into training (80%) and evaluation (20%) datasets
train_df = df_sampled.iloc[:8000]  # First 80% for training
eval_df = df_sampled.iloc[8000:]  # Remaining 20% for evaluation

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Define preprocessing function to format and tokenize input data
def preprocess_function(examples):
    questions = examples['question']
    options_a = examples['opa']
    options_b = examples['opb']
    options_c = examples['opc']
    options_d = examples['opd']
    correct_options = examples['cop']  # Index of the correct option
    explanations = examples.get('exp', [''] * len(questions))  # Use empty strings if 'exp' column does not exist

    inputs = []
    labels = []

    for question, opa, opb, opc, opd, correct_option, explanation in zip(questions, options_a, options_b, options_c, options_d, correct_options, explanations):
        # Format input string with question and options
        options_formatted = [f"{chr(97+i)}: {opt}" for i, opt in enumerate([opa, opb, opc, opd])]
        input_str = f"question: {question} {' '.join(options_formatted)}"
        
        if explanation:  # Append explanation if it exists
            input_str += f" Explanation: {explanation}"

        inputs.append(input_str)
        
        # Process label (convert correct_option index to letter)
        label_str = chr(97 + correct_option)  # Map index to letter ('a', 'b', 'c', 'd')
        labels.append(label_str)

    # Tokenize inputs and labels
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels_tokenized = tokenizer(labels, max_length=35, truncation=True, padding="max_length")["input_ids"]
    
    model_inputs["labels"] = labels_tokenized
    return model_inputs

# Apply preprocessing function to training and evaluation datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on train dataset")
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on eval dataset")

# Define training parameters for fine-tuning
training_params = TrainingArguments(
    output_dir="./results",  # Directory to save results
    num_train_epochs=3,  # Number of epochs for training
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=1,  # Number of steps to accumulate gradients
    optim="paged_adamw_32bit",  # Optimizer to use
    save_steps=200,  # Save model every 200 steps
    logging_steps=200,  # Log training information every 200 steps
    learning_rate=3e-4,  # Learning rate for training
    weight_decay=0.001,  # Weight decay for regularization
    fp16=True,  # Use 16-bit floating point precision
    bf16=False,  # Use bfloat16 precision (if False, it will not be used)
    max_grad_norm=0.3,  # Maximum gradient norm for clipping
    max_steps=-1,  # Number of training steps (if -1, train for num_train_epochs)
    warmup_ratio=0.03,  # Ratio of steps for learning rate warmup
    group_by_length=True,  # Group samples by length for efficient training
    lr_scheduler_type="constant",  # Learning rate scheduler type
    report_to="tensorboard",  # Report results to TensorBoard
    evaluation_strategy="epoch",  # Evaluate model at the end of each epoch
    eval_steps=200,  # Evaluate model every 200 steps
    push_to_hub=False,  # Disable pushing to Hugging Face model hub
)

# Define a custom callback class for logging training progress
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

# Initialize the SFTTrainer with the specified parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    peft_config=peft_params,  # Pass LoRA configuration
    dataset_text_field="text",  # Specify the text field in the dataset
    max_seq_length=None,  # No maximum sequence length constraint
    tokenizer=tokenizer,
    args=training_params,
    packing=False,  # Disable packing sequences
    callbacks=[log_callback],  # Add the custom logging callback
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Save training logs to an Excel file for later analysis
df = pd.DataFrame(log_callback.training_logs)
df.to_excel('training_logs.xlsx', index=False)
