import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    TrainerCallback
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import pandas as pd
import logging

# Set up logging to record the training process into a file
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Define the base model and the new model name
base_model = "meta-llama/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-arunabh"

# Set the data type for computation (16-bit floating point)
compute_dtype = getattr(torch, "float16")

# Configure quantization settings for loading the model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model in 4-bit precision for efficiency
    bnb_4bit_quant_type="nf4",  # Set the type of 4-bit quantization
    bnb_4bit_compute_dtype=compute_dtype,  # Set computation to 16-bit
    bnb_4bit_use_double_quant=False,  # Disable double quantization
)

# Load and quantize the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}  # Map the model to the available device (e.g., GPU)
)

# Prepare the model for k-bit training (low-bit precision training)
model = prepare_model_for_kbit_training(model)

# Define LoRA (Low-Rank Adaptation) configuration for fine-tuning
peft_params = LoraConfig(
    lora_alpha=16,  # Scaling factor for LoRA layers
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    r=64,  # Rank of the LoRA update matrices
    bias="none",  # Do not update biases
    task_type="CAUSAL_LM",  # Specify task type as Causal Language Modeling
)

# Apply the PEFT configuration to the model
model = get_peft_model(model, peft_params)

# Disable caching and set other model configurations for training
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load the tokenizer for the model and adjust padding settings
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token as the EOS token
tokenizer.padding_side = "right"  # Pad sequences on the right side
# Function to print the number of trainable parameters in the model
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# Log the number of trainable parameters
print_trainable_parameters(model)
# Load the medical QA dataset for training
DATA_NAME = "medalpaca/medical_meadow_medqa"
med_answers_qa = load_dataset(DATA_NAME)
med_answers_qa = med_answers_qa["train"].select(range(10000))  # Select a subset of 10,000 examples
med_answers_qa = med_answers_qa.train_test_split(test_size=0.2)  # Split into training and testing sets

# Define a preprocessing function to format the dataset for training
def preprocess_function(examples):
    inputs = []
    for instruction, input_data in zip(examples["instruction"], examples["input"]):
        # Extract and format the question and options from input data
        question = input_data.split("\n")[0].strip()
        options_str = input_data.split("\n")[1].strip()[:-1]
        labels_list = options_str.split("', '")

        options_formatted = [f"{key}: {value}" for key, value in zip(labels_list, options_str.split("', '")[1::2])]
        input_str = f"instruction: {instruction} input: {question} {' '.join(options_formatted)}"
        inputs.append(input_str)

    # Tokenize inputs and prepare labels for the model
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=35, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Apply the preprocessing function to the dataset
tokenized_dataset = med_answers_qa.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on dataset")

# Set up training arguments for the model
training_params = TrainingArguments(
    output_dir="./results",  # Directory to save training results
    num_train_epochs=5,  # Number of epochs to train
    per_device_train_batch_size=2,  # Batch size per device during training
    per_device_eval_batch_size=2,  # Batch size per device during evaluation
    gradient_accumulation_steps=1,  # Accumulate gradients over this many steps before updating model weights
    optim="paged_adamw_32bit",  # Use AdamW optimizer with 32-bit precision
    save_steps=100,  # Save the model every 100 steps
    logging_steps=100,  # Log training progress every 100 steps
    learning_rate=2e-4,  # Set learning rate for the optimizer
    weight_decay=0.001,  # Apply weight decay for regularization
    fp16=True,  # Use 16-bit floating point precision for faster training
    bf16=False,  # Do not use bfloat16 precision
    max_grad_norm=0.3,  # Maximum gradient norm for gradient clipping
    max_steps=-1,  # Train for the specified number of steps (or indefinitely if set to -1)
    warmup_ratio=0.03,  # Warm up the learning rate for the first 3% of steps
    group_by_length=True,  # Group sequences of similar length to improve training efficiency
    lr_scheduler_type="constant",  # Use a constant learning rate scheduler
    report_to="tensorboard",  # Report training progress to TensorBoard
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    eval_steps=200,  # Evaluate the model every 200 steps
    push_to_hub=False,  # Do not push the model to the Hugging Face Hub
)

# Define a custom callback for logging training progress
class LogCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_logs = []

    # Log training details at each step
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
            logger.info(log_entry)  # Record the log entry

# Instantiate the logging callback
log_callback = LogCallback()

# Set up the trainer for supervised fine-tuning (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],  # Provide the training dataset
    eval_dataset=tokenized_dataset["test"],  # Provide the evaluation dataset
    peft_config=peft_params,  # Apply PEFT configurations
    dataset_text_field="text",  # Specify the text field in the dataset
    max_seq_length=None,  # Do not set a maximum sequence length (use tokenizer settings)
    tokenizer=tokenizer,  # Provide the tokenizer
    args=training_params,  # Use the defined training arguments
    packing=False,  # Disable sequence packing
    callbacks=[log_callback],  # Add the logging callback to the trainer
)

# Start the training process
trainer.train()

# Save the trained model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Save the training logs to an Excel file for further analysis
df = pd.DataFrame(log_callback.training_logs)
df.to_excel('training_logs.xlsx', index=False)
