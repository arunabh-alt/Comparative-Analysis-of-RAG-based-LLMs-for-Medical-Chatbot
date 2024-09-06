import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, TrainingArguments, TrainerCallback
from trl import SFTTrainer
import os
import pandas as pd
import logging

# Set up logging to record training details to a file
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Load the tokenizer from a pre-trained model
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GPTQ")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to be the end of sequence token
tokenizer.padding_side = 'right'  # Configure padding to the right side of sequences

# Load the model with quantization settings for efficient inference
quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True, tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    quantization_config=quantization_config_loading,
    device_map="auto"  # Automatically map the model to available devices (e.g., GPUs)
)

# Disable the use of cache and enable gradient checkpointing for memory efficiency during training
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Prepare the model for k-bit (low-bit precision) training to reduce memory usage
model = prepare_model_for_kbit_training(model)

# Configure PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) settings
peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

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

# Load and prepare the medical QA dataset
DATA_NAME = "medalpaca/medical_meadow_medqa"
med_answers_qa = load_dataset(DATA_NAME)
med_answers_qa = med_answers_qa["train"].select(range(10000))  # Select a subset of 10,000 examples for training
med_answers_qa = med_answers_qa.train_test_split(test_size=0.2)  # Split the dataset into training and test sets

# Preprocessing function to format the dataset for training
def preprocess_function(examples):
    inputs = []
    for instruction, input_data in zip(examples["instruction"], examples["input"]):
        question = input_data.split("\n")[0].strip()  # Extract the question part
        options_str = input_data.split("\n")[1].strip()[:-1]  # Extract the options part
        labels_list = options_str.split("', '")  # Split options into a list

        # Format the input for the model
        options_formatted = [f"{key}: {value}" for key, value in zip(labels_list, options_str.split("', '")[1::2])]
        input_str = f"instruction: {instruction} input: {question} {' '.join(options_formatted)}"
        inputs.append(input_str)

    # Tokenize inputs and prepare labels
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=35, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Apply the preprocessing function to the dataset
tokenized_dataset = med_answers_qa.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on dataset")

# Define training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,  # Set batch size for each device during training
    per_device_eval_batch_size=4,  # Set batch size for each device during evaluation
    gradient_accumulation_steps=1,  # Accumulate gradients over this many steps before updating model weights
    optim="paged_adamw_32bit",  # Use AdamW optimizer with 32-bit precision
    learning_rate=3e-4,  # Set learning rate for the optimizer
    lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
    save_strategy="epoch",  # Save model checkpoint at the end of each epoch
    logging_steps=100,  # Log training progress every 100 steps
    num_train_epochs=5,  # Train for 5 epochs
    fp16=True,  # Use 16-bit floating point precision for faster training
    push_to_hub=False  # Do not push model to the Hugging Face Hub
)

# Custom callback to log training progress
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
                'epoch': logs.get('epoch')
            }
            self.training_logs.append(log_entry)
            logger.info(log_entry)  # Log the training progress

# Instantiate the logging callback
log_callback = LogCallback()

# Set up the trainer for supervised fine-tuning (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],  # Provide the training dataset
    eval_dataset=tokenized_dataset["test"],  # Provide the evaluation dataset
    peft_config=peft_config,  # Apply PEFT configurations
    args=training_arguments,  # Use the defined training arguments
    tokenizer=tokenizer,  # Provide the tokenizer
    callbacks=[log_callback],  # Add the logging callback to the trainer
)

# Start the training process
trainer.train()

# Save the training logs to an Excel file for further analysis
df = pd.DataFrame(log_callback.training_logs)
df.to_excel('training_logs.xlsx', index=False)
