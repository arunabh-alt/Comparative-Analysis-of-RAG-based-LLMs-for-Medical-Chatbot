import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import logging
import pandas as pd

# Configure standard logging to save training logs to a file
logging.basicConfig(filename='training_logs.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Set the CUDA device to use GPU 0 for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the pre-trained model and tokenizer
model_name = "google/flan-t5-large"
quant_config = BitsAndBytesConfig(load_in_8bit=True)  # Config for 8-bit quantization
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for the model
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# Function to print the number of trainable parameters and total parameters in the model
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

print_trainable_parameters(model)

# Load and preprocess the dataset
DATA_NAME = "medalpaca/medical_meadow_medqa"
med_answers_qa = load_dataset(DATA_NAME)  # Load the dataset
med_answers_qa = med_answers_qa["train"].select(range(10000))  # Select a subset of the dataset
med_answers_qa = med_answers_qa.train_test_split(test_size=0.2)  # Split dataset into training and test sets

# Function to preprocess the dataset
def preprocess_function(examples):
    inputs = []
    for instruction, input_data in zip(examples["instruction"], examples["input"]):
        question = input_data.split("\n")[0].strip()
        options_str = input_data.split("\n")[1].strip()[:-1]
        labels_list = options_str.split("', '")

        options_formatted = [f"{key}: {value}" for key, value in zip(labels_list, options_str.split("', '")[1::2])]
        input_str = f"instruction: {instruction} input: {question} {' '.join(options_formatted)}"
        inputs.append(input_str)
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=35, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Apply preprocessing to the dataset
tokenized_dataset = med_answers_qa.map(preprocess_function, batched=True, load_from_cache_file=False, desc="Running tokenizer on dataset")

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # Directory to save results
    eval_strategy="epoch",  # Evaluation strategy: evaluate at the end of each epoch
    learning_rate=3e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    num_train_epochs=5,  # Number of epochs for training
    logging_steps=100,  # Log training progress every 100 steps
    predict_with_generate=True,  # Enable generation for prediction
    push_to_hub=False,  # Do not push the model to Hugging Face Hub
    save_total_limit=3  # Limit the number of saved checkpoints
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
                'epoch': logs.get('epoch')
            }
            self.training_logs.append(log_entry)
            logger.info(log_entry)  # Log the training progress

# Instantiate the custom logging callback
log_callback = LogCallback()

# Data collator for handling dynamic padding and batch processing
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize the Seq2SeqTrainer with the model, arguments, and datasets
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[log_callback]  # Add custom callback to the trainer
)

# Disable cache to avoid using previously cached results
model.config.use_cache = False

# Start training the model
trainer.train()

# Save the trained model to the specified directory
trainer.save_model("results/T5-Finetune-Arunabh")

# Save the training logs to an Excel file for further analysis
df = pd.DataFrame(log_callback.training_logs)
df.to_excel('training_logs.xlsx', index=False)
