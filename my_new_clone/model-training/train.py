# The training of the model for the classification and summarization
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
import torch
from sklearn.metrics import f1_score
import os
import sys

# Adding the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Forcing the CPU usage as CUDA was unable to install into some system used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    # Loading the dataset
    df = pd.read_csv("../data-generation/financial_transactions.csv")

    # Adding the label IDs
    df['label'] = df['category'].map(config.label2id)

    # Splitting into train and test dataset as 80-20 split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Converting to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Initializing the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["description"],
            truncation=True,
            max_length=64,
            padding="max_length"
        )

    # Tokenized datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Loading the model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.CATEGORIES),
        id2label=config.id2label,
        label2id=config.label2id
    )

    # Applying LoRA as CPU version while GPU is required for the QLoRA
    lora_config = LoraConfig(**config.LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training the arguments
    training_args = TrainingArguments(**config.TRAINING_ARGS)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Metrics function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predictions, average="weighted")
        acc = np.mean(predictions == labels)
        return {"accuracy": acc, "f1": f1}

    # Trainer for the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training the dataset
    print("🚀 Starting training...")
    trainer.train()

    # Saving the model for future usage
    trainer.save_model()
    tokenizer.save_pretrained(config.TRAINING_ARGS["output_dir"])
    print(f"💾 Model saved to {config.TRAINING_ARGS['output_dir']}")

    # Evaluating the model
    results = trainer.evaluate()
    print("\n📊 Evaluation Results:")
    print(f"Validation Loss: {results['eval_loss']:.4f}")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")


if __name__ == "__main__":
    main()

# This was the model training while it is CPU based so it will be 2-3x slower than the GPU