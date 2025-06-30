# The inference for the dashboard for the Streamlit dashboard for the visulization
from utils import load_model, preprocess_transaction, extract_amount
import torch


class TransactionClassifier:
    def __init__(self, model_path="../trained_models/finetuned_model"):
        self.model, self.tokenizer = load_model(model_path)
        self.model.eval()  # Setting the model to evaluation mode

    def categorize(self, transactions):
        """Categorizing one or more transactions"""
        device = self.model.device  # Getting the MODEL'S device as CPU or GPU
        if isinstance(transactions, str):
            transactions = [transactions]

        results = []
        for tx in transactions:
            # Preprocess the input data
            cleaned_tx = preprocess_transaction(tx)

            # Tokenize the input
            inputs = self.tokenizer(
                cleaned_tx,
                return_tensors="pt",
                max_length=64,
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()} # this also regarding the device

            # Predicting through the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predicted category
            logits = outputs.logits
            predicted_idx = logits.argmax().item()
            category = self.model.config.id2label[predicted_idx]

            # Extract amount from the input
            amount = extract_amount(tx)

            # Calculate confidence for the prediction
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence = probabilities[0][predicted_idx].item()

            results.append({
                "transaction": tx,
                "category": category,
                "amount": amount,
                "confidence": confidence
            })

        return results
# This is a simple inference for the dashboard while API will also be created through FastAPI for the project