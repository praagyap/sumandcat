# The configuration file for the whole model and the dashboard.
class FinancialConfig:
    def __init__(self):
        # Model selection as Distil RoBERTa transformer
        self.MODEL_NAME = "distilroberta-base"

        # Transaction categories for the categorization through the transformers
        self.CATEGORIES = [
            "Mobile Topup", "Education", "Utilities",
            "Food", "Transportation", "Healthcare",
            "Entertainment", "Shopping", "Transfer"
        ]

        # Creating the labeled mappings
        self.id2label = {i: cat for i, cat in enumerate(self.CATEGORIES)}
        self.label2id = {cat: i for i, cat in enumerate(self.CATEGORIES)}

        # Visualization colors for the dashboard so the results are presented with more color.
        self.CATEGORY_COLORS = {
            "Mobile Topup": "#FF9AA2",
            "Education": "#FFB7B2",
            "Utilities": "#FFDAC1",
            "Food": "#E2F0CB",
            "Transportation": "#B5EAD7",
            "Healthcare": "#C7CEEA",
            "Entertainment": "#F8B195",
            "Shopping": "#F67280",
            "Transfer": "#6C5B7B"
        }

        # LoRA configuration for parameters tuning
        self.LORA_CONFIG = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["query", "value"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "SEQ_CLS"
        }

        # Training arguments which will be used
        self.TRAINING_ARGS = {
            "output_dir": "../trained_models/finetuned_model",
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 100,
            "save_total_limit": 2,
            "report_to": "none",
            "no_cuda": True
        }


config = FinancialConfig()

# because of the no_cuda set to True, the whole model is running on the CPU as CUDA was not possible in all
# available systems
