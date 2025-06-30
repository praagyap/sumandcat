# The generation of the datasets for the usage in the model training
import pandas as pd
from faker import Faker
import random
import re
import os
import sys

# Adding the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def generate_dataset(num_samples=300):
    """Generating synthetic financial transaction datasets"""
    fake = Faker()

    templates = {
        "Mobile Topup": [
            "Paid NPR {amount} for {operator} topup",
            "Recharged {operator} mobile with NPR {amount}",
            "Mobile recharge for {operator} NPR {amount}"
        ],
        "Education": [
            "Paid Rs {amount} to {institution} for tuition",
            "Sent {amount} to {institution} for course fees",
            "Education payment to {institution} Rs {amount}",
            "Paid Rs {amount} for online course fees"
        ],
        "Utilities": [
            "Paid Rs {amount} for {utility} bill",
            "{utility} bill payment NPR {amount}",
            "Cleared {utility} dues Rs {amount}"
        ],
        "Food": [
            "Spent ₹{amount} at {restaurant}",
            "Dining at {restaurant} Rs {amount}",
            "Food order from {restaurant} ₹{amount}"
        ],
        "Transportation": [
            "Fuel at {station} Rs {amount}",
            "Taxi fare Rs {amount}",
            "Public transport pass NPR {amount}"
        ],
        "Healthcare": [
            "Medical bill at {hospital} Rs {amount}",
            "Pharmacy purchase NPR {amount}",
            "Doctor consultation fee Rs {amount}"
        ],
        "Entertainment": [
            "{service} subscription Rs {amount}",
            "Movie tickets at {cinema} NPR {amount}",
            "Concert tickets Rs {amount}"
        ],
        "Shopping": [
            "Purchase at {store} Rs {amount}",
            "Online shopping {amount}",
            "{item} bought for Rs {amount}"
        ],
        "Transfer": [
            "Sent Rs {amount} to {person}",
            "Transfer to {account} NPR {amount}",
            "Money sent to {person} ₹{amount}"
        ]
    }

    # Entity generators
    entity_generators = {
        "operator": lambda: random.choice(["NTC", "Ncell", "Smart Cell"]),
        "institution": lambda: random.choice(["Tribhuvan University", "Kathmandu University", "Whitehouse College"]),
        "utility": lambda: random.choice(["electricity", "water", "internet", "gas"]),
        "restaurant": lambda: random.choice(["Roadhouse Cafe", "OR2K", "KFC", "Pizza Hut"]),
        "station": lambda: random.choice(["Nepal Oil", "Sajha Petrol", "Shiva Fuel"]),
        "hospital": lambda: random.choice(["Norvic", "Grande", "Mediciti", "Teaching Hospital"]),
        "service": lambda: random.choice(["Netflix", "YouTube Premium", "Spotify", "Amazon Prime"]),
        "cinema": lambda: random.choice(["QFX", "Big Movies", "FCube"]),
        "store": lambda: random.choice(["Daraz", "Sastodeal", "New Road Shop"]),
        "item": lambda: random.choice(["clothes", "electronics", "groceries", "books"]),
        "person": lambda: fake.name(),
        "account": lambda: fake.bban()
    }

    data = []
    for _ in range(num_samples):
        category = random.choice(config.CATEGORIES)
        template = random.choice(templates[category])

        # Replacing the placeholders with actual values for the dataset
        for placeholder in re.findall(r"\{(\w+)\}", template):
            if placeholder in entity_generators:
                value = entity_generators[placeholder]()
                template = template.replace(f"{{{placeholder}}}", value)

        # Adding the random amount but for more accuracy, adding the relevent amount could be done
        amount = random.randint(50, 20000)
        description = template.replace("{amount}", str(amount))

        data.append({
            "description": description,
            "category": category,
            "amount": amount
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate dataset with 1000 samples
    df = generate_dataset(1000)

    # Saving to the CSV file
    df.to_csv("financial_transactions.csv", index=False)
    print("✅ Dataset generated with 1000 transactions")
    print(f"📁 Saved to {os.path.abspath('financial_transactions.csv')}")

# The dataset was generated with the data for learning
# For emojis, just press Win+: and a tab with emojis appears