import pandas as pd
from collections import Counter

def simple_summarizer(transactions: pd.DataFrame) -> str:
    df = transactions.copy()
    df['Amount'] = df['Amount'].astype(int)  # Ensure numeric amounts

    category_counts = Counter(df['Category'])
    category_sums = df.groupby('Category')['Amount'].sum().to_dict()

    # Finding the most frequent category
    most_frequent = category_counts.most_common(1)[0]
    remaining_counts = category_counts.most_common()[1:]

    second_highest_count = remaining_counts[0][1] if remaining_counts else 0
    second_categories = [cat for cat, count in category_counts.items()
                         if count == second_highest_count and cat != most_frequent[0]]

    if len(second_categories) > 1:
        place = " each"
    else:
        place = ""

    # highest amount on the category
    highest_spend_category = max(category_sums, key=category_sums.get)

    # Formating the required output
    output = (
        f"The most frequent category in the users transactions is {most_frequent[0]} "
        f"which occurred {most_frequent[1]} times while below it is "
        f"{' and '.join(second_categories)} which occurred {second_highest_count} times{place}. "
        f"While the most amount is spend in {highest_spend_category} "
        f"which is NPR {category_sums[highest_spend_category]}."
    )

    return output
