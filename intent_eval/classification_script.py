import pandas as pd
from sklearn.metrics import classification_report
from fuzzywuzzy import fuzz

# Sample DataFrame with "Expected Parse" and "Predicted Parse" columns
# df = pd.DataFrame({
#     "Expected Parse": [
#         {"intent": "set_reminder", "arguments": {"event_name": "Earth Fare sales event", "start_date": "September 20", "end_date": "September 26"}},
#         {"intent": "set_reminder", "arguments": {"event_name": "Meeting", "date": "April 30"}}
#     ],
#     "Predicted Parse": [
#         {"intent": "set_reminder", "arguments": {"event_name": "Earth Fare sales event", "date": "September 12"}},
#         {"intent": "set_reminder", "arguments": {"event_name": "Meeting", "date": "April 30"}}
#     ]
# })

# Function to compute fuzzy match accuracy for argument fields
def compute_fuzzy_match_accuracy(expected_args, predicted_args):
    fuzzy_scores = {}
    for key in expected_args:
        if key in predicted_args:
            expected_value = expected_args[key]
            predicted_value = predicted_args[key]
            ratio = fuzz.ratio(expected_value, predicted_value)
            fuzzy_scores[key] = ratio
    total_keys = len(fuzzy_scores)
    total_accuracy = sum(fuzzy_scores.values()) / total_keys if total_keys > 0 else 0
    return total_accuracy, fuzzy_scores

# Initialize lists to store intent and argument data
expected_intents = []
predicted_intents = []
fuzzy_accuracy_scores = []

# Iterate over rows of the DataFrame
for index, row in df.iterrows():
    expected_parse = row["Expected Parse"]
    predicted_parse = row["Predicted Parse"]

    # Extracting intent from both expected and predicted parses
    expected_intent = expected_parse["intent"]
    predicted_intent = predicted_parse["intent"]

    # Append intent data to lists
    expected_intents.append(expected_intent)
    predicted_intents.append(predicted_intent)

    # Compute fuzzy match accuracy for the argument field
    expected_args = expected_parse["arguments"]
    predicted_args = predicted_parse["arguments"]
    accuracy, _ = compute_fuzzy_match_accuracy(expected_args, predicted_args)
    fuzzy_accuracy_scores.append(accuracy)

# Overall classification report for intent
overall_classification_report = classification_report(expected_intents, predicted_intents)

# Overall fuzzy match accuracy for argument fields
overall_fuzzy_accuracy = sum(fuzzy_accuracy_scores) / len(fuzzy_accuracy_scores)

print("Overall Classification Report for 'Intent' Field:")
print(overall_classification_report)

print("\nOverall Fuzzy Match Accuracy for Argument Fields:")
print(f"Overall Accuracy: {overall_fuzzy_accuracy:.2f}")