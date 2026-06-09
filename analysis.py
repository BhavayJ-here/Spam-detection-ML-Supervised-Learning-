import pandas as pd
import numpy as np
from collections import Counter
import re
import os


if not os.path.exists("full_results.csv"):
    print("ERROR: full_results.csv not found")
    print("Please run spam_detector.py first")
    exit()

results = pd.read_csv("full_results.csv")

print("Results loaded")
print(f"Total test samples: {len(results)}")
print()


false_positives = results[(results["actual_label"] == 0) & (results["predicted_label"] == 1)]
false_negatives = results[(results["actual_label"] == 1) & (results["predicted_label"] == 0)]
true_positives  = results[(results["actual_label"] == 1) & (results["predicted_label"] == 1)]
true_negatives  = results[(results["actual_label"] == 0) & (results["predicted_label"] == 0)]

print("Prediction Breakdown:")
print(f"  Correct spam catches (TP)   : {len(true_positives)}")
print(f"  Correct ham deliveries (TN) : {len(true_negatives)}")
print(f"  Ham wrongly blocked (FP)    : {len(false_positives)}")
print(f"  Spam that slipped (FN)      : {len(false_negatives)}")
print()


precision = len(true_positives) / (len(true_positives) + len(false_positives) + 0.0001)
recall    = len(true_positives) / (len(true_positives) + len(false_negatives) + 0.0001)
f1        = 2 * precision * recall / (precision + recall + 0.0001)
accuracy  = (len(true_positives) + len(true_negatives)) / len(results)

print("Metrics (calculated manually):")
print(f"  Accuracy  : {round(accuracy * 100, 2)}%")
print(f"  Precision : {round(precision * 100, 2)}%")
print(f"  Recall    : {round(recall * 100, 2)}%")
print(f"  F1 Score  : {round(f1 * 100, 2)}%")
print()


def get_common_words(messages, top_n=10):
    all_words = []
    for msg in messages:
        words = str(msg).lower().split()
        cleaned = [re.sub(r'[^a-z]', '', w) for w in words]
        cleaned = [w for w in cleaned if len(w) > 2]
        all_words.extend(cleaned)
    return Counter(all_words).most_common(top_n)


print("Top words in FALSE POSITIVES (ham the model got wrong):")
if len(false_positives) > 0:
    for word, count in get_common_words(false_positives["original_message"]):
        bar = "=" * count
        print(f"  {word:<15} [{bar:<20}] {count}")
else:
    print("  None — no false positives!")
print()

print("Top words in FALSE NEGATIVES (spam that slipped through):")
if len(false_negatives) > 0:
    for word, count in get_common_words(false_negatives["original_message"]):
        bar = "=" * count
        print(f"  {word:<15} [{bar:<20}] {count}")
else:
    print("  None — no false negatives!")
print()


print("Message Length Analysis:")
true_positives["msg_length"]  = true_positives["original_message"].str.len()
true_negatives["msg_length"]  = true_negatives["original_message"].str.len()
false_negatives["msg_length"] = false_negatives["original_message"].str.len()
false_positives["msg_length"] = false_positives["original_message"].str.len()

print(f"  Avg length - correctly caught spam  : {round(true_positives['msg_length'].mean(), 1)} chars")
print(f"  Avg length - spam that slipped      : {round(false_negatives['msg_length'].mean() if len(false_negatives) > 0 else 0, 1)} chars")
print(f"  Avg length - correctly delivered ham: {round(true_negatives['msg_length'].mean(), 1)} chars")
print(f"  Avg length - ham wrongly blocked    : {round(false_positives['msg_length'].mean() if len(false_positives) > 0 else 0, 1)} chars")
print()


print("Confidence Score Analysis:")
print(f"  Avg confidence for TP : {round(true_positives['spam_probability'].mean(), 3)}")
print(f"  Avg confidence for TN : {round(true_negatives['spam_probability'].mean(), 3)}")
if len(false_positives) > 0:
    print(f"  Avg confidence for FP : {round(false_positives['spam_probability'].mean(), 3)}")
if len(false_negatives) > 0:
    print(f"  Avg confidence for FN : {round(false_negatives['spam_probability'].mean(), 3)}")
print()


uncertain = results[results["routing"] == "UNCERTAIN"]
print(f"Uncertain messages flagged for manual review: {len(uncertain)}")
if len(uncertain) > 0:
    print(f"  Of those, actual spam : {sum(uncertain['actual_label'] == 1)}")
    print(f"  Of those, actual ham  : {sum(uncertain['actual_label'] == 0)}")
print()


print("Routing Distribution:")
routing_counts = results["routing"].value_counts()
total = len(results)
for label, count in routing_counts.items():
    pct = round(count / total * 100, 1)
    bar = "=" * int(pct)
    print(f"  {label:<10} [{bar:<50}] {count} ({pct}%)")
print()


print("Summary:")
print(f"  The model caught {round(recall*100, 1)}% of all spam (recall)")
print(f"  When it said spam, it was right {round(precision*100, 1)}% of the time (precision)")
print(f"  {len(uncertain)} messages were uncertain and got flagged for manual review")
print(f"  {len(false_positives)} legitimate emails were wrongly blocked")
print(f"  {len(false_negatives)} spam emails slipped through")
