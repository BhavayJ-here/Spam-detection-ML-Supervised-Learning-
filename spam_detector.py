import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


dataset_path = "data/SMSSpamCollection"

data = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "message"])

print("Dataset loaded")
print(f"Total messages: {len(data)}")
print(f"Spam: {sum(data['label'] == 'spam')}")
print(f"Ham:  {sum(data['label'] == 'ham')}")
print()


data["label_num"] = data["label"].map({"spam": 1, "ham": 0})


def get_extra_features(message):
    features = {}
    features["length"] = len(message)
    features["num_words"] = len(message.split())
    features["num_exclamation"] = message.count("!")
    features["num_uppercase"] = sum(1 for c in message if c.isupper())
    features["has_link"] = 1 if ("http" in message or "www" in message) else 0
    features["num_digits"] = sum(1 for c in message if c.isdigit())
    return features

extra_features_list = []
for msg in data["message"]:
    extra_features_list.append(get_extra_features(msg))

extra_df = pd.DataFrame(extra_features_list)


tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["message"])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

all_features = pd.concat([tfidf_df.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1)

print(f"Total features created: {all_features.shape[1]}")
print()


X = all_features
y = data["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print()


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model trained")
print()


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

spam_probabilities = y_proba[:, 1]


HIGH_CONFIDENCE_SPAM = 0.75
HIGH_CONFIDENCE_HAM  = 0.35

def route_message(probability):
    if probability >= HIGH_CONFIDENCE_SPAM:
        return "SPAM"
    elif probability <= HIGH_CONFIDENCE_HAM:
        return "HAM"
    else:
        return "UNCERTAIN"

routing_results = []
for prob in spam_probabilities:
    routing_results.append(route_message(prob))

routing_series = pd.Series(routing_results)

print("Confidence Threshold Results:")
print(f"  AUTO-SPAM   : {sum(routing_series == 'SPAM')}")
print(f"  UNCERTAIN   : {sum(routing_series == 'UNCERTAIN')}")
print(f"  AUTO-HAM    : {sum(routing_series == 'HAM')}")
print()


print("Model Performance:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))


cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(f"                 Predicted Ham   Predicted Spam")
print(f"  Actual Ham     {str(tn).ljust(15)} {fp}")
print(f"  Actual Spam    {str(fn).ljust(15)} {tp}")
print()


results_df = pd.DataFrame()
results_df["actual_label"]    = y_test.values
results_df["predicted_label"] = y_pred
results_df["spam_probability"] = spam_probabilities
results_df["routing"]         = routing_results
results_df["original_message"] = data.loc[X_test.index, "message"].values

false_positives = results_df[(results_df["actual_label"] == 0) & (results_df["predicted_label"] == 1)]
false_negatives = results_df[(results_df["actual_label"] == 1) & (results_df["predicted_label"] == 0)]

print(f"False Positives (ham flagged as spam) : {len(false_positives)}")
print(f"False Negatives (spam that slipped)   : {len(false_negatives)}")
print()

if len(false_positives) > 0:
    print("Sample false positives:")
    for i, row in false_positives.head(3).iterrows():
        print(f"  [prob={round(row['spam_probability'], 2)}] {row['original_message'][:80]}")
    print()

if len(false_negatives) > 0:
    print("Sample false negatives:")
    for i, row in false_negatives.head(3).iterrows():
        print(f"  [prob={round(row['spam_probability'], 2)}] {row['original_message'][:80]}")
    print()


bins = np.linspace(0, 1, 11)
ham_probs  = spam_probabilities[np.array(y_test) == 0]
spam_probs = spam_probabilities[np.array(y_test) == 1]

ham_counts,  _ = np.histogram(ham_probs,  bins=bins)
spam_counts, _ = np.histogram(spam_probs, bins=bins)

print("Probability Distribution (each bar = 10% range):")
print(f"  {'Range':<12} {'Ham':>6}  {'Spam':>6}")
print(f"  {'-'*28}")
for i in range(len(bins) - 1):
    range_label = f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%"
    ham_bar  = "=" * min(ham_counts[i],  40)
    spam_bar = "=" * min(spam_counts[i], 40)
    print(f"  {range_label:<12} HAM  [{ham_bar:<40}] {ham_counts[i]}")
    print(f"  {'':<12} SPAM [{spam_bar:<40}] {spam_counts[i]}")
print()
print(f"  Spam threshold  : >= {HIGH_CONFIDENCE_SPAM} → auto block")
print(f"  Ham threshold   : <= {HIGH_CONFIDENCE_HAM} → auto deliver")
print(f"  In between      : flag for manual review")
print()


results_df.to_csv("full_results.csv", index=False)
print("Full results saved to full_results.csv")
