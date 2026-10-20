import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv('sms.tsv', sep='\t', names=['label', 'message'])

# 2. Convert label to numbers (ham=0, spam=1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Features & Target
X = df['message']
y = df['label_num']

# 4. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 5. TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train the Multinomial Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Predict on Test Data (Model-oda exam)
y_pred = model.predict(X_test_tfidf)

# 8. Check Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred, target_names=['HAM (Normal)', 'SPAM']))

# 9. Test with custom messages!
test_messages = [
    "Congratulations! You won a $1,000 Walmart gift card. Click here to claim.",
    "Hey bro, are we meeting at the library tomorrow for studying?"
]
test_messages_tfidf = tfidf.transform(test_messages)
predictions = model.predict(test_messages_tfidf)

print("\n--- Custom Message Live Test ---")
for msg, pred in zip(test_messages, predictions):
    result = "SPAM" if pred == 1 else "NOT SPAM (HAM)"
    print(f"Message: '{msg}' --> Prediction: {result}")
    import pickle

# Model and TF-IDF vectorizer-ah file-ah save panrom
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nSuccess! 'model.pkl' and 'tfidf.pkl' saved successfully! ✅")