# fake-email-detection
to detect fake email 
# ============================================
# 📧 Fake Email Detection using ML
# ============================================

# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================
# Step 1: Load Dataset
# ============================================

# Make sure spam.csv is in same folder
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# ============================================
# Step 2: Data Preprocessing
# ============================================

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check for null values
df.dropna(inplace=True)

print("Dataset Loaded Successfully!")
print(df.head())

# ============================================
# Step 3: Train-Test Split
# ============================================

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# Step 4: Feature Extraction (TF-IDF)
# ============================================

vectorizer = TfidfVectorizer(stop_words='english')

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# ============================================
# Step 5: Train Model (Naive Bayes)
# ============================================

model = MultinomialNB()
model.fit(X_train_features, y_train)

# ============================================
# Step 6: Model Evaluation
# ============================================

y_pred = model.predict(X_test_features)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================
# Step 7: Prediction Function
# ============================================

def predict_email(text):
    input_data = vectorizer.transform([text])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        return "Spam Email ❌"
    else:
        return "Real Email ✅"

# ============================================
# Step 8: Test Custom Emails
# ============================================

print("\nTesting Custom Emails:\n")

test_emails = [
    "Congratulations! You have won a free lottery. Click here to claim now!",
    "Hi Aakriti, please submit your assignment by tomorrow.",
    "Get rich fast!!! Earn money from home without investment.",
    "Let's meet for the project discussion at 5 PM."
]

for email in test_emails:
    print(f"Email: {email}")
    print("Prediction:", predict_email(email))
    print("-" * 50)

# ============================================
# Step 9: User Input (Optional)
# ============================================

while True:
    user_input = input("\nEnter an email message (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Exiting program...")
        break
    
    result = predict_email(user_input)
    print("Result:", result)
