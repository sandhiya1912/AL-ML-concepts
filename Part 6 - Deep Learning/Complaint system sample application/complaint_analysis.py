# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Download nltk stopwords
nltk.download('stopwords')

# Load complaints dataset (adjust the file path as needed)
dataset = pd.read_csv('complaints_data.tsv', delimiter='\t', quoting=3)

# Preprocess the complaints text
corpus = []
ps = PorterStemmer()
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Complaint'][i])
    review = review.lower()
    review = review.split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  # Keeping the word 'not' for sentiment analysis
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Create Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

# Set y to the 'Priority' column for initial complaint priority (we can modify this based on sentiment)
y = dataset.iloc[:, -1].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train a Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to predict and prioritize complaints
def predict_multiple_complaints(complaints_list):
    priorities = []
    
    for complaint_text in complaints_list:
        # Preprocess each complaint
        complaint = re.sub('[^a-zA-Z]', ' ', complaint_text)
        complaint = complaint.lower()
        complaint = complaint.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        complaint = [ps.stem(word) for word in complaint if not word in set(all_stopwords)]
        complaint = ' '.join(complaint)

        # Create Bag of Words model for the input complaint
        complaint_vector = cv.transform([complaint]).toarray()

        # Predict sentiment and priority
        predicted_priority = classifier.predict(complaint_vector)

        # Assign priority based on prediction
        if predicted_priority[0] == 1:
            priorities.append('High Priority')
        elif predicted_priority[0] == 2:
            priorities.append('Medium Priority')
        else:
            priorities.append('Low Priority')

    return priorities

# Example usage with multiple complaints
multiple_complaints = [
    "The garbage truck comes very late every day.",
    "The streetlights are not working.",
    "My water supply is irregular.",
    "The public park is not clean.",
    "The traffic lights are broken."
]

# Predict and print priorities for each complaint
predicted_priorities = predict_multiple_complaints(multiple_complaints)
for complaint, priority in zip(multiple_complaints, predicted_priorities):
    print(f'Complaint: "{complaint}" is classified as {priority}')
