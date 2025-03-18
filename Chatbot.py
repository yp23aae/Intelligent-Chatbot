import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle


data = pd.read_csv('go_emotions_dataset.csv')
print(data.head())  


texts = data['text']
labels = data.iloc[:, 3:]  
print(labels.columns) 


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=labels.columns))



with open('emotion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
