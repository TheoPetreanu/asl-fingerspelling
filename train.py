import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
df = pd.read_csv('training_data.csv', header=None)
X = df.iloc[:, 1:].values  # all columns except first (the features)
y = df.iloc[:, 0].values   # first column (the label)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(kernel='rbf', probability=True, C=10)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print(classification_report(y_test, y_pred))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved to model.pkl")