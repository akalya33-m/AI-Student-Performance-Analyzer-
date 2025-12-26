import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("data/students.csv")

X = df[['math', 'reading', 'writing', 'attendance']]
y = df['result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open("models/performance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")
