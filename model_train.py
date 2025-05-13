import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("iris.csv")

# Drop Id column if exists
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Separate features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as iris_model.pkl")
