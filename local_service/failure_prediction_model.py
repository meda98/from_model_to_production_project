import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv("predictive_maintenance.csv")

# Rename columns to match project sensor terminology
df = df.rename(columns={
    "Air temperature [K]": "temperature",
    "Process temperature [K]": "humidity",
    "Rotational speed [rpm]": "sound_volume"
})

# Select features using the new names
X = df[["temperature", "humidity", "sound_volume"]]

# Define target variable
y = df["Target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Save model
joblib.dump(model, "model.pkl")

# Evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))