# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_personality_dataset.csv")

# Set X = Features and y = target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=13)
knn_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
# joblib.dump(knn_model, "personality_model.pkl")
# print("Model saved as personality_model.pkl")
