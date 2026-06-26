# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from PIL import Image
import pickle

# Load the MNIST dataset
data = pd.read_csv("train.csv")

print("Dataset Shape:", data.shape)

# Display a sample handwritten digit
label = data.iloc[0, 0]
image = data.iloc[0, 1:].values.reshape(28, 28)

plt.imshow(image, cmap='gray')
plt.title(f"Digit: {label}")
plt.show()

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Normalize pixel values
X = X / 255.0

# Create inverted images for background variation
X_inverted = 1 - X

# Combine original and inverted images
X = pd.concat([X, X_inverted], ignore_index=True)
y = pd.concat([y, y], ignore_index=True)

print("Augmented Dataset Shape:", X.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate classification report
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Display sample predictions
for i in range(5):

    image = X_test.iloc[i].values.reshape(28, 28)

    plt.imshow(image, cmap='gray')
    plt.title(
        f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]}"
    )
    plt.show()

# Save trained model
with open("mnist_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully")

# Save sample digit image
sample_image = X_test.iloc[0].values.reshape(28, 28)
sample_image = (sample_image * 255).astype(np.uint8)

img = Image.fromarray(sample_image)
img.save("digit_sample.png")

print("Sample Image Saved Successfully")