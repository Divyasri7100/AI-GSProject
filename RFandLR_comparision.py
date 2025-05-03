# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('forestfires.csv')

# Encode categorical features
df['month'] = LabelEncoder().fit_transform(df['month'])
df['day'] = LabelEncoder().fit_transform(df['day'])

# Create binary target: fire occurred or not
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)

# Features and Target
X = df.drop(['area', 'fire'], axis=1)
y = df['fire']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------
# Train Random Forest Model
# --------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# -------------------------------
# Train Logistic Regression Model
# -------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# ----------------------------
# Compare Model Performance
# ----------------------------
print("🔍 Model Comparison Results:\n")

# Accuracy Scores
rf_acc = accuracy_score(y_test, rf_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"Random Forest Accuracy:        {rf_acc:.4f}")
print(f"Logistic Regression Accuracy:  {lr_acc:.4f}\n")

# Classification Reports
print("📊 Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("📊 Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred))

# ----------------------------
# Plot Confusion Matrices
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Logistic Regression Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
