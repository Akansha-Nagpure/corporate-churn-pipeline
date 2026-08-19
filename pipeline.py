# 1. Data Simulation and ETL Pipeline
# To generate realistic corporate data. This represents the Extract, Transform, Load (ETL) phase.

import pandas as pd
import numpy as np

print("--- Starting Checkpoint 2: Data ETL Phase ---")

# Set random seed so the data numbers remain consistent every run
np.random.seed(42)
data_size = 150

# Simulate enterprise data fields
raw_data = {
    'CustomerID': range(1001, 1001 + data_size),
    'TenureMonths': np.random.randint(1, 60, size=data_size),
    'MonthlyCharges': np.random.uniform(25.0, 125.0, size=data_size),
    'CustomerFeedback': np.random.choice([
        "Amazing experience, customer support is fast",
        "Terrible interface, system drops constantly and high billing charges",
        "Average platform, but billing issues are annoying",
        "Bad connection, moving to another network soon"
    ], size=data_size),
    'Churn': np.random.choice([0, 1], p=[0.7, 0.3], size=data_size) # 0 = Stayed, 1 = Left
}

# Convert dictionary records into a structured data frame matrix
df = pd.DataFrame(raw_data)
print(f"Dataset successfully created with {df.shape[0]} rows and {df.shape[1]} features.")
print(df.head(3))

# 2. Feature Engineering and NLP Vectorization
# In this step we transform text data into math vectors using an NLP technique called TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n--- Starting Checkpoint 3: Feature Engineering & NLP Vectorization ---")

# Initialize the NLP tool to keep the top 5 most meaningful words
tfidf = TfidfVectorizer(max_features=5, stop_words='english')

# Convert customer text reviews into numerical scoring arrays
feedback_features = tfidf.fit_transform(df['CustomerFeedback']).toarray()

# Convert the math array into a clean DataFrame structure
feedback_df = pd.DataFrame(feedback_features, columns=[f"nlp_{word}" for word in tfidf.get_feature_names_out()])

# Combine the numerical customer data with the new text feature data
X_features = pd.concat([df[['TenureMonths', 'MonthlyCharges']], feedback_df], axis=1)
y_target = df['Churn']

print("Engineered Feature Matrix (X) Sample with NLP columns:")
print(X_features.head(3))

# 3. Model Training and Metric Evaluation
# In this step we split data, normalize scales, train a Random Forest model, and print performance metrics.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n--- Starting Checkpoint 4: ML Training & Assessment ---")

# 1. Split data into training data (75%) and evaluation testing data (25%)
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42, stratify=y_target)

# 2. Standardize data scales so charges and months balance out mathematically
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the ensemble classification model
classifier = RandomForestClassifier(n_estimators=50, random_state=42)
classifier.fit(X_train_scaled, y_train)

# 4. Generate predictions and evaluate accuracy
predictions = classifier.predict(X_test_scaled)
print(f"Final Model Prediction Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Performance Matrix Summary:")
print(classification_report(y_test, predictions))

