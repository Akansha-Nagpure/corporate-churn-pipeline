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
