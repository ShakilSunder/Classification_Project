import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Create the directory if it does not exist
os.makedirs('model', exist_ok=True)

# Load the dataset with error handling
try:
    df = pd.read_csv('po_data.csv', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")

# Print the column names to check if 'Message' exists
print("Columns in DataFrame:", df.columns)

# Check for unique values in the 'Category' column
print("Unique values in 'Category' column before mapping:", df['Category'].unique())

# Strip leading and trailing whitespace from 'Category' column
df['Category'] = df['Category'].str.strip()

# Check again for unique values in the 'Category' column after stripping whitespace
print("Unique values in 'Category' column after stripping whitespace:", df['Category'].unique())

# Convert the Category to binary (PO: 1, NO: 0)
df['Category'] = df['Category'].map({'PO': 1, 'NO': 0})

# Check for any unmapped values
if df['Category'].isnull().any():
    print("Unmapped values found in 'Category' column:", df[df['Category'].isnull()])

# Ensure 'Category' is treated as a categorical variable
df['Category'] = df['Category'].astype('category')

# Fill NaN values in the 'Message' column with an empty string
df['Message'] = df['Message'].fillna('')

# TF-IDF vectorization for the 'Message' column
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message'])

# Save the feature names
feature_names = vectorizer.get_feature_names_out()
joblib.dump(feature_names, 'model/feature_names.joblib')

# Extract the target variable
y = df['Category'].cat.codes  # Convert categories to numeric codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],  # Removed 'auto' as it is invalid
    'max_depth': [10, 20, 30, None],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(best_rf_model, 'model/random_forest_model.joblib')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully!")
