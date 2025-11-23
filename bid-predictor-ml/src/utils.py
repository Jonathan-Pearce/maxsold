# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data
# Combine title and description into a single feature
data['text'] = data['title'] + ' ' + data['description']

# Define features and target variable
X = data[['text', 'viewed']]
y = data['current_bid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train['text'])
X_test_tfidf = tfidf.transform(X_test['text'])

# Combine TF-IDF features with the 'viewed' feature
X_train_final = np.hstack((X_train_tfidf.toarray(), X_train['viewed'].values.reshape(-1, 1)))
X_test_final = np.hstack((X_test_tfidf.toarray(), X_test['viewed'].values.reshape(-1, 1)))

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_final, y_train)

# Make predictions
y_pred = model.predict(X_test_final)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')