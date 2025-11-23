### Step 1: Data Preparation

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Display the first few rows
print(data.head())
```

### Step 2: Preprocessing and Feature Engineering

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Fill missing values if any
data.fillna('', inplace=True)

# Split the data into features and target variable
X = data[['title', 'description', 'viewed']]
y = data['current_bid']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_title = tfidf_vectorizer.fit_transform(X_train['title'])
X_train_description = tfidf_vectorizer.fit_transform(X_train['description'])

X_test_title = tfidf_vectorizer.transform(X_test['title'])
X_test_description = tfidf_vectorizer.transform(X_test['description'])

# Combine the features
import scipy.sparse as sp

X_train_combined = sp.hstack([X_train_title, X_train_description, X_train['viewed'].values.reshape(-1, 1)])
X_test_combined = sp.hstack([X_test_title, X_test_description, X_test['viewed'].values.reshape(-1, 1)])
```

### Step 3: Model Selection

We'll use a simple regression model, such as `RandomForestRegressor`.

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
```

### Step 4: Training the Model

```python
# Train the model
model.fit(X_train_combined, y_train)
```

### Step 5: Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test_combined)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```

### Step 6: Prediction

You can now use the trained model to make predictions on new data.

```python
# Example of making predictions on new data
new_data = pd.DataFrame({
    'title': ['Example title'],
    'description': ['Example description'],
    'viewed': [100]
})

# Preprocess the new data
new_data_title = tfidf_vectorizer.transform(new_data['title'])
new_data_description = tfidf_vectorizer.transform(new_data['description'])
new_data_combined = sp.hstack([new_data_title, new_data_description, new_data['viewed'].values.reshape(-1, 1)])

# Predict current_bid
predicted_bid = model.predict(new_data_combined)
print(f'Predicted Current Bid: {predicted_bid[0]}')
```

### Notes:
- Ensure you have the necessary libraries installed (`pandas`, `scikit-learn`, `nltk`, etc.).
- You may want to tune the model parameters and preprocess the text data further (e.g., removing stop words, stemming, etc.) for better performance.
- Depending on your dataset size and complexity, you might consider using more advanced models or techniques like deep learning.