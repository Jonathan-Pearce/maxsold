### Step 1: Data Preparation

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())
```

### Step 2: Feature Engineering

We will convert the `title` and `description` into numerical features using TF-IDF vectorization.

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine title and description for feature extraction
data['text'] = data['title'] + ' ' + data['description']

# Split the data into features and target variable
X = data[['text', 'viewed']]
y = data['current_bid']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train['text'])
X_test_tfidf = tfidf.transform(X_test['text'])
```

### Step 3: Model Selection

We can use a regression model since `current_bid` is a continuous variable. Here, we will use a Random Forest Regressor.

```python
from sklearn.ensemble import RandomForestRegressor

# Combine TF-IDF features with the 'viewed' feature
import scipy.sparse as sp

X_train_final = sp.hstack((X_train_tfidf, X_train[['viewed']].values))
X_test_final = sp.hstack((X_test_tfidf, X_test[['viewed']].values))

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_final, y_train)
```

### Step 4: Evaluation

We will evaluate the model using Mean Absolute Error (MAE) and R-squared metrics.

```python
from sklearn.metrics import mean_absolute_error, r2_score

# Make predictions
y_pred = model.predict(X_test_final)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
```

### Step 5: Prediction

Now you can use the trained model to make predictions on new data.

```python
# Example of making a prediction
new_data = pd.DataFrame({
    'title': ['Example Title'],
    'description': ['Example description of the item.'],
    'viewed': [100]
})

# Preprocess the new data
new_data['text'] = new_data['title'] + ' ' + new_data['description']
new_data_tfidf = tfidf.transform(new_data['text'])

# Combine TF-IDF features with the 'viewed' feature
new_data_final = sp.hstack((new_data_tfidf, new_data[['viewed']].values))

# Make a prediction
predicted_bid = model.predict(new_data_final)
print(f'Predicted Current Bid: {predicted_bid[0]}')
```

### Notes:
- Make sure to install the required libraries if you haven't already:

```bash
pip install pandas scikit-learn nltk
```

- Adjust the `max_features` in `TfidfVectorizer` based on your dataset size and complexity.
- You can experiment with different models and hyperparameters to improve performance.
- Ensure that your dataset is clean and preprocessed (e.g., handling missing values, removing duplicates) before training the model.