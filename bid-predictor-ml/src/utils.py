# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "/workspaces/maxsold/data/auction/items_all_2025-11-21.csv"


def load_and_prepare(path: str = DATA_PATH):
    df = pd.read_csv(path)
    # Ensure text fields are strings (no NaN for TF-IDF)
    df['title'] = df.get('title', '').fillna('').astype(str)
    df['description'] = df.get('description', '').fillna('').astype(str)
    df['text'] = (df['title'] + ' ' + df['description']).str.strip()

    # Numeric feature
    df['viewed'] = pd.to_numeric(df.get('viewed', 0), errors='coerce').fillna(0)

    # Target
    df['current_bid'] = pd.to_numeric(df.get('current_bid', np.nan), errors='coerce')

    # Drop rows without a target
    df = df.dropna(subset=['current_bid']).reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame, tfidf_max_features: int = 5000, test_size: float = 0.2, random_state: int = 42):
    X = df[['text', 'viewed']].copy()
    y = df['current_bid'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # TF-IDF on text
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
    X_train_text = tfidf.fit_transform(X_train['text'].astype(str))
    X_test_text = tfidf.transform(X_test['text'].astype(str))

    # Numeric features -> keep sparse
    X_train_num = csr_matrix(X_train['viewed'].astype(float).values.reshape(-1, 1))
    X_test_num = csr_matrix(X_test['viewed'].astype(float).values.reshape(-1, 1))

    # Combine sparse text + numeric
    X_train_final = hstack([X_train_text, X_train_num], format='csr')
    X_test_final = hstack([X_test_text, X_test_num], format='csr')

    # Model (use Ridge for some regularization)
    model = Ridge(alpha=1.0, random_state=random_state)
    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_test_final)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "tfidf": tfidf,
        "mse": mse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def main():
    df = load_and_prepare()
    if df.empty:
        print("No rows with valid current_bid found in dataset.")
        return

    results = train_and_evaluate(df)
    print("Rows used:", len(df))
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"R^2 Score: {results['r2']:.4f}")


if __name__ == "__main__":
    main()