import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def preprocess_data(data, columns=None):
    # Convert categorical columns to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    if columns is not None:
        # Ensure the data has the same columns as the training data
        for col in columns:
            if col not in data.columns:
                data[col] = 0  # Add missing columns with 0 values
        data = data[columns]  # Ensure column order is the same
    return data

def train_model():
    data = pd.read_csv('data/preprocessed_train.csv')  # Use the preprocessed data
    
    # Handle missing values in the target column 'Rating'
    data = data.dropna(subset=['Rating'])  # Remove rows where 'Rating' is NaN
    
    # Preprocess the data (convert categorical to numeric)
    data = preprocess_data(data)
    
    # Features and target
    X = data.drop(columns=['Rating'])
    y = data['Rating']
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation Mean Squared Error: {mse}")
    
    # Save the model and column names
    joblib.dump(model, 'models/movie_rating_model.pkl')
    joblib.dump(X_train.columns, 'models/columns.pkl')


if __name__ == "__main__":
    train_model()
