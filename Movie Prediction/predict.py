import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

def preprocess_data(data, columns):
    # Convert categorical columns to dummy variables
    data = pd.get_dummies(data)
    
    # Add missing columns with default value of 0
    missing_cols = set(columns) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # Ensure the same column order as during training
    data = data[columns]
    
    return data

def predict_new_movie(new_data):
    # Load the pre-trained model and columns
    model = joblib.load('models/movie_rating_model.pkl')
    columns = joblib.load('models/columns.pkl')
    
    # Preprocess the new data
    new_data = preprocess_data(new_data, columns)
    
    # Predict
    predicted_rating = model.predict(new_data)
    return predicted_rating

if __name__ == "__main__":
    # Example new movie data
    new_movie_data = pd.DataFrame({
        'Duration': ['101 min'],
        'Name': ['Example Movie'],
        'Year': [2024]
    })
    
    # Predict the rating
    rating = predict_new_movie(new_movie_data)
    print("Predicted Rating:", rating)
