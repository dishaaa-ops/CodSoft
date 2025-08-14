from sklearn.metrics import r2_score, mean_squared_error
import joblib
import pandas as pd

def evaluate_model():
    # Load the model and data
    model = joblib.load('models/movie_rating_model.pkl')
    data = pd.read_csv('data/preprocessed_train.csv')
    
    X = data.drop(columns=['Rating'])
    y = data['Rating']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    print(f"R^2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    evaluate_model()
