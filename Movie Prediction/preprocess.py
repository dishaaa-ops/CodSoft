import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data():
    # Load the CSV file with proper encoding
    data = pd.read_csv('data/movies.csv', encoding='ISO-8859-1')
    return data

def preprocess_data(data):
    # Remove non-numeric characters from 'Votes' and convert to float
    data['Votes'] = data['Votes'].replace('[\$,M]', '', regex=True).astype(float).fillna(0)
    
    # Handle categorical data using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
    
    # Specify the categorical columns to encode
    categorical_columns = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    
    # Apply one-hot encoding to the categorical columns
    encoded_data = encoder.fit_transform(data[categorical_columns])
    
    # Convert the encoded data to a DataFrame and concatenate with original data
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Drop the original categorical columns and concatenate the one-hot encoded columns
    data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)
    
    return data

if __name__ == '__main__':
    data = load_data()
    processed_data = preprocess_data(data)
    print(processed_data.head())
    processed_data.to_csv('data/preprocessed_train.csv', index=False)