import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Step 1: Process "basic_data.csv" without loading it entirely into memory
def process_basic_data():
    basic_data_file = "basic_data.csv"
    processed_data = []

    # Open and read the "basic_data.csv" file
    with open(basic_data_file, "r") as file:
        lines = file.readlines()  # Read all lines from the file

    for line in lines:
        # Split the line on whitespace, then further split on commas
        values = []
        for item in line.split():
            values.extend(item.split(','))  # Further split items by commas

        try:
            # Convert all extracted values to floats
            values = list(map(float, values))
        except ValueError as e:
            # Log and skip lines that cannot be processed
            print(f"Error processing line: {line.strip()}")
            print(f"Error details: {e}")
            continue

        # Append the processed list of float values to `processed_data`
        processed_data.append(values)

    print("Processing of basic_data.csv complete.")
    # Return or further process `processed_data` if needed
    return processed_data

# Function to create features for a single batch
def create_features(batch, target_value=3.23):
    """Create features from a batch of numbers."""
    features = {
        "mean": np.mean(batch),
        "variance": np.var(batch),
        "min": np.min(batch),
        "max": np.max(batch),
        "frequency_near_target": sum(3.00 <= x <= 3.50 for x in batch),
        "target_position": np.argmax(batch == target_value) if target_value in batch else -1,
    }
    return features

# Function to prepare data for training
def prepare_data(batches):
    print("=====  Prepare Data  =======")
    """Generate features and split data for training from batches."""
    feature_list = [create_features(batch) for batch in batches]
    df_features = pd.DataFrame(feature_list)
    X = df_features.drop("target_position", axis=1)  # Inputs (features)
    y = df_features["target_position"]              # Output (target positions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train the model
def train_model(X_train, y_train, model_file):
    print("=====  Train Modal  =======")
    print(X_train, y_train)
    """Train a Random Forest model and save it to a file."""
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    dump(model, model_file)  # Save the model to a file
    print(f"Model saved to {model_file}")
    return model

# Function to load the model
def load_model(model_file):
    """Load a trained model from a file."""
    return load(model_file)

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Function to predict using a single batch
def predict_current_batch(model, current_batch):
    """Predict the target position for a current batch."""
    current_features = create_features(current_batch)
    current_features_df = pd.DataFrame([current_features]).drop("target_position", axis=1)
    predicted_position = model.predict(current_features_df)
    return predicted_position[0] if predicted_position.size > 0 else None

# Main function to orchestrate the workflow
def main():
    # File paths
    source_file = "origin.csv"  # Replace with your source CSV file name
    basic_data_file = "basic_data.csv"
    model_file = "trained_model.joblib"  # File to save/load the model
    
    try:
        # Step 1: Process the generated "basic_data.csv"
        print("Starting processing of 'basic_data.csv'...")
        processed_data = process_basic_data()
        print(f"Processing completed. Total processed entries: {len(processed_data)}")
        # print(processed_data)
        # Optional: Display a sample of the processed data
        if processed_data:
            print("Sample of processed data (first 3 entries):")
            for i, sample in enumerate(processed_data[:3], 1):
                print(f"Entry {i}: {sample}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

    # Step 2: Prepare data for training
    X_train, X_test, y_train, y_test = prepare_data(processed_data)

    # Step 3: Train or load the model
    try:
        model = load_model(model_file)
        print(f"Loaded model from {model_file}")
    except FileNotFoundError:
        model = train_model(X_train, y_train, model_file)

    # Step 4: Evaluate the model
    # mse = evaluate_model(model, X_test, y_test)
    # print(f"Mean Squared Error (MSE): {mse}")

    # Step 5: Predict using 300 random input numbers
    # current_data_path = "300_current_data.csv"
    # current_batch = pd.read_csv(current_data_path).iloc[:, 0].values
    # print("\nCurrent Batch of 300 Random Numbers:")
    # print(current_batch)  # Print the 300 numbers to test

    # Step 6: Predict using the model
    # predicted_position = predict_current_batch(model, current_batch)
    # print(f"\nPredicted maximum range (position) for 3.23: {predicted_position}")

# Entry point of the script
if __name__ == "__main__":
    main()
