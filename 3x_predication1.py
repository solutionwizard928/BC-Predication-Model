import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
filename = "basic_data.csv"  # Replace with your CSV file name
data = pd.read_csv(filename, header=None, names=["numbers"])

# Step 2: Create batches of size 300
batch_size = 300
numbers = data["numbers"].values
batches = [
    numbers[i:i + batch_size] for i in range(0, len(numbers), batch_size)
    if len(numbers[i:i + batch_size]) == batch_size
]

# Step 3: Feature engineering
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

# Generate feature list
feature_list = [create_features(batch) for batch in batches]
df_features = pd.DataFrame(feature_list)

# Step 4: Prepare data for training
X = df_features.drop("target_position", axis=1)  # Inputs (features)
y = df_features["target_position"]              # Output (target positions)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained on {len(batches)} batches of size {batch_size}")
print(f"Mean Squared Error (MSE): {mse}")

# Step 8: Predict using 300 random input numbers
current_data_path = "300_current_data.csv"
current_batch = pd.read_csv(current_data_path).iloc[:, 0].values

print("Current Batch of 300 Random Numbers:")
print(current_batch)  # Print the 300 numbers to test

# Extract features from the batch
current_features = create_features(current_batch)

# Remove target_position feature (target) from the features during prediction
current_features.pop("target_position", None)  # Remove 'target_position' if it exists

# Convert to DataFrame (exclude target_position during prediction)
current_features_df = pd.DataFrame([current_features])

# Check the feature names of current_features_df
print(f"\nFeatures used for prediction: {current_features_df.columns.tolist()}")

# Predict the maximum position of 3.23 in the current batch
predicted_position = model.predict(current_features_df)  # Get prediction

# Safely access the predicted position
predicted_position_value = predicted_position[0] if predicted_position.size > 0 else None
print(f"\nPredicted maximum range (position) for 3.23: {predicted_position_value}")

