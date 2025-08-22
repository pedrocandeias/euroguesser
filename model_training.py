import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(filepath):
    # Load and preprocess data from filepath
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    # Normalize main numbers and stars
    data[:, :5] = data[:, :5] / 50.0  # Assuming main numbers are in columns 0-4
    data[:, 5:] = data[:, 5:] / 12.0  # Assuming star numbers are in columns 5-6
    return data

def create_model(input_shape=(7,)):
    # Define and compile a simple Sequential model
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(7, activation='sigmoid')  # Output layer; adjust according to your dataset
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def train_model(model, data):
    # Split the data into features and potentially targets, assuming an autoencoder-like setup for now
    # Here, using the same data as both input and output because it's not clear what's being predicted
    x_train = data[:, :]  # Using all columns as features for training
    y_train = data[:, :]  # Assuming you're trying to reconstruct the input, hence using it as target as well

    # Splitting the dataset into training (80%) and validation (20%)
    split_idx = int(0.8 * len(x_train))
    x_train, x_val = x_train[:split_idx], x_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=2)
    return history

def main():
    filepath = 'scraped_euromillions_training.txt'  # Using the comprehensive scraped dataset
    print(f"Loading training data from {filepath}...")
    data = load_data(filepath)
    print(f"Loaded {len(data)} training samples")
    
    model = create_model(input_shape=(7,))
    print("Starting model training...")
    history = train_model(model, data)
    
    # Save the trained model in the recommended SavedModel format
    model.save('lottery_model.keras')
    print("Model training complete and saved as 'lottery_model.keras'")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()