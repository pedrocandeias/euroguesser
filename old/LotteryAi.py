import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    # Normalize data
    data[:, :5] = data[:, :5] / 50.0  # Normalizing main numbers
    data[:, 5:] = data[:, 5:] / 12.0  # Normalizing stars
    return data

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(7,)),
        Dense(64, activation='relu'),
        Dense(7, activation='sigmoid')  # Assuming you want the same structure output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def train_model(model, data):
    # For this example, not explicitly splitting into x and y, as your data is all features
    # If you were predicting the next draw, you'd need a different approach to set up your training data
    split_idx = int(len(data) * 0.8)
    x_train, x_val = data[:split_idx], data[split_idx:]
    
    history = model.fit(x_train, x_train, epochs=100, validation_data=(x_val, x_val), verbose=2)
    return history

def main():
    filepath = 'euro_millions_entries_training.txt'  # Update this path to your dataset file
    data = load_data(filepath)
    model = create_model()
    history = train_model(model, data)
    print("Model training complete.")

if __name__ == "__main__":
    main()
