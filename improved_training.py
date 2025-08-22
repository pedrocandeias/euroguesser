#!/usr/bin/env python3
"""
Improved EuroMillions Training Script

This script uses the comprehensive historical dataset to train an improved
version of your existing model architecture with better preprocessing
and evaluation metrics.

Compatible with your existing prediction.py workflow.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_comprehensive_data(csv_path='euromillions_historical_results.csv'):
    """Load and preprocess the comprehensive historical dataset."""
    print(f"Loading comprehensive dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} draws from {df['Date'].min()} to {df['Date'].max()}")
    
    # Extract numbers into array format
    data = []
    for _, row in df.iterrows():
        numbers = [
            row['Main1'], row['Main2'], row['Main3'], row['Main4'], row['Main5'],
            row['Star1'], row['Star2']
        ]
        data.append(numbers)
    
    return np.array(data, dtype=np.float32)


def normalize_data(data):
    """Normalize the lottery data."""
    normalized_data = data.copy()
    
    # Normalize main numbers (1-50 -> 0-1)
    normalized_data[:, :5] = normalized_data[:, :5] / 50.0
    
    # Normalize star numbers (1-12 -> 0-1)  
    normalized_data[:, 5:] = normalized_data[:, 5:] / 12.0
    
    return normalized_data


def create_sequences(data, sequence_length=10):
    """Create sequences for time series prediction."""
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # Use last sequence_length draws as features
        X.append(data[i-sequence_length:i].flatten())
        # Predict next draw
        y.append(data[i])
    
    return np.array(X), np.array(y)


def create_improved_model(input_shape):
    """Create an improved model architecture."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(7, activation='sigmoid')  # Output layer for 7 numbers
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def analyze_predictions(y_true, y_pred):
    """Analyze prediction accuracy."""
    # Denormalize for analysis
    y_true_denorm = y_true.copy()
    y_pred_denorm = y_pred.copy()
    
    y_true_denorm[:, :5] *= 50.0
    y_true_denorm[:, 5:] *= 12.0
    y_pred_denorm[:, :5] *= 50.0
    y_pred_denorm[:, 5:] *= 12.0
    
    # Round predictions to integers
    y_pred_rounded = np.round(y_pred_denorm)
    
    # Calculate exact matches
    exact_matches = 0
    partial_matches = []
    
    for i in range(len(y_true_denorm)):
        true_main = set(y_true_denorm[i, :5])
        pred_main = set(y_pred_rounded[i, :5])
        main_overlap = len(true_main.intersection(pred_main))
        
        true_stars = set(y_true_denorm[i, 5:])
        pred_stars = set(y_pred_rounded[i, 5:])
        star_overlap = len(true_stars.intersection(pred_stars))
        
        if main_overlap == 5 and star_overlap == 2:
            exact_matches += 1
            
        partial_matches.append((main_overlap, star_overlap))
    
    print(f"\\nPrediction Analysis:")
    print(f"Exact matches (5+2): {exact_matches}/{len(y_true_denorm)} ({exact_matches/len(y_true_denorm)*100:.2f}%)")
    
    # Count partial matches
    for main_count in range(6):
        for star_count in range(3):
            matches = sum(1 for m, s in partial_matches if m == main_count and s == star_count)
            if matches > 0:
                print(f"Matches with {main_count} main + {star_count} stars: {matches} ({matches/len(partial_matches)*100:.1f}%)")


def train_improved_model():
    """Train the improved model with comprehensive data."""
    print("EuroMillions Improved Training System")
    print("=" * 40)
    
    # Load data
    data = load_comprehensive_data()
    print(f"Raw data shape: {data.shape}")
    
    # Normalize data
    normalized_data = normalize_data(data)
    print("Data normalized successfully")
    
    # Create sequences
    sequence_length = 15  # Use last 15 draws to predict next
    X, y = create_sequences(normalized_data, sequence_length)
    print(f"Created {len(X)} training sequences")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Split data (use time-based split to avoid data leakage)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create and train model
    model = create_improved_model(X.shape[1])
    print("\\nModel architecture:")
    model.summary()
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    print("\\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\\nEvaluating model...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Detailed prediction analysis
    y_pred = model.predict(X_test, verbose=0)
    analyze_predictions(y_test, y_pred)
    
    # Save the improved model
    model.save('improved_lottery_model.keras')
    print("\\nModel saved as 'improved_lottery_model.keras'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    
    return model, history


def generate_prediction(model_path='improved_lottery_model.keras', data_path='euromillions_historical_results.csv'):
    """Generate a prediction using the trained model."""
    print("\\nGenerating prediction for next draw...")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and prepare recent data
    data = load_comprehensive_data(data_path)
    normalized_data = normalize_data(data)
    
    # Use last sequence for prediction
    sequence_length = 15
    if len(normalized_data) >= sequence_length:
        last_sequence = normalized_data[-sequence_length:].flatten()
        last_sequence = last_sequence.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(last_sequence, verbose=0)[0]
        
        # Denormalize
        main_pred = prediction[:5] * 50.0
        star_pred = prediction[5:] * 12.0
        
        # Convert to valid lottery numbers
        main_numbers = []
        star_numbers = []
        
        # For main numbers (1-50), select 5 unique numbers
        main_candidates = np.clip(np.round(main_pred), 1, 50).astype(int)
        all_main = list(range(1, 51))
        all_main.sort(key=lambda x: abs(x - main_candidates[0]))  # Sort by closeness to first prediction
        
        for num in all_main:
            if num not in main_numbers:
                main_numbers.append(num)
            if len(main_numbers) == 5:
                break
        
        # For star numbers (1-12), select 2 unique numbers
        star_candidates = np.clip(np.round(star_pred), 1, 12).astype(int)
        all_stars = list(range(1, 13))
        all_stars.sort(key=lambda x: abs(x - star_candidates[0]))  # Sort by closeness to first prediction
        
        for num in all_stars:
            if num not in star_numbers:
                star_numbers.append(num)
            if len(star_numbers) == 2:
                break
        
        print(f"\\nPredicted EuroMillions Numbers:")
        print(f"Main Numbers: {sorted(main_numbers)}")
        print(f"Star Numbers: {sorted(star_numbers)}")
        print(f"\\nRaw predictions (before conversion):")
        print(f"Main: {main_pred}")
        print(f"Stars: {star_pred}")
        
        return {
            'main_numbers': sorted(main_numbers),
            'star_numbers': sorted(star_numbers),
            'raw_prediction': prediction
        }
    else:
        print("Error: Not enough historical data for prediction")
        return None


if __name__ == "__main__":
    try:
        # Train the model
        model, history = train_improved_model()
        
        # Generate a prediction
        prediction = generate_prediction()
        
        print("\\n" + "=" * 40)
        print("TRAINING AND PREDICTION COMPLETE!")
        print("=" * 40)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()