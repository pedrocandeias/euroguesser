#!/usr/bin/env python3
"""
Advanced EuroMillions Prediction Model using TensorFlow
Uses comprehensive historical dataset with 1,869 draws (2004-2025)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EuroMillionsTensorFlowPredictor:
    def __init__(self, data_file='euromillions_historical_results.csv'):
        self.data_file = data_file
        self.model = None
        self.scaler_main = StandardScaler()
        self.scaler_stars = StandardScaler()
        self.history = None
        
        # EuroMillions constraints
        self.main_min, self.main_max = 1, 50
        self.star_min, self.star_max = 1, 12
        
    def load_and_preprocess_data(self):
        """Load and preprocess the historical EuroMillions data"""
        print("Loading historical data...")
        
        # Load the comprehensive dataset
        df = pd.read_csv(self.data_file)
        print(f"Loaded {len(df)} historical draws from {df['Date'].min()} to {df['Date'].max()}")
        
        # Extract features (main numbers and stars)
        main_numbers = df[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].values
        star_numbers = df[['Star1', 'Star2']].values
        
        # Normalize the data
        main_normalized = self.scaler_main.fit_transform(main_numbers)
        stars_normalized = self.scaler_stars.fit_transform(star_numbers)
        
        # Combine features
        features = np.concatenate([main_normalized, stars_normalized], axis=1)
        
        # Create sequences for time-series prediction
        sequence_length = 10
        X, y = self.create_sequences(features, sequence_length)
        
        return X, y, features
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for time-series prediction"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def create_model(self, input_shape):
        """Create advanced TensorFlow model architecture"""
        print("Creating TensorFlow model...")
        
        model = keras.Sequential([
            # LSTM layers for sequence learning
            layers.LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.2),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, dropout=0.2),
            
            # Dense layers for pattern recognition
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            
            # Output layer (7 values: 5 main + 2 stars)
            layers.Dense(7, activation='sigmoid')
        ])
        
        # Compile with custom metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the TensorFlow model"""
        print("Training model...")
        
        # Create model
        self.model = self.create_model((X.shape[1], X.shape[2]))
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict_next_draw(self, recent_data, num_predictions=1):
        """Generate predictions for next draw(s)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = []
        current_sequence = recent_data[-10:].copy()  # Use last 10 draws
        
        for _ in range(num_predictions):
            # Predict next values
            pred = self.model.predict(current_sequence.reshape(1, 10, 7), verbose=0)
            predictions.append(pred[0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred[0]
        
        return np.array(predictions)
    
    def denormalize_predictions(self, predictions):
        """Convert normalized predictions back to lottery numbers"""
        results = []
        
        for pred in predictions:
            # Denormalize main numbers and stars
            main_denorm = self.scaler_main.inverse_transform(pred[:5].reshape(1, -1))[0]
            stars_denorm = self.scaler_stars.inverse_transform(pred[5:].reshape(1, -1))[0]
            
            # Round and constrain to valid ranges
            main_numbers = np.clip(np.round(main_denorm), self.main_min, self.main_max).astype(int)
            star_numbers = np.clip(np.round(stars_denorm), self.star_min, self.star_max).astype(int)
            
            # Ensure unique main numbers
            main_numbers = self.ensure_unique_numbers(main_numbers, self.main_min, self.main_max, 5)
            star_numbers = self.ensure_unique_numbers(star_numbers, self.star_min, self.star_max, 2)
            
            results.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers)
            })
        
        return results
    
    def ensure_unique_numbers(self, numbers, min_val, max_val, count):
        """Ensure we have unique numbers within the valid range"""
        unique_nums = list(set(numbers))
        
        # If we don't have enough unique numbers, generate random ones
        while len(unique_nums) < count:
            new_num = np.random.randint(min_val, max_val + 1)
            if new_num not in unique_nums:
                unique_nums.append(new_num)
        
        return np.array(unique_nums[:count])
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='euromillions_tensorflow_model.keras'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='euromillions_tensorflow_model.keras'):
        """Load a pre-trained model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False
    
    def generate_statistical_insights(self, features):
        """Generate statistical insights from the data"""
        print("\n" + "="*60)
        print("STATISTICAL INSIGHTS FROM HISTORICAL DATA")
        print("="*60)
        
        # Load original data for analysis
        df = pd.read_csv(self.data_file)
        
        # Main numbers analysis
        main_cols = ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']
        all_main = df[main_cols].values.flatten()
        
        print(f"\nMain Numbers (1-50) Statistics:")
        print(f"Most frequent: {np.bincount(all_main)[1:51].argmax() + 1} (appeared {np.bincount(all_main)[1:51].max()} times)")
        print(f"Least frequent: {np.bincount(all_main)[1:51].argmin() + 1} (appeared {np.bincount(all_main)[1:51].min()} times)")
        print(f"Average: {np.mean(all_main):.1f}")
        
        # Stars analysis
        star_cols = ['Star1', 'Star2']
        all_stars = df[star_cols].values.flatten()
        
        print(f"\nStar Numbers (1-12) Statistics:")
        print(f"Most frequent: {np.bincount(all_stars)[1:13].argmax() + 1} (appeared {np.bincount(all_stars)[1:13].max()} times)")
        print(f"Least frequent: {np.bincount(all_stars)[1:13].argmin() + 1} (appeared {np.bincount(all_stars)[1:13].min()} times)")
        print(f"Average: {np.mean(all_stars):.1f}")


def main():
    """Main function to run the EuroMillions TensorFlow predictor"""
    print("EuroMillions TensorFlow Prediction System")
    print("="*50)
    
    # Initialize predictor
    predictor = EuroMillionsTensorFlowPredictor()
    
    try:
        # Load and preprocess data
        X, y, features = predictor.load_and_preprocess_data()
        print(f"Data shape: {X.shape} -> {y.shape}")
        
        # Generate statistical insights
        predictor.generate_statistical_insights(features)
        
        # Train model
        print("\n" + "="*50)
        print("TRAINING TENSORFLOW MODEL")
        print("="*50)
        
        history = predictor.train_model(X, y, epochs=150, batch_size=16)
        
        # Save model
        predictor.save_model()
        
        # Plot training history
        try:
            predictor.plot_training_history()
        except:
            print("Could not display training plots (matplotlib may not be available)")
        
        # Generate predictions
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        # Use the most recent data for prediction
        recent_data = features[-10:]
        predictions = predictor.predict_next_draw(recent_data, num_predictions=3)
        results = predictor.denormalize_predictions(predictions)
        
        print(f"\nNext 3 EuroMillions Predictions:")
        print("-" * 40)
        
        for i, result in enumerate(results, 1):
            main_nums = result['main_numbers']
            star_nums = result['star_numbers']
            print(f"Prediction {i}: {main_nums[0]:2d} {main_nums[1]:2d} {main_nums[2]:2d} {main_nums[3]:2d} {main_nums[4]:2d} | Stars: {star_nums[0]:2d} {star_nums[1]:2d}")
        
        print(f"\nModel performance:")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        print("\n" + "="*50)
        print("PREDICTION COMPLETE")
        print("="*50)
        print("Note: These predictions are based on statistical patterns")
        print("and should be used for entertainment purposes only.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()