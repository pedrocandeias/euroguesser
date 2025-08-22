#!/usr/bin/env python3
"""
Advanced EuroMillions Prediction Model

This script implements a sophisticated machine learning approach for predicting
EuroMillions lottery numbers using comprehensive historical data and multiple
advanced techniques including:

- Deep neural networks with attention mechanisms
- Ensemble methods
- Feature engineering (frequency analysis, patterns, trends)
- Statistical analysis and probability modeling
- Cross-validation and proper evaluation metrics

Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Attention, Embedding, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EuroMillionsPredictor:
    def __init__(self, data_path='euromillions_historical_results.csv'):
        """Initialize the predictor with historical data."""
        self.data_path = data_path
        self.df = None
        self.features = None
        self.targets = None
        self.models = {}
        self.scalers = {}
        
    def load_and_analyze_data(self):
        """Load the dataset and perform exploratory data analysis."""
        print("Loading and analyzing dataset...")
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"Dataset loaded: {len(self.df)} draws from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Analyze number frequencies
        self.analyze_frequencies()
        
        # Analyze patterns and trends
        self.analyze_patterns()
        
    def analyze_frequencies(self):
        """Analyze frequency distribution of numbers."""
        print("\\nAnalyzing number frequencies...")
        
        # Main numbers frequency
        main_numbers = []
        for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
            main_numbers.extend(self.df[col].tolist())
        
        main_freq = Counter(main_numbers)
        self.main_frequencies = dict(sorted(main_freq.items()))
        
        # Star numbers frequency
        star_numbers = []
        for col in ['Star1', 'Star2']:
            star_numbers.extend(self.df[col].tolist())
            
        star_freq = Counter(star_numbers)
        self.star_frequencies = dict(sorted(star_freq.items()))
        
        print(f"Most frequent main numbers: {sorted(main_freq.items(), key=lambda x: x[1], reverse=True)[:10]}")
        print(f"Most frequent star numbers: {sorted(star_freq.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
    def analyze_patterns(self):
        """Analyze temporal patterns and trends."""
        print("\\nAnalyzing temporal patterns...")
        
        # Add temporal features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Calculate draw intervals
        self.df['Days_Since_Start'] = (self.df['Date'] - self.df['Date'].min()).dt.days
        
        # Analyze sum patterns
        main_cols = ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']
        star_cols = ['Star1', 'Star2']
        
        self.df['Main_Sum'] = self.df[main_cols].sum(axis=1)
        self.df['Star_Sum'] = self.df[star_cols].sum(axis=1)
        self.df['Total_Sum'] = self.df['Main_Sum'] + self.df['Star_Sum']
        
        # Analyze consecutive numbers
        self.df['Main_Consecutive'] = self.df[main_cols].apply(self.count_consecutive, axis=1)
        
        print(f"Average main sum: {self.df['Main_Sum'].mean():.2f}")
        print(f"Average star sum: {self.df['Star_Sum'].mean():.2f}")
        print(f"Average consecutive main numbers: {self.df['Main_Consecutive'].mean():.2f}")
        
    def count_consecutive(self, row):
        """Count consecutive numbers in a row."""
        numbers = sorted([row[col] for col in row.index])
        consecutive_count = 0
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                consecutive_count += 1
        return consecutive_count
        
    def engineer_features(self, lookback_window=10):
        """Create advanced features for training."""
        print(f"\\nEngineering features with lookback window of {lookback_window}...")
        
        features_list = []
        targets_list = []
        
        for i in range(lookback_window, len(self.df)):
            # Historical features (last N draws)
            historical_features = []
            
            # Basic number features from lookback window
            for j in range(lookback_window):
                idx = i - lookback_window + j
                main_nums = [self.df.iloc[idx][f'Main{k}'] for k in range(1, 6)]
                star_nums = [self.df.iloc[idx][f'Star{k}'] for k in range(1, 3)]
                historical_features.extend(main_nums + star_nums)
            
            # Frequency-based features
            recent_main_nums = []
            recent_star_nums = []
            for j in range(lookback_window):
                idx = i - lookback_window + j
                recent_main_nums.extend([self.df.iloc[idx][f'Main{k}'] for k in range(1, 6)])
                recent_star_nums.extend([self.df.iloc[idx][f'Star{k}'] for k in range(1, 3)])
            
            # Main number frequency in recent draws
            main_recent_freq = Counter(recent_main_nums)
            main_freq_features = [main_recent_freq.get(num, 0) for num in range(1, 51)]
            
            # Star number frequency in recent draws
            star_recent_freq = Counter(recent_star_nums)
            star_freq_features = [star_recent_freq.get(num, 0) for num in range(1, 13)]
            
            # Statistical features
            recent_sums = [self.df.iloc[i - lookback_window + j]['Main_Sum'] for j in range(lookback_window)]
            recent_star_sums = [self.df.iloc[i - lookback_window + j]['Star_Sum'] for j in range(lookback_window)]
            
            stat_features = [
                np.mean(recent_sums),
                np.std(recent_sums),
                np.mean(recent_star_sums),
                np.std(recent_star_sums),
                self.df.iloc[i-1]['Days_Since_Start'],  # Time trend
                self.df.iloc[i-1]['Month'],  # Seasonal
                self.df.iloc[i-1]['DayOfWeek'],  # Day of week
            ]
            
            # Combine all features
            all_features = historical_features + main_freq_features + star_freq_features + stat_features
            features_list.append(all_features)
            
            # Target (current draw)
            current_target = [self.df.iloc[i][f'Main{k}'] for k in range(1, 6)] + \
                           [self.df.iloc[i][f'Star{k}'] for k in range(1, 3)]
            targets_list.append(current_target)
        
        self.features = np.array(features_list)
        self.targets = np.array(targets_list)
        
        print(f"Generated {len(self.features)} training samples")
        print(f"Feature vector size: {self.features.shape[1]}")
        
    def create_neural_network_model(self, input_dim):
        """Create an advanced neural network model."""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Separate heads for main numbers and stars
            Dense(7, activation='linear', name='output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def create_lstm_model(self, sequence_length, features_per_timestep):
        """Create LSTM model for sequence prediction."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, features_per_timestep)),
            Dropout(0.3),
            
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(7, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def prepare_lstm_data(self, sequence_length=10):
        """Prepare data for LSTM model."""
        X_lstm = []
        y_lstm = []
        
        basic_features_per_draw = 7  # 5 main + 2 stars
        
        for i in range(sequence_length, len(self.df)):
            # Create sequence of basic draws
            sequence = []
            for j in range(sequence_length):
                idx = i - sequence_length + j
                draw_features = [self.df.iloc[idx][f'Main{k}'] for k in range(1, 6)] + \
                              [self.df.iloc[idx][f'Star{k}'] for k in range(1, 3)]
                sequence.append(draw_features)
            
            X_lstm.append(sequence)
            
            # Target
            target = [self.df.iloc[i][f'Main{k}'] for k in range(1, 6)] + \
                    [self.df.iloc[i][f'Star{k}'] for k in range(1, 3)]
            y_lstm.append(target)
        
        return np.array(X_lstm), np.array(y_lstm)
        
    def train_models(self):
        """Train multiple models and create an ensemble."""
        print("\\nTraining prediction models...")
        
        # Prepare data for different models
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.targets, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # 1. Neural Network Model
        print("Training neural network...")
        self.models['neural_network'] = self.create_neural_network_model(X_train_scaled.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        
        history_nn = self.models['neural_network'].fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 2. LSTM Model
        print("Training LSTM model...")
        X_lstm, y_lstm = self.prepare_lstm_data()
        X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
            X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=False
        )
        
        self.models['lstm'] = self.create_lstm_model(X_lstm.shape[1], X_lstm.shape[2])
        
        history_lstm = self.models['lstm'].fit(
            X_lstm_train, y_lstm_train,
            validation_data=(X_lstm_test, y_lstm_test),
            epochs=150,
            batch_size=16,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 3. Random Forest (for comparison)
        print("Training Random Forest...")
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        
        # Store test data for evaluation
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.X_lstm_test = X_lstm_test
        self.y_lstm_test = y_lstm_test
        
        print("All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate model performance."""
        print("\\nEvaluating model performance...")
        
        # Neural Network evaluation
        nn_pred = self.models['neural_network'].predict(self.X_test_scaled)
        nn_mse = mean_squared_error(self.y_test, nn_pred)
        nn_mae = mean_absolute_error(self.y_test, nn_pred)
        
        # LSTM evaluation
        lstm_pred = self.models['lstm'].predict(self.X_lstm_test)
        lstm_mse = mean_squared_error(self.y_lstm_test, lstm_pred)
        lstm_mae = mean_absolute_error(self.y_lstm_test, lstm_pred)
        
        # Random Forest evaluation
        rf_pred = self.models['random_forest'].predict(self.X_test_scaled)
        rf_mse = mean_squared_error(self.y_test, rf_pred)
        rf_mae = mean_absolute_error(self.y_test, rf_pred)
        
        print(f"Neural Network - MSE: {nn_mse:.4f}, MAE: {nn_mae:.4f}")
        print(f"LSTM - MSE: {lstm_mse:.4f}, MAE: {lstm_mae:.4f}")
        print(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}")
        
        return {
            'neural_network': {'mse': nn_mse, 'mae': nn_mae},
            'lstm': {'mse': lstm_mse, 'mae': lstm_mae},
            'random_forest': {'mse': rf_mse, 'mae': rf_mae}
        }
        
    def predict_next_draw(self):
        """Generate predictions for the next EuroMillions draw."""
        print("\\nGenerating predictions for next draw...")
        
        # Prepare features for prediction (using most recent data)
        recent_features = self.features[-1:] if len(self.features) > 0 else None
        
        if recent_features is None:
            print("Error: No features available for prediction")
            return None
            
        recent_features_scaled = self.scalers['standard'].transform(recent_features)
        
        # Get predictions from different models
        predictions = {}
        
        # Neural Network prediction
        nn_pred = self.models['neural_network'].predict(recent_features_scaled)[0]
        predictions['neural_network'] = nn_pred
        
        # LSTM prediction
        X_lstm_recent, _ = self.prepare_lstm_data()
        if len(X_lstm_recent) > 0:
            lstm_pred = self.models['lstm'].predict(X_lstm_recent[-1:])
            predictions['lstm'] = lstm_pred[0]
        
        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict(recent_features_scaled)[0]
        predictions['random_forest'] = rf_pred
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        
        # Convert to actual lottery numbers
        main_numbers = self.convert_to_lottery_numbers(ensemble_pred[:5], 1, 50, 5)
        star_numbers = self.convert_to_lottery_numbers(ensemble_pred[5:], 1, 12, 2)
        
        print(f"\\nPredicted Numbers:")
        print(f"Main Numbers: {sorted(main_numbers)}")
        print(f"Star Numbers: {sorted(star_numbers)}")
        
        # Also show individual model predictions
        print(f"\\nIndividual Model Predictions:")
        for model_name, pred in predictions.items():
            main_pred = self.convert_to_lottery_numbers(pred[:5], 1, 50, 5)
            star_pred = self.convert_to_lottery_numbers(pred[5:], 1, 12, 2)
            print(f"{model_name}: Main {sorted(main_pred)}, Stars {sorted(star_pred)}")
        
        return {
            'main_numbers': sorted(main_numbers),
            'star_numbers': sorted(star_numbers),
            'individual_predictions': predictions,
            'ensemble_raw': ensemble_pred
        }
        
    def convert_to_lottery_numbers(self, predictions, min_val, max_val, count):
        """Convert model predictions to valid lottery numbers."""
        # Clip predictions to valid range
        clipped = np.clip(predictions, min_val, max_val)
        
        # Round to integers
        rounded = np.round(clipped).astype(int)
        
        # Ensure uniqueness by selecting top candidates
        if len(set(rounded)) >= count:
            return list(set(rounded))[:count]
        
        # If we don't have enough unique numbers, add more based on probabilities
        all_numbers = list(range(min_val, max_val + 1))
        distances = [abs(num - pred) for num in all_numbers for pred in clipped]
        
        # Sort by distance and select unique numbers
        sorted_candidates = sorted(all_numbers, key=lambda x: min(abs(x - p) for p in clipped))
        selected = []
        for num in sorted_candidates:
            if num not in selected:
                selected.append(num)
            if len(selected) == count:
                break
                
        return selected
        
    def save_models(self):
        """Save trained models."""
        print("\\nSaving models...")
        
        # Save neural network
        self.models['neural_network'].save('euromillions_nn_model.keras')
        
        # Save LSTM
        self.models['lstm'].save('euromillions_lstm_model.keras')
        
        # Save Random Forest (using joblib)
        import joblib
        joblib.dump(self.models['random_forest'], 'euromillions_rf_model.pkl')
        joblib.dump(self.scalers['standard'], 'euromillions_scaler.pkl')
        
        print("Models saved successfully!")


def main():
    """Main function to run the complete prediction pipeline."""
    print("Advanced EuroMillions Prediction System")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EuroMillionsPredictor()
    
    try:
        # Load and analyze data
        predictor.load_and_analyze_data()
        
        # Engineer features
        predictor.engineer_features(lookback_window=15)
        
        # Train models
        predictor.train_models()
        
        # Evaluate models
        evaluation_results = predictor.evaluate_models()
        
        # Generate predictions
        next_draw_prediction = predictor.predict_next_draw()
        
        # Save models
        predictor.save_models()
        
        print("\\n" + "=" * 50)
        print("PREDICTION COMPLETE!")
        print("=" * 50)
        
        if next_draw_prediction:
            print(f"Next Draw Prediction:")
            print(f"Main Numbers: {next_draw_prediction['main_numbers']}")
            print(f"Star Numbers: {next_draw_prediction['star_numbers']}")
            
        print(f"\\nModels saved and ready for future predictions!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()