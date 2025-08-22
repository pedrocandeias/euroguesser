#!/usr/bin/env python3
"""
Improved EuroMillions Prediction System with Multiple Strategies
Addresses the issue of identical predictions through ensemble methods and randomization
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
from collections import Counter
import random
warnings.filterwarnings('ignore')

class ImprovedEuroMillionsPredictor:
    def __init__(self, data_file='euromillions_historical_results.csv'):
        self.data_file = data_file
        self.models = {}  # Multiple models for ensemble
        self.scaler_main = StandardScaler()
        self.scaler_stars = StandardScaler()
        self.history = None
        self.df = None
        
        # EuroMillions constraints
        self.main_min, self.main_max = 1, 50
        self.star_min, self.star_max = 1, 12
        
        # Load data once
        self.load_data()
        
    def load_data(self):
        """Load the historical EuroMillions data"""
        print("Loading historical data...")
        self.df = pd.read_csv(self.data_file)
        print(f"Loaded {len(self.df)} historical draws from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
    def create_diverse_models(self):
        """Create multiple models with different architectures for ensemble"""
        
        # Model 1: LSTM-based (original approach)
        model1 = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(10, 7), dropout=0.3),
            layers.LSTM(32, dropout=0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(16, activation='relu'),
            layers.Dense(7, activation='sigmoid')
        ], name='lstm_model')
        
        # Model 2: CNN-based for pattern recognition
        model2 = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(10, 7)),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(25, activation='relu'),
            layers.Dense(7, activation='sigmoid')
        ], name='cnn_model')
        
        # Model 3: Dense network with different architecture
        model3 = keras.Sequential([
            layers.Flatten(input_shape=(10, 7)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(7, activation='sigmoid')
        ], name='dense_model')
        
        # Compile all models
        for model in [model1, model2, model3]:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        self.models = {
            'lstm': model1,
            'cnn': model2,
            'dense': model3
        }
        
        return self.models
    
    def prepare_training_data(self):
        """Prepare data for training with different preprocessing"""
        main_numbers = self.df[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].values
        star_numbers = self.df[['Star1', 'Star2']].values
        
        # Normalize the data
        main_normalized = self.scaler_main.fit_transform(main_numbers)
        stars_normalized = self.scaler_stars.fit_transform(star_numbers)
        
        # Combine features
        features = np.concatenate([main_normalized, stars_normalized], axis=1)
        
        # Create sequences
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
    
    def train_ensemble(self, X, y, epochs=50, batch_size=16):
        """Train all models in the ensemble"""
        print("Training ensemble models...")
        
        histories = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            
            # Use different train/validation splits for diversity
            val_split = 0.15 + (0.1 * np.random.random())  # Random validation split
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callbacks,
                verbose=1
            )
            
            histories[name] = history
        
        return histories
    
    def statistical_prediction(self, num_predictions=1):
        """Generate predictions using statistical analysis"""
        predictions = []
        
        for _ in range(num_predictions):
            # Frequency-based prediction with randomness
            main_freq = Counter()
            star_freq = Counter()
            
            # Weight recent draws more heavily
            weights = np.exp(np.linspace(-2, 0, len(self.df)))
            
            for i, (_, row) in enumerate(self.df.iterrows()):
                weight = weights[i]
                for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                    main_freq[row[col]] += weight
                for col in ['Star1', 'Star2']:
                    star_freq[row[col]] += weight
            
            # Add randomness to frequency selection
            main_candidates = []
            for num, freq in main_freq.most_common(15):  # Top 15 most frequent
                main_candidates.extend([num] * int(freq * 10))
            
            star_candidates = []
            for num, freq in star_freq.most_common(8):  # Top 8 most frequent
                star_candidates.extend([num] * int(freq * 10))
            
            # Select with randomness
            main_numbers = []
            while len(main_numbers) < 5:
                num = random.choice(main_candidates)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            star_numbers = []
            while len(star_numbers) < 2:
                num = random.choice(star_candidates)
                if num not in star_numbers:
                    star_numbers.append(num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers)
            })
        
        return predictions
    
    def gap_analysis_prediction(self, num_predictions=1):
        """Generate predictions based on gap analysis"""
        predictions = []
        
        for _ in range(num_predictions):
            # Calculate gaps for each number
            main_gaps = {i: [] for i in range(1, 51)}
            star_gaps = {i: [] for i in range(1, 13)}
            
            # Calculate gaps since last appearance
            for num in range(1, 51):
                last_seen = -1
                for i, row in self.df.iterrows():
                    if num in [row['Main1'], row['Main2'], row['Main3'], row['Main4'], row['Main5']]:
                        if last_seen >= 0:
                            main_gaps[num].append(i - last_seen)
                        last_seen = i
                # Current gap
                if last_seen >= 0:
                    main_gaps[num].append(len(self.df) - last_seen)
            
            for num in range(1, 13):
                last_seen = -1
                for i, row in self.df.iterrows():
                    if num in [row['Star1'], row['Star2']]:
                        if last_seen >= 0:
                            star_gaps[num].append(i - last_seen)
                        last_seen = i
                if last_seen >= 0:
                    star_gaps[num].append(len(self.df) - last_seen)
            
            # Select numbers with longer current gaps (due for appearance)
            main_candidates = []
            for num, gaps in main_gaps.items():
                if gaps:
                    avg_gap = np.mean(gaps[:-1]) if len(gaps) > 1 else gaps[0]
                    current_gap = gaps[-1]
                    if current_gap >= avg_gap * 0.8:  # Due or overdue
                        main_candidates.append((num, current_gap))
            
            star_candidates = []
            for num, gaps in star_gaps.items():
                if gaps:
                    avg_gap = np.mean(gaps[:-1]) if len(gaps) > 1 else gaps[0]
                    current_gap = gaps[-1]
                    if current_gap >= avg_gap * 0.8:
                        star_candidates.append((num, current_gap))
            
            # Sort by gap and add randomness
            main_candidates.sort(key=lambda x: x[1], reverse=True)
            star_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select with weighted randomness
            main_numbers = []
            for i in range(min(10, len(main_candidates))):
                if len(main_numbers) < 5:
                    if random.random() < 0.7:  # 70% chance to pick from top candidates
                        main_numbers.append(main_candidates[i][0])
                    else:
                        # Random selection
                        remaining = [x[0] for x in main_candidates if x[0] not in main_numbers]
                        if remaining:
                            main_numbers.append(random.choice(remaining))
            
            # Fill remaining with random
            while len(main_numbers) < 5:
                num = random.randint(1, 50)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            star_numbers = []
            for i in range(min(4, len(star_candidates))):
                if len(star_numbers) < 2:
                    if random.random() < 0.7:
                        star_numbers.append(star_candidates[i][0])
                    else:
                        remaining = [x[0] for x in star_candidates if x[0] not in star_numbers]
                        if remaining:
                            star_numbers.append(random.choice(remaining))
            
            while len(star_numbers) < 2:
                num = random.randint(1, 12)
                if num not in star_numbers:
                    star_numbers.append(num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers)
            })
        
        return predictions
    
    def ensemble_neural_prediction(self, recent_data, num_predictions=1, temperature=1.0):
        """Generate predictions using ensemble of neural networks with temperature sampling"""
        if not self.models:
            raise ValueError("Models not trained yet!")
        
        predictions = []
        
        for _ in range(num_predictions):
            # Get predictions from all models
            model_predictions = []
            
            for name, model in self.models.items():
                pred = model.predict(recent_data.reshape(1, 10, 7), verbose=0)
                # Apply temperature for diversity
                pred = pred[0] ** (1.0 / temperature)
                model_predictions.append(pred)
            
            # Ensemble: weighted average with noise
            weights = np.random.dirichlet([1, 1, 1])  # Random weights for diversity
            ensemble_pred = np.average(model_predictions, axis=0, weights=weights)
            
            # Add noise for diversity
            noise = np.random.normal(0, 0.05, ensemble_pred.shape)
            ensemble_pred += noise
            ensemble_pred = np.clip(ensemble_pred, 0, 1)
            
            # Convert to actual numbers
            result = self.denormalize_prediction(ensemble_pred)
            predictions.append(result)
        
        return predictions
    
    def denormalize_prediction(self, pred):
        """Convert normalized prediction back to lottery numbers"""
        # Denormalize
        main_denorm = self.scaler_main.inverse_transform(pred[:5].reshape(1, -1))[0]
        stars_denorm = self.scaler_stars.inverse_transform(pred[5:].reshape(1, -1))[0]
        
        # Add randomness and ensure valid ranges
        main_numbers = []
        star_numbers = []
        
        # Convert with randomness
        for val in main_denorm:
            # Add some randomness around the predicted value
            noise = np.random.normal(0, 2)
            num = int(np.clip(np.round(val + noise), self.main_min, self.main_max))
            if num not in main_numbers and len(main_numbers) < 5:
                main_numbers.append(num)
        
        # Fill remaining main numbers
        while len(main_numbers) < 5:
            num = random.randint(self.main_min, self.main_max)
            if num not in main_numbers:
                main_numbers.append(num)
        
        for val in stars_denorm:
            noise = np.random.normal(0, 1)
            num = int(np.clip(np.round(val + noise), self.star_min, self.star_max))
            if num not in star_numbers and len(star_numbers) < 2:
                star_numbers.append(num)
        
        while len(star_numbers) < 2:
            num = random.randint(self.star_min, self.star_max)
            if num not in star_numbers:
                star_numbers.append(num)
        
        return {
            'main_numbers': sorted(main_numbers),
            'star_numbers': sorted(star_numbers)
        }
    
    def generate_diverse_predictions(self, num_predictions=5):
        """Generate diverse predictions using multiple strategies"""
        print("\n" + "="*60)
        print("GENERATING DIVERSE PREDICTIONS")
        print("="*60)
        
        all_predictions = []
        
        # Strategy 1: Statistical frequency analysis
        print("\n1. Statistical Frequency Analysis:")
        stat_preds = self.statistical_prediction(num_predictions // 2 + 1)
        for i, pred in enumerate(stat_preds, 1):
            print(f"   Stat-{i}: {pred['main_numbers'][0]:2d} {pred['main_numbers'][1]:2d} {pred['main_numbers'][2]:2d} {pred['main_numbers'][3]:2d} {pred['main_numbers'][4]:2d} | Stars: {pred['star_numbers'][0]:2d} {pred['star_numbers'][1]:2d}")
            all_predictions.append(('Statistical', pred))
        
        # Strategy 2: Gap analysis
        print("\n2. Gap Analysis (Due Numbers):")
        gap_preds = self.gap_analysis_prediction(num_predictions // 2 + 1)
        for i, pred in enumerate(gap_preds, 1):
            print(f"   Gap-{i}:  {pred['main_numbers'][0]:2d} {pred['main_numbers'][1]:2d} {pred['main_numbers'][2]:2d} {pred['main_numbers'][3]:2d} {pred['main_numbers'][4]:2d} | Stars: {pred['star_numbers'][0]:2d} {pred['star_numbers'][1]:2d}")
            all_predictions.append(('Gap Analysis', pred))
        
        # Strategy 3: Neural ensemble (if models are trained)
        if self.models:
            print("\n3. Neural Network Ensemble:")
            # Prepare recent data
            main_numbers = self.df[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].values
            star_numbers = self.df[['Star1', 'Star2']].values
            main_normalized = self.scaler_main.transform(main_numbers)
            stars_normalized = self.scaler_stars.transform(star_numbers)
            features = np.concatenate([main_normalized, stars_normalized], axis=1)
            recent_data = features[-10:]
            
            # Generate with different temperatures for diversity
            temperatures = [0.8, 1.0, 1.2]
            for temp in temperatures:
                neural_preds = self.ensemble_neural_prediction(recent_data, 1, temperature=temp)
                for i, pred in enumerate(neural_preds, 1):
                    print(f"   Neural-T{temp}: {pred['main_numbers'][0]:2d} {pred['main_numbers'][1]:2d} {pred['main_numbers'][2]:2d} {pred['main_numbers'][3]:2d} {pred['main_numbers'][4]:2d} | Stars: {pred['star_numbers'][0]:2d} {pred['star_numbers'][1]:2d}")
                    all_predictions.append((f'Neural-T{temp}', pred))
        
        return all_predictions[:num_predictions]
    
    def save_models(self, base_path='euromillions_ensemble'):
        """Save all trained models"""
        for name, model in self.models.items():
            model.save(f'{base_path}_{name}.keras')
        print(f"Models saved with prefix: {base_path}")
    
    def load_models(self, base_path='euromillions_ensemble'):
        """Load pre-trained models"""
        model_names = ['lstm', 'cnn', 'dense']
        loaded_models = {}
        
        for name in model_names:
            filepath = f'{base_path}_{name}.keras'
            if os.path.exists(filepath):
                loaded_models[name] = keras.models.load_model(filepath)
                print(f"Loaded {name} model from {filepath}")
            else:
                print(f"Model file {filepath} not found")
        
        self.models = loaded_models
        return len(loaded_models) > 0


def main():
    """Main function to run the improved EuroMillions predictor"""
    print("Improved EuroMillions Prediction System")
    print("="*50)
    
    # Initialize predictor
    predictor = ImprovedEuroMillionsPredictor()
    
    try:
        # Try to load existing models first
        if not predictor.load_models():
            print("No pre-trained models found. Training new ensemble...")
            
            # Create and train models
            predictor.create_diverse_models()
            X, y, features = predictor.prepare_training_data()
            
            print(f"Training data shape: {X.shape} -> {y.shape}")
            histories = predictor.train_ensemble(X, y, epochs=30, batch_size=16)
            
            # Save trained models
            predictor.save_models()
        else:
            print("Using pre-trained models.")
            # Still need to prepare data for scaling
            X, y, features = predictor.prepare_training_data()
        
        # Generate diverse predictions
        predictions = predictor.generate_diverse_predictions(num_predictions=10)
        
        print(f"\n" + "="*60)
        print("SUMMARY OF DIVERSE PREDICTIONS")
        print("="*60)
        
        for i, (strategy, pred) in enumerate(predictions, 1):
            main_nums = pred['main_numbers']
            star_nums = pred['star_numbers']
            print(f"{i:2d}. [{strategy:12s}] {main_nums[0]:2d} {main_nums[1]:2d} {main_nums[2]:2d} {main_nums[3]:2d} {main_nums[4]:2d} | Stars: {star_nums[0]:2d} {star_nums[1]:2d}")
        
        print(f"\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
        print("Multiple strategies used for diverse predictions:")
        print("- Statistical frequency analysis with weighted randomness")
        print("- Gap analysis for 'due' numbers")
        print("- Neural network ensemble with temperature sampling")
        print("- Each method provides different perspectives on number selection")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()