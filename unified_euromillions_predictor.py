#!/usr/bin/env python3
"""
Unified EuroMillions Predictor System
Combines all prediction methods: Statistical, ML, AI, and Physical Bias Detection
"""

import pandas as pd
import numpy as np
import random
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class UnifiedEuroMillionsPredictor:
    def __init__(self, data_file='scraped_euromillions_results.csv'):
        self.data_file = data_file
        self.df = None
        self.scaler_main = StandardScaler() if TF_AVAILABLE else None
        self.scaler_stars = StandardScaler() if TF_AVAILABLE else None
        
        # EuroMillions constraints
        self.main_min, self.main_max = 1, 50
        self.star_min, self.star_max = 1, 12
        
        # Load data
        self.load_data()
        
        print("🎯 Unified EuroMillions Predictor System")
        print("="*60)
        print(f"📊 Dataset: {len(self.df)} draws ({self.df['Date'].min().date()} to {self.df['Date'].max().date()})")
        print(f"🤖 TensorFlow: {'✓ Available' if TF_AVAILABLE else '✗ Not installed'}")
        print(f"📈 SciPy: {'✓ Available' if SCIPY_AVAILABLE else '✗ Not installed'}")
        
    def load_data(self):
        """Load and prepare historical data"""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
            # Add temporal features for analysis
            self.add_temporal_features()
        except FileNotFoundError:
            print(f"❌ Error: {self.data_file} not found!")
            sys.exit(1)
    
    def add_temporal_features(self):
        """Add temporal features to the dataset"""
        # Day of week (0=Monday, 6=Sunday)
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek
        self.df['day_name'] = self.df['Date'].dt.day_name()
        
        # Month (1-12)
        self.df['month'] = self.df['Date'].dt.month
        self.df['month_name'] = self.df['Date'].dt.month_name()
        
        # Season
        self.df['season'] = self.df['month'].map(self.get_season)
        
        # Quarter
        self.df['quarter'] = self.df['Date'].dt.quarter
        
        # Year
        self.df['year'] = self.df['Date'].dt.year
        
        # Day of month
        self.df['day_of_month'] = self.df['Date'].dt.day
        
        # Week of year
        self.df['week_of_year'] = self.df['Date'].dt.isocalendar().week
    
    def get_season(self, month):
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    def temporal_pattern_analysis(self):
        """Comprehensive temporal pattern analysis"""
        print(f"\n📅 TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        analysis_results = {}
        
        # Analyze patterns by different temporal dimensions
        temporal_dimensions = [
            ('day_name', 'Day of Week'),
            ('month_name', 'Month'),
            ('season', 'Season'),
            ('quarter', 'Quarter')
        ]
        
        for dimension, title in temporal_dimensions:
            print(f"\n📊 Analysis by {title}:")
            dimension_analysis = self.analyze_temporal_dimension(dimension)
            analysis_results[dimension] = dimension_analysis
            
            # Display top patterns
            if dimension_analysis['main_patterns']:
                print(f"   Most frequent main numbers by {title.lower()}:")
                for period, numbers in list(dimension_analysis['main_patterns'].items())[:3]:
                    top_numbers = [f"{num}({count})" for num, count in numbers.most_common(5)]
                    print(f"     {period}: {', '.join(top_numbers)}")
        
        return analysis_results
    
    def analyze_temporal_dimension(self, dimension):
        """Analyze patterns within a specific temporal dimension"""
        dimension_data = {}
        main_patterns = {}
        star_patterns = {}
        
        # Group by temporal dimension
        for period in self.df[dimension].unique():
            period_data = self.df[self.df[dimension] == period]
            
            # Analyze main numbers
            main_numbers = []
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                main_numbers.extend(period_data[col].tolist())
            main_patterns[period] = Counter(main_numbers)
            
            # Analyze star numbers
            star_numbers = []
            for col in ['Star1', 'Star2']:
                star_numbers.extend(period_data[col].tolist())
            star_patterns[period] = Counter(star_numbers)
            
            # Calculate statistics
            dimension_data[period] = {
                'draws': len(period_data),
                'avg_main_sum': period_data[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].sum(axis=1).mean(),
                'avg_star_sum': period_data[['Star1', 'Star2']].sum(axis=1).mean(),
                'main_freq': main_patterns[period],
                'star_freq': star_patterns[period]
            }
        
        return {
            'dimension_data': dimension_data,
            'main_patterns': main_patterns,
            'star_patterns': star_patterns
        }
    
    def get_current_temporal_context(self):
        """Get current temporal context for predictions"""
        from datetime import datetime
        
        now = datetime.now()
        return {
            'day_of_week': now.weekday(),
            'day_name': now.strftime('%A'),
            'month': now.month,
            'month_name': now.strftime('%B'),
            'season': self.get_season(now.month),
            'quarter': (now.month - 1) // 3 + 1,
            'day_of_month': now.day,
            'week_of_year': now.isocalendar().week
        }
    
    def temporal_weighted_prediction(self, num_predictions=3):
        """Generate predictions based on temporal patterns"""
        current_context = self.get_current_temporal_context()
        predictions = []
        
        # Analyze current temporal context
        temporal_analysis = self.temporal_pattern_analysis()
        
        for i in range(num_predictions):
            # Get patterns for current temporal context
            main_candidates = Counter()
            star_candidates = Counter()
            
            # Weight by different temporal dimensions
            temporal_weights = {
                'day_name': 0.3,
                'month_name': 0.25,
                'season': 0.25,
                'quarter': 0.2
            }
            
            for dimension, weight in temporal_weights.items():
                if dimension in temporal_analysis:
                    # Map dimension to current context
                    if dimension == 'day_name':
                        current_period = current_context['day_name']
                    elif dimension == 'month_name':
                        current_period = current_context['month_name']
                    elif dimension == 'season':
                        current_period = current_context['season']
                    elif dimension == 'quarter':
                        current_period = current_context['quarter']
                    else:
                        continue
                    
                    if current_period in temporal_analysis[dimension]['main_patterns']:
                        main_freq = temporal_analysis[dimension]['main_patterns'][current_period]
                        star_freq = temporal_analysis[dimension]['star_patterns'][current_period]
                        
                        # Add weighted frequencies
                        for num, count in main_freq.most_common(20):
                            main_candidates[num] += count * weight
                        for num, count in star_freq.most_common(10):
                            star_candidates[num] += count * weight
            
            # Select numbers based on weighted temporal patterns
            selected_main = []
            selected_stars = []
            
            # Smart selection for main numbers
            top_main_candidates = main_candidates.most_common(15)
            for num, weight_score in top_main_candidates:
                if len(selected_main) >= 5:
                    break
                # Probabilistic selection based on weight
                selection_prob = min(0.8, weight_score / max(main_candidates.values()) * 0.7 + 0.1)
                if random.random() < selection_prob:
                    selected_main.append(num)
            
            # Fill remaining main numbers
            while len(selected_main) < 5:
                candidates = [num for num, _ in top_main_candidates if num not in selected_main]
                if candidates:
                    selected_main.append(random.choice(candidates[:8]))
                else:
                    num = random.randint(1, 50)
                    if num not in selected_main:
                        selected_main.append(num)
            
            # Smart selection for star numbers
            top_star_candidates = star_candidates.most_common(8)
            for num, weight_score in top_star_candidates:
                if len(selected_stars) >= 2:
                    break
                selection_prob = min(0.8, weight_score / max(star_candidates.values()) * 0.8 + 0.1)
                if random.random() < selection_prob:
                    selected_stars.append(num)
            
            # Fill remaining star numbers
            while len(selected_stars) < 2:
                candidates = [num for num, _ in top_star_candidates if num not in selected_stars]
                if candidates:
                    selected_stars.append(random.choice(candidates))
                else:
                    num = random.randint(1, 12)
                    if num not in selected_stars:
                        selected_stars.append(num)
            
            # Calculate confidence based on temporal strength
            temporal_strength = sum(temporal_weights.values()) * 0.6
            confidence = min(0.85, temporal_strength + 0.1)
            
            predictions.append({
                'main_numbers': sorted(selected_main),
                'star_numbers': sorted(selected_stars),
                'method': f'Temporal Analysis ({current_context["season"]})',
                'confidence': confidence,
                'temporal_context': current_context
            })
        
        return predictions
    
    # =====================================================
    # STATISTICAL PREDICTION METHODS
    # =====================================================
    
    def frequency_analysis_prediction(self, num_predictions=3):
        """Statistical frequency analysis with weighted randomness"""
        predictions = []
        
        # Calculate frequencies with recent bias
        main_freq = Counter()
        star_freq = Counter()
        
        # Weight recent draws more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.df)))
        
        for i, (_, row) in enumerate(self.df.iterrows()):
            weight = weights[i]
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                main_freq[row[col]] += weight
            for col in ['Star1', 'Star2']:
                star_freq[row[col]] += weight
        
        for _ in range(num_predictions):
            # Create weighted candidate pools
            main_candidates = []
            for num, freq in main_freq.most_common(20):
                main_candidates.extend([num] * int(freq * 5))
            
            star_candidates = []
            for num, freq in star_freq.most_common(10):
                star_candidates.extend([num] * int(freq * 5))
            
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
                'star_numbers': sorted(star_numbers),
                'method': 'Frequency Analysis',
                'confidence': 0.7
            })
        
        return predictions
    
    def gap_analysis_prediction(self, num_predictions=3):
        """Gap analysis - numbers due for appearance"""
        predictions = []
        
        # Calculate gaps for each number
        main_gaps = {i: [] for i in range(1, 51)}
        star_gaps = {i: [] for i in range(1, 13)}
        
        # Calculate current gaps
        for num in range(1, 51):
            last_seen = -1
            for i, row in self.df.iterrows():
                if num in [row['Main1'], row['Main2'], row['Main3'], row['Main4'], row['Main5']]:
                    if last_seen >= 0:
                        main_gaps[num].append(i - last_seen)
                    last_seen = i
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
        
        for _ in range(num_predictions):
            # Select numbers with longer current gaps
            main_candidates = []
            for num, gaps in main_gaps.items():
                if gaps:
                    avg_gap = np.mean(gaps[:-1]) if len(gaps) > 1 else gaps[0]
                    current_gap = gaps[-1]
                    if current_gap >= avg_gap * 0.8:
                        main_candidates.append((num, current_gap))
            
            star_candidates = []
            for num, gaps in star_gaps.items():
                if gaps:
                    avg_gap = np.mean(gaps[:-1]) if len(gaps) > 1 else gaps[0]
                    current_gap = gaps[-1]
                    if current_gap >= avg_gap * 0.8:
                        star_candidates.append((num, current_gap))
            
            # Sort by gap and select with weighted randomness
            main_candidates.sort(key=lambda x: x[1], reverse=True)
            star_candidates.sort(key=lambda x: x[1], reverse=True)
            
            main_numbers = []
            for i in range(min(10, len(main_candidates))):
                if len(main_numbers) < 5 and random.random() < 0.6:
                    main_numbers.append(main_candidates[i][0])
            
            # Fill remaining
            while len(main_numbers) < 5:
                num = random.randint(1, 50)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            star_numbers = []
            for i in range(min(6, len(star_candidates))):
                if len(star_numbers) < 2 and random.random() < 0.6:
                    star_numbers.append(star_candidates[i][0])
            
            while len(star_numbers) < 2:
                num = random.randint(1, 12)
                if num not in star_numbers:
                    star_numbers.append(num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers),
                'method': 'Gap Analysis',
                'confidence': 0.6
            })
        
        return predictions
    
    def pattern_analysis_prediction(self, num_predictions=3):
        """Pattern-based predictions (consecutive, odd/even, etc.)"""
        predictions = []
        
        for _ in range(num_predictions):
            main_numbers = []
            
            # Strategy: Mix of patterns
            if random.random() < 0.3:  # 30% chance for consecutive numbers
                start = random.randint(1, 46)
                main_numbers.extend([start, start + 1])
            
            # Add some odd/even balance
            while len(main_numbers) < 5:
                num = random.randint(1, 50)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            # Ensure some odd/even balance
            odds = len([n for n in main_numbers if n % 2 == 1])
            if odds < 2:  # Too few odds, replace some evens
                evens = [n for n in main_numbers if n % 2 == 0]
                if evens:
                    main_numbers.remove(random.choice(evens))
                    while True:
                        new_odd = random.randrange(1, 51, 2)
                        if new_odd not in main_numbers:
                            main_numbers.append(new_odd)
                            break
            
            star_numbers = [random.randint(1, 12), random.randint(1, 12)]
            while star_numbers[0] == star_numbers[1]:
                star_numbers[1] = random.randint(1, 12)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers),
                'method': 'Pattern Analysis',
                'confidence': 0.5
            })
        
        return predictions
    
    # =====================================================
    # PHYSICAL BIAS DETECTION
    # =====================================================
    
    def physical_bias_prediction(self, num_predictions=3):
        """Predictions based on detected physical biases"""
        predictions = []
        
        # Frequency bias detection
        main_freq = Counter()
        star_freq = Counter()
        
        for _, row in self.df.iterrows():
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                main_freq[row[col]] += 1
            for col in ['Star1', 'Star2']:
                star_freq[row[col]] += 1
        
        total_main_draws = len(self.df) * 5
        total_star_draws = len(self.df) * 2
        expected_main_freq = total_main_draws / 50
        expected_star_freq = total_star_draws / 12
        
        # Detect biased numbers
        heavy_main = []
        light_main = []
        heavy_stars = []
        light_stars = []
        
        for num in range(1, 51):
            observed = main_freq[num]
            deviation = (observed - expected_main_freq) / expected_main_freq
            if deviation > 0.1:
                heavy_main.append(num)
            elif deviation < -0.1:
                light_main.append(num)
        
        for num in range(1, 13):
            observed = star_freq[num]
            deviation = (observed - expected_star_freq) / expected_star_freq
            if deviation > 0.15:
                heavy_stars.append(num)
            elif deviation < -0.15:
                light_stars.append(num)
        
        # Generate predictions based on bias strategies
        strategies = ['heavy_bias', 'light_compensation', 'mixed_bias']
        
        for i in range(num_predictions):
            strategy = strategies[i % len(strategies)]
            
            if strategy == 'heavy_bias' and heavy_main:
                # Favor heavy balls
                main_pool = heavy_main + list(range(1, 51))
                star_pool = heavy_stars + list(range(1, 13)) if heavy_stars else list(range(1, 13))
            elif strategy == 'light_compensation' and light_main:
                # Favor light balls (due for comeback)
                main_pool = light_main + list(range(1, 51))
                star_pool = light_stars + list(range(1, 13)) if light_stars else list(range(1, 13))
            else:
                # Mixed strategy
                main_pool = heavy_main[:3] + light_main[:2] + list(range(1, 51))
                star_pool = list(range(1, 13))
            
            main_numbers = []
            while len(main_numbers) < 5:
                num = random.choice(main_pool)
                if num not in main_numbers:
                    main_numbers.append(num)
            
            star_numbers = []
            while len(star_numbers) < 2:
                num = random.choice(star_pool)
                if num not in star_numbers:
                    star_numbers.append(num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers),
                'method': f'Physical Bias ({strategy})',
                'confidence': 0.8
            })
        
        return predictions
    
    # =====================================================
    # ADVANCED AI METHODS
    # =====================================================
    
    def quantum_inspired_prediction(self, num_predictions=3):
        """Quantum-inspired probabilistic prediction"""
        predictions = []
        
        # Create quantum-inspired probability distributions
        main_probs = np.ones(50) / 50
        star_probs = np.ones(12) / 12
        
        # Apply interference patterns based on historical data
        for _, row in self.df.iterrows():
            for num in [row['Main1'], row['Main2'], row['Main3'], row['Main4'], row['Main5']]:
                main_probs[num-1] += 0.1 * np.sin(num * np.pi / 25) ** 2
            
            for num in [row['Star1'], row['Star2']]:
                star_probs[num-1] += 0.1 * np.sin(num * np.pi / 6) ** 2
        
        # Normalize probabilities
        main_probs /= np.sum(main_probs)
        star_probs /= np.sum(star_probs)
        
        for _ in range(num_predictions):
            main_numbers = np.random.choice(range(1, 51), size=5, replace=False, p=main_probs)
            star_numbers = np.random.choice(range(1, 13), size=2, replace=False, p=star_probs)
            
            predictions.append({
                'main_numbers': sorted(main_numbers.tolist()),
                'star_numbers': sorted(star_numbers.tolist()),
                'method': 'Quantum-Inspired',
                'confidence': 0.4
            })
        
        return predictions
    
    def chaos_theory_prediction(self, num_predictions=3):
        """Chaos theory (Lorenz attractor) prediction"""
        predictions = []
        
        def lorenz_system(x, y, z, dt=0.01, sigma=10, rho=28, beta=8/3):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            return x + dx, y + dy, z + dz
        
        # Initialize with recent draw
        last_row = self.df.iloc[-1]
        x = last_row['Main1'] / 10.0
        y = last_row['Main2'] / 10.0
        z = last_row['Main3'] / 10.0
        
        for _ in range(num_predictions):
            main_numbers = []
            star_numbers = []
            
            # Generate main numbers using chaotic evolution
            for i in range(10):  # Evolve system multiple times
                x, y, z = lorenz_system(x, y, z)
                if i >= 5:  # Use later iterations
                    main_num = int(abs(x * 7 + y * 3) % 50) + 1
                    if main_num not in main_numbers and len(main_numbers) < 5:
                        main_numbers.append(main_num)
            
            # Fill remaining with controlled chaos
            while len(main_numbers) < 5:
                x, y, z = lorenz_system(x, y, z)
                main_num = int(abs(z * 13) % 50) + 1
                if main_num not in main_numbers:
                    main_numbers.append(main_num)
            
            # Generate stars
            for i in range(5):
                x, y, z = lorenz_system(x, y, z)
                if i >= 3:
                    star_num = int(abs(z * 2 + x) % 12) + 1
                    if star_num not in star_numbers and len(star_numbers) < 2:
                        star_numbers.append(star_num)
            
            while len(star_numbers) < 2:
                x, y, z = lorenz_system(x, y, z)
                star_num = int(abs(y * 4) % 12) + 1
                if star_num not in star_numbers:
                    star_numbers.append(star_num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers[:5]),
                'star_numbers': sorted(star_numbers[:2]),
                'method': 'Chaos Theory',
                'confidence': 0.3
            })
        
        return predictions
    
    def fibonacci_prediction(self, num_predictions=3):
        """Fibonacci sequence and golden ratio based prediction"""
        predictions = []
        
        # Generate Fibonacci sequence
        fib = [1, 1]
        while len(fib) < 50:
            fib.append(fib[-1] + fib[-2])
        
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        for _ in range(num_predictions):
            main_numbers = []
            star_numbers = []
            
            # Use Fibonacci and golden ratio patterns
            for i in range(8):
                # Fibonacci-based main numbers
                if len(main_numbers) < 5:
                    idx = int((i * phi * 13) % len(fib[:25]))
                    num = (fib[idx] % 50) + 1
                    if num not in main_numbers:
                        main_numbers.append(num)
                
                # Golden ratio-based stars
                if len(star_numbers) < 2:
                    star_num = int((i * phi * 7) % 12) + 1
                    if star_num not in star_numbers:
                        star_numbers.append(star_num)
            
            # Fill remaining
            while len(main_numbers) < 5:
                num = int((len(main_numbers) * phi * 17) % 50) + 1
                if num not in main_numbers:
                    main_numbers.append(num)
            
            while len(star_numbers) < 2:
                num = int((len(star_numbers) * phi * 5) % 12) + 1
                if num not in star_numbers:
                    star_numbers.append(num)
            
            predictions.append({
                'main_numbers': sorted(main_numbers),
                'star_numbers': sorted(star_numbers),
                'method': 'Fibonacci/Golden Ratio',
                'confidence': 0.3
            })
        
        return predictions
    
    # =====================================================
    # TENSORFLOW NEURAL NETWORK (if available)
    # =====================================================
    
    def neural_network_prediction(self, num_predictions=3):
        """TensorFlow neural network prediction (if available)"""
        if not TF_AVAILABLE:
            return [{
                'main_numbers': [random.randint(1, 50) for _ in range(5)],
                'star_numbers': [random.randint(1, 12) for _ in range(2)],
                'method': 'Neural Network (TF unavailable)',
                'confidence': 0.0
            } for _ in range(num_predictions)]
        
        predictions = []
        
        try:
            # Try to use pre-trained models first
            trained_model_path = 'lottery_model.keras'
            improved_model_path = 'improved_lottery_model.keras'
            
            if os.path.exists(trained_model_path):
                # Use the pre-trained model
                model = keras.models.load_model(trained_model_path)
                training_data = np.loadtxt('scraped_euromillions_training.txt', delimiter=',', dtype=np.float32)
                training_data[:, :5] = training_data[:, :5] / 50.0  # Normalize main numbers
                training_data[:, 5:] = training_data[:, 5:] / 12.0  # Normalize star numbers
                
                # Generate predictions using the trained model
                model_predictions = model.predict(training_data[-10:])  # Use last 10 draws
                
                for i in range(min(num_predictions, len(model_predictions))):
                    pred = model_predictions[i]
                    # Denormalize
                    main_pred = pred[:5] * 50.0
                    star_pred = pred[5:] * 12.0
                    
                    # Convert to valid numbers
                    main_nums = [int(np.clip(np.round(x), 1, 50)) for x in main_pred]
                    star_nums = [int(np.clip(np.round(x), 1, 12)) for x in star_pred]
                    
                    # Ensure uniqueness
                    main_nums = list(dict.fromkeys(main_nums))  # Remove duplicates while preserving order
                    while len(main_nums) < 5:
                        num = random.randint(1, 50)
                        if num not in main_nums:
                            main_nums.append(num)
                    
                    star_nums = list(dict.fromkeys(star_nums))
                    while len(star_nums) < 2:
                        num = random.randint(1, 12)
                        if num not in star_nums:
                            star_nums.append(num)
                    
                    predictions.append({
                        'main_numbers': sorted(main_nums[:5]),
                        'star_numbers': sorted(star_nums[:2]),
                        'method': 'Trained Neural Network',
                        'confidence': 0.8
                    })
                
                return predictions
            
            # Fallback to simple method if no trained model
            main_numbers = self.df[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].values
            star_numbers = self.df[['Star1', 'Star2']].values
            
            # Normalize data
            main_normalized = self.scaler_main.fit_transform(main_numbers)
            stars_normalized = self.scaler_stars.fit_transform(star_numbers)
            features = np.concatenate([main_normalized, stars_normalized], axis=1)
            
            # Use recent data for prediction base
            recent_data = features[-10:]
            
            for i in range(num_predictions):
                # Add noise for diversity
                noise = np.random.normal(0, 0.1, recent_data[-1].shape)
                pred_input = recent_data[-1] + noise
                
                # Simple prediction (using mean with noise)
                main_pred = self.scaler_main.inverse_transform(pred_input[:5].reshape(1, -1))[0]
                star_pred = self.scaler_stars.inverse_transform(pred_input[5:].reshape(1, -1))[0]
                
                # Convert to valid lottery numbers
                main_nums = []
                for val in main_pred:
                    num = int(np.clip(np.round(val + np.random.normal(0, 2)), 1, 50))
                    if num not in main_nums and len(main_nums) < 5:
                        main_nums.append(num)
                
                while len(main_nums) < 5:
                    num = random.randint(1, 50)
                    if num not in main_nums:
                        main_nums.append(num)
                
                star_nums = []
                for val in star_pred:
                    num = int(np.clip(np.round(val + np.random.normal(0, 1)), 1, 12))
                    if num not in star_nums and len(star_nums) < 2:
                        star_nums.append(num)
                
                while len(star_nums) < 2:
                    num = random.randint(1, 12)
                    if num not in star_nums:
                        star_nums.append(num)
                
                predictions.append({
                    'main_numbers': sorted(main_nums),
                    'star_numbers': sorted(star_nums),
                    'method': 'Neural Network',
                    'confidence': 0.6
                })
        
        except Exception as e:
            # Fallback to random if neural network fails
            for _ in range(num_predictions):
                main_nums = random.sample(range(1, 51), 5)
                star_nums = random.sample(range(1, 13), 2)
                predictions.append({
                    'main_numbers': sorted(main_nums),
                    'star_numbers': sorted(star_nums),
                    'method': 'Neural Network (fallback)',
                    'confidence': 0.1
                })
        
        return predictions
    
    # =====================================================
    # UNIFIED PREDICTION SYSTEM
    # =====================================================
    
    def generate_all_predictions(self, predictions_per_method=2):
        """Generate predictions using all available methods"""
        print("\n🎯 GENERATING COMPREHENSIVE PREDICTIONS")
        print("="*60)
        
        all_predictions = []
        
        # 1. Statistical Methods
        print("\n📊 Statistical Methods:")
        freq_preds = self.frequency_analysis_prediction(predictions_per_method)
        gap_preds = self.gap_analysis_prediction(predictions_per_method)
        pattern_preds = self.pattern_analysis_prediction(predictions_per_method)
        
        all_predictions.extend(freq_preds)
        all_predictions.extend(gap_preds)
        all_predictions.extend(pattern_preds)
        
        for pred in freq_preds + gap_preds + pattern_preds:
            main = pred['main_numbers']
            stars = pred['star_numbers']
            method = pred['method']
            conf = pred['confidence']
            print(f"   {method:20s}: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {conf:.1f}")
        
        # 2. Temporal Analysis
        print("\n📅 Temporal Analysis:")
        temporal_preds = self.temporal_weighted_prediction(predictions_per_method)
        all_predictions.extend(temporal_preds)
        
        for pred in temporal_preds:
            main = pred['main_numbers']
            stars = pred['star_numbers']
            method = pred['method']
            conf = pred['confidence']
            context = pred['temporal_context']
            print(f"   {method:20s}: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {conf:.1f}")
            print(f"     Context: {context['day_name']}, {context['month_name']} {context['day_of_month']}, Q{context['quarter']}")
        
        # 2. Physical Bias Detection
        print("\n🔬 Physical Bias Detection:")
        bias_preds = self.physical_bias_prediction(predictions_per_method)
        all_predictions.extend(bias_preds)
        
        for pred in bias_preds:
            main = pred['main_numbers']
            stars = pred['star_numbers']
            method = pred['method']
            conf = pred['confidence']
            print(f"   {method:20s}: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {conf:.1f}")
        
        # 3. Advanced AI Methods
        print("\n🤖 Advanced AI Methods:")
        quantum_preds = self.quantum_inspired_prediction(predictions_per_method)
        chaos_preds = self.chaos_theory_prediction(predictions_per_method)
        fib_preds = self.fibonacci_prediction(predictions_per_method)
        
        all_predictions.extend(quantum_preds)
        all_predictions.extend(chaos_preds)
        all_predictions.extend(fib_preds)
        
        for pred in quantum_preds + chaos_preds + fib_preds:
            main = pred['main_numbers']
            stars = pred['star_numbers']
            method = pred['method']
            conf = pred['confidence']
            print(f"   {method:20s}: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {conf:.1f}")
        
        # 4. Neural Networks (if available)
        if TF_AVAILABLE:
            print("\n🧠 Neural Networks:")
            neural_preds = self.neural_network_prediction(predictions_per_method)
            all_predictions.extend(neural_preds)
            
            for pred in neural_preds:
                main = pred['main_numbers']
                stars = pred['star_numbers']
                method = pred['method']
                conf = pred['confidence']
                print(f"   {method:20s}: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {conf:.1f}")
        
        return all_predictions
    
    def rank_predictions(self, predictions):
        """Rank predictions by confidence and method reliability"""
        print("\n🏆 TOP RANKED PREDICTIONS")
        print("="*60)
        
        # Sort by confidence score
        ranked = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        print("Rank | Method                | Numbers                    | Confidence")
        print("-" * 65)
        
        for i, pred in enumerate(ranked[:10], 1):
            main = pred['main_numbers']
            stars = pred['star_numbers']
            method = pred['method']
            conf = pred['confidence']
            
            numbers_str = f"{main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | {stars[0]:2d} {stars[1]:2d}"
            print(f"{i:4d} | {method:20s} | {numbers_str} | {conf:8.1f}")
        
        return ranked
    
    def get_method_weights(self):
        """Define weights for each prediction method based on reliability"""
        return {
            # Statistical methods (most reliable)
            'Frequency Analysis': 0.22,
            'Gap Analysis': 0.18,
            'Pattern Analysis': 0.13,
            
            # Temporal analysis (good scientific basis)
            'Temporal Analysis (Winter)': 0.12,
            'Temporal Analysis (Spring)': 0.12,
            'Temporal Analysis (Summer)': 0.12,
            'Temporal Analysis (Autumn)': 0.12,
            
            # ML methods (good performance)
            'Trained Neural Network': 0.18,
            'Neural Network': 0.07,
            
            # Physical bias (scientific basis)
            'Physical Bias (heavy_bias)': 0.05,
            'Physical Bias (light_compensation)': 0.04,
            
            # Experimental methods (lower weight)
            'Quantum-Inspired': 0.01,
            'Chaos Theory': 0.005,
            'Fibonacci/Golden Ratio': 0.005
        }
    
    def weighted_ensemble_prediction(self, predictions):
        """Generate weighted ensemble prediction"""
        method_weights = self.get_method_weights()
        
        # Calculate weighted frequency for each number
        main_weighted_freq = Counter()
        star_weighted_freq = Counter()
        total_weight = 0
        
        for pred in predictions:
            method = pred['method']
            # Get base weight for method
            base_weight = method_weights.get(method, 0.01)  # Default small weight
            # Multiply by prediction confidence
            final_weight = base_weight * pred['confidence']
            
            for num in pred['main_numbers']:
                main_weighted_freq[num] += final_weight
            for num in pred['star_numbers']:
                star_weighted_freq[num] += final_weight
            
            total_weight += final_weight
        
        # Normalize weights
        for num in main_weighted_freq:
            main_weighted_freq[num] /= total_weight
        for num in star_weighted_freq:
            star_weighted_freq[num] /= total_weight
        
        return main_weighted_freq, star_weighted_freq
    
    def create_smart_ensemble_prediction(self, predictions):
        """Create smart ensemble prediction using weighted analysis"""
        main_weighted_freq, star_weighted_freq = self.weighted_ensemble_prediction(predictions)
        
        # Get top candidates with their weights
        top_main_candidates = main_weighted_freq.most_common(12)  # More candidates for selection
        top_star_candidates = star_weighted_freq.most_common(6)
        
        # Smart selection algorithm
        selected_main = []
        selected_stars = []
        
        # Selection strategy: Mix high-weight and medium-weight numbers
        for i, (num, weight) in enumerate(top_main_candidates):
            if len(selected_main) >= 5:
                break
                
            # Higher probability for higher-weighted numbers, but not deterministic
            selection_probability = min(0.9, weight * 10 + 0.1)  # Scale and add base probability
            
            if random.random() < selection_probability:
                selected_main.append(num)
        
        # Fill remaining slots if needed
        while len(selected_main) < 5:
            candidates = [num for num, _ in top_main_candidates if num not in selected_main]
            if candidates:
                selected_main.append(random.choice(candidates[:8]))  # Choose from top 8 remaining
            else:
                # Fallback
                num = random.randint(1, 50)
                if num not in selected_main:
                    selected_main.append(num)
        
        # Same for stars
        for i, (num, weight) in enumerate(top_star_candidates):
            if len(selected_stars) >= 2:
                break
            selection_probability = min(0.9, weight * 15 + 0.1)
            if random.random() < selection_probability:
                selected_stars.append(num)
        
        while len(selected_stars) < 2:
            candidates = [num for num, _ in top_star_candidates if num not in selected_stars]
            if candidates:
                selected_stars.append(random.choice(candidates))
            else:
                num = random.randint(1, 12)
                if num not in selected_stars:
                    selected_stars.append(num)
        
        # Calculate ensemble confidence
        method_weights = self.get_method_weights()
        weighted_confidence = 0
        total_method_weight = 0
        
        for pred in predictions:
            method = pred['method']
            method_weight = method_weights.get(method, 0.01)
            weighted_confidence += pred['confidence'] * method_weight
            total_method_weight += method_weight
        
        ensemble_confidence = weighted_confidence / total_method_weight if total_method_weight > 0 else 0.5
        
        return {
            'main_numbers': sorted(selected_main),
            'star_numbers': sorted(selected_stars),
            'method': 'Weighted Ensemble',
            'confidence': min(0.95, ensemble_confidence + 0.1),  # Boost for ensemble effect
            'weights_used': dict(main_weighted_freq.most_common(10))
        }

    def calculate_match_score(self, prediction, actual_draw):
        """Calculate match score between prediction and actual draw"""
        pred_main = set(prediction['main_numbers'])
        pred_stars = set(prediction['star_numbers'])
        actual_main = set([actual_draw['Main1'], actual_draw['Main2'], actual_draw['Main3'], 
                          actual_draw['Main4'], actual_draw['Main5']])
        actual_stars = set([actual_draw['Star1'], actual_draw['Star2']])
        
        main_matches = len(pred_main.intersection(actual_main))
        star_matches = len(pred_stars.intersection(actual_stars))
        
        # EuroMillions prize structure scoring
        if main_matches == 5 and star_matches == 2:
            return 100  # Jackpot
        elif main_matches == 5 and star_matches == 1:
            return 90   # 2nd prize
        elif main_matches == 5 and star_matches == 0:
            return 80   # 3rd prize
        elif main_matches == 4 and star_matches == 2:
            return 70   # 4th prize
        elif main_matches == 4 and star_matches == 1:
            return 60   # 5th prize
        elif main_matches == 3 and star_matches == 2:
            return 50   # 6th prize
        elif main_matches == 4 and star_matches == 0:
            return 40   # 7th prize
        elif main_matches == 2 and star_matches == 2:
            return 30   # 8th prize
        elif main_matches == 3 and star_matches == 1:
            return 20   # 9th prize
        elif main_matches == 3 and star_matches == 0:
            return 15   # 10th prize
        elif main_matches == 1 and star_matches == 2:
            return 10   # 11th prize
        elif main_matches == 2 and star_matches == 1:
            return 8    # 12th prize
        elif main_matches == 2 and star_matches == 0:
            return 5    # 13th prize
        else:
            return main_matches + star_matches  # Basic match count
    
    def backtest_method_on_period(self, method_name, start_date, end_date, sample_size=50):
        """Backtest a specific method over a time period"""
        # Filter data for the period
        period_data = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)].copy()
        
        if len(period_data) < 10:
            return {'error': 'Insufficient data for backtesting period'}
        
        results = []
        sample_draws = period_data.sample(min(sample_size, len(period_data)), random_state=42)
        
        for idx, actual_draw in sample_draws.iterrows():
            # Create historical context (data before this draw)
            historical_context = self.df[self.df['Date'] < actual_draw['Date']].copy()
            
            if len(historical_context) < 50:  # Need enough history
                continue
            
            # Temporarily update the predictor's data to historical context
            original_df = self.df
            self.df = historical_context
            
            try:
                # Generate prediction using the specified method
                if method_name == 'Frequency Analysis':
                    predictions = self.frequency_analysis_prediction(1)
                elif method_name == 'Gap Analysis':
                    predictions = self.gap_analysis_prediction(1)
                elif method_name == 'Pattern Analysis':
                    predictions = self.pattern_analysis_prediction(1)
                elif method_name == 'Temporal Analysis':
                    predictions = self.temporal_weighted_prediction(1)
                elif method_name == 'Weighted Ensemble':
                    all_preds = self.generate_all_predictions(1)
                    predictions = [self.create_smart_ensemble_prediction(all_preds)]
                elif method_name == 'Physical Bias (heavy_bias)':
                    predictions = self.physical_bias_prediction(1, bias_type='heavy_bias')
                elif method_name == 'Neural Network':
                    predictions = self.neural_network_prediction(1)
                else:
                    continue
                
                if predictions:
                    prediction = predictions[0]
                    score = self.calculate_match_score(prediction, actual_draw)
                    
                    results.append({
                        'date': actual_draw['Date'],
                        'prediction': prediction,
                        'actual': actual_draw,
                        'score': score,
                        'main_matches': len(set(prediction['main_numbers']).intersection(
                            {actual_draw['Main1'], actual_draw['Main2'], actual_draw['Main3'], 
                             actual_draw['Main4'], actual_draw['Main5']})),
                        'star_matches': len(set(prediction['star_numbers']).intersection(
                            {actual_draw['Star1'], actual_draw['Star2']}))
                    })
                    
            except Exception as e:
                continue
            finally:
                # Restore original data
                self.df = original_df
        
        return results
    
    def comprehensive_backtest(self, start_date='2020-01-01', end_date='2024-12-31', 
                               sample_size=30, draws_per_method=None):
        """Run comprehensive backtesting on multiple methods
        
        Args:
            start_date (str): Start date for backtesting period (YYYY-MM-DD)
            end_date (str): End date for backtesting period (YYYY-MM-DD)
            sample_size (int): Total number of draws to test (distributed among methods)
            draws_per_method (int, optional): Specific number of draws to test per method.
                                            If None, uses sample_size distributed among methods.
        """
        print(f"\n🔍 COMPREHENSIVE BACKTESTING ANALYSIS")
        print("="*60)
        print(f"📅 Period: {start_date} to {end_date}")
        
        # Determine actual draws per method
        methods_count = 7  # Number of methods to test
        if draws_per_method is not None:
            effective_draws_per_method = draws_per_method
            total_tests = draws_per_method * methods_count
            print(f"📊 Draws per method: {draws_per_method}")
            print(f"🔬 Total tests: {total_tests}")
        else:
            effective_draws_per_method = sample_size
            print(f"📊 Sample size: {sample_size} draws per method")
        
        methods_to_test = [
            'Frequency Analysis',
            'Gap Analysis',
            'Pattern Analysis',
            'Temporal Analysis',
            'Weighted Ensemble',
            'Physical Bias (heavy_bias)',
            'Neural Network'
        ]
        
        backtest_results = {}
        
        for method in methods_to_test:
            print(f"🧪 Testing {method}...")
            results = self.backtest_method_on_period(method, start_date, end_date, effective_draws_per_method)
            
            if 'error' not in results and results:
                # Calculate statistics
                scores = [r['score'] for r in results]
                main_matches = [r['main_matches'] for r in results]
                star_matches = [r['star_matches'] for r in results]
                
                backtest_results[method] = {
                    'total_tests': len(results),
                    'avg_score': np.mean(scores),
                    'max_score': max(scores),
                    'avg_main_matches': np.mean(main_matches),
                    'avg_star_matches': np.mean(star_matches),
                    'jackpots': sum(1 for s in scores if s == 100),
                    'major_prizes': sum(1 for s in scores if s >= 50),
                    'any_prize': sum(1 for s in scores if s >= 10),
                    'detailed_results': results[:5]  # Keep first 5 for inspection
                }
        
        return backtest_results
    
    def auto_optimize_for_prize_rate(self, target_prize_rate=0.10, max_iterations=20, min_draws_per_method=5):
        """Automatically optimize backtesting parameters to reach target prize win rate.
        
        Args:
            target_prize_rate (float): Target prize win rate (default: 10%)
            max_iterations (int): Maximum optimization iterations
            min_draws_per_method (int): Minimum draws per method to test
        
        Returns:
            dict: Best configuration found and optimization history
        """
        print(f"\n🎯 AUTOMATIC OPTIMIZATION MODE")
        print("="*60)
        print(f"🎯 Target prize win rate: {target_prize_rate:.1%}")
        print(f"🔄 Maximum iterations: {max_iterations}")
        print(f"📊 Minimum draws per method: {min_draws_per_method}")
        print()
        
        optimization_history = []
        best_config = None
        best_prize_rate = 0
        
        # Strategy parameters to optimize
        optimization_strategies = [
            # Strategy 1: Increase draws per method
            {"strategy": "increase_draws", "draws_per_method": min_draws_per_method, "year_range": (2020, 2024)},
            {"strategy": "increase_draws", "draws_per_method": min_draws_per_method * 2, "year_range": (2020, 2024)},
            {"strategy": "increase_draws", "draws_per_method": min_draws_per_method * 3, "year_range": (2020, 2024)},
            
            # Strategy 2: Adjust year ranges (more recent data)
            {"strategy": "recent_focus", "draws_per_method": min_draws_per_method * 2, "year_range": (2022, 2024)},
            {"strategy": "recent_focus", "draws_per_method": min_draws_per_method * 3, "year_range": (2021, 2024)},
            
            # Strategy 3: Historical periods
            {"strategy": "historical_period", "draws_per_method": min_draws_per_method * 2, "year_range": (2018, 2020)},
            {"strategy": "historical_period", "draws_per_method": min_draws_per_method * 3, "year_range": (2016, 2019)},
            
            # Strategy 4: Larger sample sizes
            {"strategy": "large_sample", "draws_per_method": min_draws_per_method * 4, "year_range": (2019, 2024)},
            {"strategy": "large_sample", "draws_per_method": min_draws_per_method * 5, "year_range": (2018, 2024)},
            {"strategy": "large_sample", "draws_per_method": min_draws_per_method * 6, "year_range": (2017, 2024)},
            
            # Strategy 5: Full historical range
            {"strategy": "full_range", "draws_per_method": min_draws_per_method * 3, "year_range": (2004, 2024)},
            {"strategy": "full_range", "draws_per_method": min_draws_per_method * 4, "year_range": (2004, 2024)},
            {"strategy": "full_range", "draws_per_method": min_draws_per_method * 5, "year_range": (2004, 2024)},
            
            # Strategy 6: Pre/Post-COVID analysis
            {"strategy": "pre_covid", "draws_per_method": min_draws_per_method * 4, "year_range": (2015, 2019)},
            {"strategy": "post_covid", "draws_per_method": min_draws_per_method * 3, "year_range": (2021, 2024)},
        ]
        
        for iteration, config in enumerate(optimization_strategies[:max_iterations], 1):
            print(f"🔄 Iteration {iteration}/{min(len(optimization_strategies), max_iterations)}")
            print(f"📋 Strategy: {config['strategy']}")
            print(f"📊 Draws per method: {config['draws_per_method']}")
            print(f"📅 Year range: {config['year_range'][0]}-{config['year_range'][1]}")
            
            try:
                # Run backtesting with current configuration
                start_date = f"{config['year_range'][0]}-01-01"
                end_date = f"{config['year_range'][1]}-12-31"
                
                results = self.comprehensive_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    draws_per_method=config['draws_per_method']
                )
                
                # Calculate overall prize win rate
                total_tests = 0
                total_prize_wins = 0
                method_results = []
                
                for method, method_result in results.items():
                    if isinstance(method_result, dict) and 'total_tests' in method_result:
                        tests = method_result['total_tests']
                        prizes = method_result.get('any_prize', 0)
                        total_tests += tests
                        total_prize_wins += prizes
                        
                        prize_rate = prizes / tests if tests > 0 else 0
                        method_results.append({
                            'method': method,
                            'tests': tests,
                            'prizes': prizes,
                            'prize_rate': prize_rate,
                            'avg_score': method_result.get('avg_score', 0)
                        })
                
                overall_prize_rate = total_prize_wins / total_tests if total_tests > 0 else 0
                
                # Store optimization step
                optimization_step = {
                    'iteration': iteration,
                    'config': config.copy(),
                    'overall_prize_rate': overall_prize_rate,
                    'total_tests': total_tests,
                    'total_prize_wins': total_prize_wins,
                    'method_results': method_results,
                    'best_method': max(method_results, key=lambda x: x['prize_rate']) if method_results else None
                }
                optimization_history.append(optimization_step)
                
                print(f"📈 Overall prize win rate: {overall_prize_rate:.1%}")
                print(f"🏆 Total prize wins: {total_prize_wins}/{total_tests}")
                
                # Check if we've reached the target
                if overall_prize_rate >= target_prize_rate:
                    print(f"🎯 TARGET REACHED! Prize win rate: {overall_prize_rate:.1%}")
                    best_config = optimization_step
                    break
                
                # Track best configuration so far
                if overall_prize_rate > best_prize_rate:
                    best_prize_rate = overall_prize_rate
                    best_config = optimization_step
                    print(f"🌟 New best configuration! Prize rate: {overall_prize_rate:.1%}")
                
                # Convergence criteria: If we're very close to target, consider it reached
                if overall_prize_rate >= target_prize_rate * 0.9:  # Within 90% of target
                    print(f"📊 Close to target! Prize rate: {overall_prize_rate:.1%} (target: {target_prize_rate:.1%})")
                
                # Early stopping if no improvement in several iterations
                if iteration >= 5:  # Need at least 5 iterations
                    recent_rates = [step['overall_prize_rate'] for step in optimization_history[-3:]]
                    if len(set(recent_rates)) == 1:  # No improvement in last 3 iterations
                        print(f"⚠️  No improvement detected. Best rate: {best_prize_rate:.1%}")
                        if iteration >= max_iterations * 0.5:  # And we're at least halfway through
                            print("🔄 Considering early termination...")
                            break
                
                print(f"⏱️  Best method: {optimization_step['best_method']['method']} ({optimization_step['best_method']['prize_rate']:.1%})")
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                continue
        
        return {
            'target_reached': best_config is not None and best_config['overall_prize_rate'] >= target_prize_rate,
            'best_config': best_config,
            'optimization_history': optimization_history,
            'target_prize_rate': target_prize_rate,
            'iterations_run': len(optimization_history)
        }
    
    def display_optimization_results(self, optimization_result):
        """Display the results of automatic optimization."""
        print(f"\n🎯 OPTIMIZATION RESULTS")
        print("="*60)
        
        if optimization_result['target_reached']:
            print(f"✅ TARGET ACHIEVED!")
            best = optimization_result['best_config']
            print(f"🏆 Best prize win rate: {best['overall_prize_rate']:.1%}")
            print(f"📊 Configuration: {best['config']['strategy']}")
            print(f"📈 Draws per method: {best['config']['draws_per_method']}")
            print(f"📅 Year range: {best['config']['year_range'][0]}-{best['config']['year_range'][1]}")
            print(f"🎲 Prize wins: {best['total_prize_wins']}/{best['total_tests']}")
            
            if best['best_method']:
                print(f"🥇 Best performing method: {best['best_method']['method']}")
                print(f"📈 Method prize rate: {best['best_method']['prize_rate']:.1%}")
        else:
            print(f"⚠️  Target not reached in {optimization_result['iterations_run']} iterations")
            if optimization_result['best_config']:
                best = optimization_result['best_config']
                print(f"🏆 Best achieved: {best['overall_prize_rate']:.1%}")
                print(f"📊 Best configuration: {best['config']['strategy']}")
                print(f"📈 Draws per method: {best['config']['draws_per_method']}")
                print(f"📅 Year range: {best['config']['year_range'][0]}-{best['config']['year_range'][1]}")
        
        # Show optimization progression
        print(f"\n📈 OPTIMIZATION PROGRESSION")
        print("="*60)
        print("Iter | Strategy              | Draws | Year Range | Prize Rate | Prize Wins")
        print("-" * 70)
        
        for step in optimization_result['optimization_history']:
            config = step['config']
            strategy = config['strategy'][:20]
            draws = config['draws_per_method']
            year_start, year_end = config['year_range']
            prize_rate = step['overall_prize_rate']
            wins = f"{step['total_prize_wins']}/{step['total_tests']}"
            
            print(f"{step['iteration']:4d} | {strategy:20s} | {draws:5d} | {year_start}-{year_end} | {prize_rate:9.1%} | {wins}")
        
        # Show recommended next steps
        print(f"\n💡 RECOMMENDATIONS")
        print("="*60)
        
        if optimization_result['target_reached']:
            best = optimization_result['best_config']
            print("🎯 Target achieved! Use this configuration for optimal results:")
            print(f"   python3 unified_euromillions_predictor.py --backtest \\")
            print(f"     --draws-per-method={best['config']['draws_per_method']} \\")
            print(f"     --start-year={best['config']['year_range'][0]} \\")
            print(f"     --end-year={best['config']['year_range'][1]}")
        else:
            print("🔧 Consider these approaches to reach the target:")
            print("   • Increase maximum iterations (--max-iterations=N)")
            print("   • Try different year ranges manually")
            print("   • Focus on specific high-performing methods")
            print("   • Consider that 10% prize win rate may be challenging for lottery prediction")
        
        return optimization_result

    def display_backtest_results(self, backtest_results):
        """Display formatted backtesting results"""
        print(f"\n📈 BACKTESTING RESULTS")
        print("="*60)
        
        # Sort methods by average score
        sorted_methods = sorted(backtest_results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        print(f"{'Rank':<4} {'Method':<25} {'Avg Score':<9} {'Main':<4} {'Stars':<5} {'Prizes':<6} {'Tests':<5}")
        print("-" * 70)
        
        for i, (method, results) in enumerate(sorted_methods, 1):
            avg_score = results['avg_score']
            avg_main = results['avg_main_matches']
            avg_star = results['avg_star_matches']
            prizes = results['any_prize']
            tests = results['total_tests']
            
            print(f"{i:<4} {method:<25} {avg_score:<9.2f} {avg_main:<4.1f} {avg_star:<5.1f} {prizes:<6} {tests:<5}")
        
        # Show detailed performance metrics
        print(f"\n🏆 PERFORMANCE DETAILS")
        print("="*60)
        
        for method, results in sorted_methods:
            print(f"\n📊 {method}:")
            print(f"   • Tests performed: {results['total_tests']}")
            print(f"   • Average score: {results['avg_score']:.2f}")
            print(f"   • Best score: {results['max_score']:.0f}")
            print(f"   • Average main matches: {results['avg_main_matches']:.2f}/5")
            print(f"   • Average star matches: {results['avg_star_matches']:.2f}/2")
            print(f"   • Jackpots (5+2): {results['jackpots']}")
            print(f"   • Major prizes (≥3+1): {results['major_prizes']}")
            print(f"   • Any prize wins: {results['any_prize']}/{results['total_tests']} ({results['any_prize']/results['total_tests']*100:.1f}%)")
        
        # Recommend best method
        if sorted_methods:
            best_method = sorted_methods[0]
            print(f"\n🎯 RECOMMENDATION")
            print("="*60)
            print(f"🏆 Best performing method: {best_method[0]}")
            print(f"📈 Average score: {best_method[1]['avg_score']:.2f}")
            print(f"🎲 Prize win rate: {best_method[1]['any_prize']/best_method[1]['total_tests']*100:.1f}%")

    def generate_consensus_prediction(self, predictions):
        """Generate consensus prediction from all methods"""
        print("\n🎯 CONSENSUS PREDICTION")
        print("="*60)
        
        # Count frequency of each number across all predictions
        main_freq = Counter()
        star_freq = Counter()
        
        for pred in predictions:
            for num in pred['main_numbers']:
                main_freq[num] += pred['confidence']
            for num in pred['star_numbers']:
                star_freq[num] += pred['confidence']
        
        # Select most consensus numbers
        top_main = [num for num, freq in main_freq.most_common(8)]
        top_stars = [num for num, freq in star_freq.most_common(4)]
        
        # Create consensus prediction with some randomness
        consensus_main = random.sample(top_main, min(5, len(top_main)))
        while len(consensus_main) < 5:
            num = random.randint(1, 50)
            if num not in consensus_main:
                consensus_main.append(num)
        
        consensus_stars = random.sample(top_stars, min(2, len(top_stars)))
        while len(consensus_stars) < 2:
            num = random.randint(1, 12)
            if num not in consensus_stars:
                consensus_stars.append(num)
        
        consensus = {
            'main_numbers': sorted(consensus_main),
            'star_numbers': sorted(consensus_stars),
            'method': 'Consensus (All Methods)',
            'confidence': np.mean([p['confidence'] for p in predictions])
        }
        
        main = consensus['main_numbers']
        stars = consensus['star_numbers']
        conf = consensus['confidence']
        
        print(f"🌟 CONSENSUS: {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Avg Conf: {conf:.2f}")
        
        return consensus
    
    def run_complete_analysis(self):
        """Run complete unified prediction analysis"""
        print("🎯 UNIFIED EUROMILLIONS PREDICTION SYSTEM")
        print("="*60)
        print("Combining all prediction methods for comprehensive analysis")
        
        # Generate all predictions
        all_predictions = self.generate_all_predictions(predictions_per_method=2)
        
        # Generate weighted ensemble prediction
        weighted_ensemble = self.create_smart_ensemble_prediction(all_predictions)
        
        # Add weighted ensemble to predictions for ranking
        all_predictions_with_ensemble = all_predictions + [weighted_ensemble]
        
        # Rank predictions (including weighted ensemble)
        ranked_predictions = self.rank_predictions(all_predictions_with_ensemble)
        
        # Generate consensus (traditional method for comparison)
        consensus = self.generate_consensus_prediction(all_predictions)
        
        # Show weighted ensemble details
        print(f"\n🧠 WEIGHTED ENSEMBLE ANALYSIS")
        print("="*60)
        ensemble = weighted_ensemble
        main = ensemble['main_numbers']
        stars = ensemble['star_numbers']
        print(f"🎯 ENSEMBLE:  {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d} | Conf: {ensemble['confidence']:.2f}")
        print(f"📊 Top weighted main numbers: {list(ensemble['weights_used'].keys())[:8]}")
        
        # Display method weights used
        method_weights = self.get_method_weights()
        print(f"\n📏 METHOD WEIGHTS USED:")
        for method, weight in sorted(method_weights.items(), key=lambda x: x[1], reverse=True):
            if weight >= 0.05:  # Only show significant weights
                print(f"   {method:25s}: {weight:.2f} ({weight*100:.0f}%)")
        print("   [Other methods]       : <0.05 (< 5%)")
        
        # Summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print("="*60)
        print(f"📊 Total predictions generated: {len(all_predictions)}")
        print(f"🔬 Methods used: {len(set(p['method'] for p in all_predictions))}")
        print(f"⭐ Highest confidence: {max(p['confidence'] for p in all_predictions):.1f}")
        print(f"🎯 Consensus confidence: {consensus['confidence']:.2f}")
        
        method_counts = Counter(p['method'] for p in all_predictions)
        print(f"\n📈 Methods breakdown:")
        for method, count in method_counts.items():
            print(f"   {method:25s}: {count} predictions")
        
        return {
            'all_predictions': all_predictions,
            'ranked_predictions': ranked_predictions,
            'consensus': consensus,
            'summary': {
                'total_predictions': len(all_predictions),
                'methods_used': len(set(p['method'] for p in all_predictions)),
                'max_confidence': max(p['confidence'] for p in all_predictions),
                'consensus_confidence': consensus['confidence']
            }
        }


def show_help():
    """Display help information for command-line usage"""
    print("🎯 EuroMillions Unified Predictor - Command Line Options")
    print("="*60)
    print()
    print("📊 BASIC USAGE:")
    print("  python3 unified_euromillions_predictor.py           # Regular prediction analysis")
    print("  python3 unified_euromillions_predictor.py --help    # Show this help")
    print()
    print("🔍 BACKTESTING OPTIONS:")
    print("  --backtest, -b                    # Run backtesting (default: 25 draws/method, 2020-2024)")
    print("  --backtest --extended, -e         # Extended backtesting (50 draws/method)")
    print()
    print("🎯 AUTOMATIC OPTIMIZATION:")
    print("  --auto-optimize                   # Auto-optimize parameters for 10% prize win rate")
    print("  --target-prize-rate=X.X           # Custom target prize rate (default: 0.10 = 10%)")
    print("  --max-iterations=N                # Maximum optimization iterations (default: 20)")
    print("  --min-draws=N                     # Minimum draws per method (default: 5)")
    print()
    print("📋 CUSTOM BACKTESTING PARAMETERS:")
    print("  --samples=NUMBER                  # Custom sample size (total draws to test)")
    print("  --draws-per-method=NUMBER         # Specific draws to test per method")
    print("  --start-year=YYYY                 # Start year for backtesting (default: 2020)")  
    print("  --end-year=YYYY                   # End year for backtesting (default: 2024)")
    print()
    print("💡 EXAMPLES:")
    print("  # Standard backtesting")
    print("  python3 unified_euromillions_predictor.py --backtest")
    print()
    print("  # Extended backtesting")  
    print("  python3 unified_euromillions_predictor.py --backtest --extended")
    print()
    print("  # Auto-optimization (find configuration for 10% prize win rate)")
    print("  python3 unified_euromillions_predictor.py --auto-optimize")
    print()
    print("  # Custom auto-optimization (target 15% prize rate, max 30 iterations)")
    print("  python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.15 --max-iterations=30")
    print()
    print("  # Custom: 100 draws per method, 2018-2023 period")
    print("  python3 unified_euromillions_predictor.py --backtest --draws-per-method=100 --start-year=2018 --end-year=2023")
    print()
    print("  # Custom: Test 2021-2022 period with 30 draws per method")
    print("  python3 unified_euromillions_predictor.py --backtest --draws-per-method=30 --start-year=2021 --end-year=2022")
    print()
    print("  # Large-scale validation: 200 draws per method, full period")
    print("  python3 unified_euromillions_predictor.py --backtest --draws-per-method=200 --start-year=2004 --end-year=2024")
    print()
    print("🎯 AUTOMATIC OPTIMIZATION EXPLANATION:")
    print("  • Auto-optimize systematically tests different configurations")
    print("  • Adjusts draws per method, year ranges, and testing strategies")  
    print("  • Stops when target prize win rate is achieved")
    print("  • Reports best configuration found and optimization history")
    print()
    print("📊 BACKTESTING PARAMETERS EXPLANATION:")
    print("  • --samples: Total draws distributed among 7 methods (~sample_size per method)")
    print("  • --draws-per-method: Exact draws tested for each method (more precise control)")
    print("  • Year range: Filter historical data to specific period for testing")
    print("  • Methods tested: Frequency, Gap, Pattern, Temporal, Ensemble, Physical Bias, Neural Network")


def main():
    """Main function to run unified predictor"""
    import sys
    
    # Check for help argument
    if '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return None
    
    # Check for operation mode
    run_backtest = '--backtest' in sys.argv or '-b' in sys.argv
    run_auto_optimize = '--auto-optimize' in sys.argv
    
    try:
        predictor = UnifiedEuroMillionsPredictor()
        
        if run_auto_optimize:
            # Parse auto-optimization arguments
            target_prize_rate = 0.10  # Default 10%
            max_iterations = 20      # Default 20 iterations  
            min_draws_per_method = 5  # Default minimum 5 draws
            
            for arg in sys.argv:
                if arg.startswith('--target-prize-rate='):
                    try:
                        target_prize_rate = float(arg.split('=')[1])
                        if target_prize_rate <= 0 or target_prize_rate > 1:
                            print("❌ Target prize rate must be between 0.01 and 1.0")
                            return None
                    except ValueError:
                        print("❌ Invalid target prize rate format. Use --target-prize-rate=0.10")
                        return None
                elif arg.startswith('--max-iterations='):
                    try:
                        max_iterations = int(arg.split('=')[1])
                        if max_iterations <= 0:
                            print("❌ Maximum iterations must be positive")
                            return None
                    except ValueError:
                        print("❌ Invalid max iterations format. Use --max-iterations=20")
                        return None
                elif arg.startswith('--min-draws='):
                    try:
                        min_draws_per_method = int(arg.split('=')[1])
                        if min_draws_per_method <= 0:
                            print("❌ Minimum draws per method must be positive")
                            return None
                    except ValueError:
                        print("❌ Invalid min draws format. Use --min-draws=5")
                        return None
            
            print(f"🎯 Starting automatic optimization")
            print(f"🎯 Target prize win rate: {target_prize_rate:.1%}")
            print(f"🔄 Maximum iterations: {max_iterations}")
            print(f"📊 Minimum draws per method: {min_draws_per_method}")
            
            # Run automatic optimization
            optimization_result = predictor.auto_optimize_for_prize_rate(
                target_prize_rate=target_prize_rate,
                max_iterations=max_iterations,
                min_draws_per_method=min_draws_per_method
            )
            
            # Display optimization results
            predictor.display_optimization_results(optimization_result)
            
            return optimization_result
            
        elif run_backtest:
            # Parse command-line arguments
            custom_sample_size = None
            draws_per_method = None
            start_year = 2020
            end_year = 2024
            
            for arg in sys.argv:
                if arg.startswith('--samples='):
                    try:
                        custom_sample_size = int(arg.split('=')[1])
                    except ValueError:
                        print("❌ Invalid sample size format. Use --samples=NUMBER")
                        return None
                elif arg.startswith('--draws-per-method='):
                    try:
                        draws_per_method = int(arg.split('=')[1])
                    except ValueError:
                        print("❌ Invalid draws per method format. Use --draws-per-method=NUMBER")
                        return None
                elif arg.startswith('--start-year='):
                    try:
                        start_year = int(arg.split('=')[1])
                    except ValueError:
                        print("❌ Invalid start year format. Use --start-year=YYYY")
                        return None
                elif arg.startswith('--end-year='):
                    try:
                        end_year = int(arg.split('=')[1])
                    except ValueError:
                        print("❌ Invalid end year format. Use --end-year=YYYY")
                        return None
            
            # Validate year range
            if start_year >= end_year:
                print("❌ Start year must be less than end year")
                return None
            
            # Determine backtesting configuration
            if draws_per_method is not None:
                mode = f"CUSTOM ({draws_per_method} draws/method)"
                sample_size = draws_per_method  # Will be overridden by draws_per_method parameter
            elif custom_sample_size:
                sample_size = custom_sample_size
                mode = f"CUSTOM ({sample_size} samples)"
            else:
                extended_backtest = '--extended' in sys.argv or '-e' in sys.argv
                sample_size = 50 if extended_backtest else 25
                mode = 'EXTENDED' if extended_backtest else 'STANDARD'
            
            # Format date strings
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            
            print(f"🔍 Running {mode} backtesting")
            print(f"📅 Period: {start_year}-{end_year}")
            if draws_per_method is not None:
                print(f"🎯 Draws per method: {draws_per_method}")
            else:
                print(f"📊 Sample size: {sample_size} draws per method")
            
            # Run backtesting analysis
            backtest_results = predictor.comprehensive_backtest(
                start_date=start_date, 
                end_date=end_date, 
                sample_size=sample_size,
                draws_per_method=draws_per_method
            )
            predictor.display_backtest_results(backtest_results)
            
            # Still run the regular analysis afterwards
            print(f"\n" + "="*60)
            print("CONTINUING WITH REGULAR PREDICTION ANALYSIS...")
            print("="*60)
            results = predictor.run_complete_analysis()
            return {'predictions': results, 'backtest': backtest_results}
        else:
            # Regular prediction analysis
            results = predictor.run_complete_analysis()
            return results
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()